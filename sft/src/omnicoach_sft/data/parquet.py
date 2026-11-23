"""Shared Agent Parquet: frozen media identity, bounded export and zero-copy views."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import shutil
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from omnicoach_sft.io import (
    _embedding_work_units,
    plan_grouped_parquet_layout,
    read_json,
    shard_rows,
    write_json,
)

AGENT_SHARED_MEDIA_LAYOUT = "agent_turn_shared_media_v1"
AGENT_SHARED_MEDIA_SCHEMA_VERSION = "agent_turn_shared_media_v1"
TURN_TABLE_DIR = "turns"

IMAGE_LIST_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("bytes", pa.large_binary()),
            pa.field("path", pa.string()),
        ]
    )
)
AUDIO_LIST_TYPE = pa.list_(pa.large_binary())
MEDIA_TABLE_SCHEMA = pa.schema(
    [
        pa.field("media_id", pa.string(), nullable=False),
        pa.field("images", IMAGE_LIST_TYPE, nullable=False),
        pa.field("audios", AUDIO_LIST_TYPE, nullable=False),
    ]
)

PrepareRowsFn = Callable[[list[dict[str, Any]], Path], list[dict[str, Any]]]
ValidateRowFn = Callable[[dict[str, Any], str], Any]
FinalizeArtifactFn = Callable[[Path], Any]


@dataclass(frozen=True)
class AgentParquetConfig:
    output: str | Path
    source_id: str
    schema: pa.Schema
    num_shards: int = 1
    row_group_size: int = 100
    group_ids: list[str] | None = None
    workers: int = 1
    cache_dir: str | Path | None = None
    resume: bool = False
    partial: str | Path | None = None
    prepare_rows: PrepareRowsFn | None = None
    validate_row: ValidateRowFn | None = None
    metadata: dict[str, Any] | None = None
    contract: dict[str, Any] | None = None
    identity: dict[str, Any] | None = None
    worker_index: int = 0
    worker_count: int = 1
    build_only: bool = False
    overwrite: bool = False
    finalize_artifact: FinalizeArtifactFn | None = None


@dataclass(frozen=True)
class _ShardWriteContext:
    partial: Path
    schema: pa.Schema
    source_id: str
    prepare_rows: PrepareRowsFn | None
    validate_row: ValidateRowFn | None
    cache_dir: Path
    batch_rows: int
    payload_workers: int


def _bytes(value: Any, *, context: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    raise ValueError(f"{context} must be bytes")


def agent_media_id(*, source_id: str, images: list[Any], audios: list[Any]) -> str | None:
    """Return the content-addressed id of one ordered Agent media bundle."""

    if not images and not audios:
        return None
    digest = hashlib.sha256(b"agent_turn_shared_media_v1\0")
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            raise ValueError(f"images[{index}] must be an embedded image object")
        payload = _bytes(image.get("bytes"), context=f"images[{index}].bytes")
        digest.update(b"image\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        path = image.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"images[{index}].path must be a non-empty string")
        expected_path = f"{hashlib.sha256(payload).hexdigest()}.webp"
        if path != expected_path:
            raise ValueError(f"images[{index}].path is not the WebP content digest: {path!r}")
    for index, audio in enumerate(audios):
        payload = _bytes(audio, context=f"audios[{index}]")
        digest.update(b"audio\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"{source_id}:media:{digest.hexdigest()}"


def turn_schema(schema: pa.Schema, metadata: dict[str, Any] | None = None) -> pa.Schema:
    fields = []
    for field in schema:
        if field.name == "media_id":
            continue
        dtype = {"images": IMAGE_LIST_TYPE, "audios": AUDIO_LIST_TYPE}.get(field.name)
        fields.append(pa.field(field.name, dtype, nullable=field.nullable) if dtype is not None else field)
        if field.name == "audios":
            fields.append(pa.field("media_id", pa.string()))
    if "audios" not in schema.names:
        fields.append(pa.field("media_id", pa.string()))
    attributes = dict(schema.metadata or {})
    attributes.update({str(key).encode(): str(value).encode() for key, value in (metadata or {}).items()})
    return pa.schema(fields, metadata=attributes)


def part_record(path: Path) -> dict[str, Any]:
    with pq.ParquetFile(path) as parquet:
        return {
            "file": path.name,
            "rows": parquet.metadata.num_rows,
            "row_groups": parquet.metadata.num_row_groups,
            "bytes": path.stat().st_size,
        }


def _identity(
    rows: list[dict[str, Any]],
    source_id: str,
    schema: pa.Schema,
    layout: list[list[int]],
    recipe: dict[str, Any],
) -> dict[str, Any]:
    digest = hashlib.sha256()
    paths: set[str] = set()

    def encode(value: Any) -> dict[str, str]:
        if isinstance(value, (bytes, bytearray, memoryview)):
            return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
        raise TypeError(type(value).__name__)

    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False, default=encode).encode())
        for column in ("images", "audios"):
            for value in row.get(column) or []:
                if isinstance(value, dict) and value.get("bytes") is None:
                    value = value.get("path")
                if isinstance(value, str) and value not in paths:
                    paths.add(value)
                    path = Path(value)
                    if path.is_file():
                        stat = path.stat()
                        digest.update(json.dumps([str(path.resolve()), stat.st_size, stat.st_mtime_ns]).encode())
    return {
        "source_id": source_id,
        "rows_sha256": digest.hexdigest(),
        "schema_sha256": hashlib.sha256(schema.serialize().to_pybytes()).hexdigest(),
        "layout": layout,
        "recipe": recipe,
    }


def _bundle(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[bytes]]:
    images = []
    for value in row.get("images") or []:
        if isinstance(value, dict):
            payload = bytes(value["bytes"]) if value.get("bytes") is not None else Path(value["path"]).read_bytes()
        else:
            payload = Path(value).read_bytes()
        images.append({"bytes": payload, "path": hashlib.sha256(payload).hexdigest() + ".webp"})
    audios = [
        Path(value).read_bytes() if isinstance(value, str) else bytes(value)
        for value in row.get("audios") or []
    ]
    return images, audios


def _write_shard(
    rows: list[dict[str, Any]],
    groups: list[int],
    index: int,
    ctx: _ShardWriteContext,
) -> dict[str, Any]:
    name = f"part-{index:05d}.parquet"
    done = ctx.partial / ".done" / f"{index:05d}.json"
    turn_path, media_path = ctx.partial / "turns" / name, ctx.partial / "media" / name
    if done.exists():
        receipt = read_json(done)
        for path, record in ((turn_path, receipt["turns"]), (media_path, receipt["media"])):
            if part_record(path) != record:
                raise ValueError(f"completed shard changed: {path}")
        return receipt
    work = ctx.partial / f".part-{index:05d}"
    work.mkdir(exist_ok=True)
    temporary_turn, temporary_media = work / "turns.parquet", work / "media.parquet"
    seen: set[str] = set()
    references: set[str] = set()
    loss_values: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    previous_paths: tuple[Any, ...] | None = None
    previous_bundle: tuple[list[dict[str, Any]], list[bytes]] | None = None
    offset = 0
    with (
        ThreadPoolExecutor(max_workers=ctx.payload_workers) as payload_pool,
        pq.ParquetWriter(temporary_turn, ctx.schema, compression="zstd", use_dictionary=True) as turns,
        pq.ParquetWriter(temporary_media, MEDIA_TABLE_SCHEMA, compression="zstd", use_dictionary=True) as media,
    ):
        for count in groups:
            batch = rows[offset : offset + count]
            offset += count
            # Prepare bounded windows even when one indivisible group exceeds
            # the target. The turn row group is assembled from Arrow chunks.
            chunks = []
            for start in range(0, len(batch), ctx.batch_rows):
                window = batch[start : start + ctx.batch_rows]
                prepared = ctx.prepare_rows(window, ctx.cache_dir) if ctx.prepare_rows else window
                if len(prepared) != len(window):
                    raise ValueError("media preparation changed row count")
                # Submit each unique path bundle once, heaviest first. Futures
                # are consumed in input order; at most one window holds bytes.
                keys: list[tuple[Any, ...] | int] = []
                unique: dict[tuple[Any, ...] | int, dict[str, Any]] = {}
                for local_index, row in enumerate(prepared):
                    paths = (*tuple(row.get("images") or ()), None, *tuple(row.get("audios") or ()))
                    key: tuple[Any, ...] | int = (
                        paths if all(value is None or isinstance(value, str) for value in paths) else local_index
                    )
                    keys.append(key)
                    unique.setdefault(key, row)
                bundles = {
                    key: payload_pool.submit(_bundle, row)
                    for key, row in sorted(unique.items(), key=lambda item: -_embedding_work_units(item[1]))
                    if key != previous_paths
                }
                cached_paths, cached_bundle = previous_paths, previous_bundle
                output = []
                for local_index, original in enumerate(prepared):
                    row = dict(original)
                    paths_key = keys[local_index]
                    path_bundle = isinstance(paths_key, tuple)
                    if path_bundle and paths_key == cached_paths and cached_bundle is not None:
                        images, audios = cached_bundle
                    else:
                        images, audios = bundles[paths_key].result()
                    previous_paths = paths_key if path_bundle else None
                    previous_bundle = (images, audios)
                    row.update(images=images, audios=audios)
                    if ctx.validate_row:
                        ctx.validate_row(row, f"shard {index} row {offset - count + start + local_index}")
                    if "omni_num_images" in row and row["omni_num_images"] != len(images):
                        raise ValueError("image count changed during materialization")
                    if "omni_audio_num_samples" in row and len(row["omni_audio_num_samples"]) != len(audios):
                        raise ValueError("audio count changed during materialization")
                    media_id = agent_media_id(source_id=ctx.source_id, images=images, audios=audios)
                    if media_id is not None:
                        references.add(media_id)
                        if media_id not in seen:
                            media.write_table(
                                pa.Table.from_pylist(
                                    [{"media_id": media_id, "images": images, "audios": audios}],
                                    schema=MEDIA_TABLE_SCHEMA,
                                ),
                                row_group_size=1,
                            )
                            seen.add(media_id)
                    row.update(images=[], audios=[], media_id=media_id)
                    output.append(row)
                    totals.update(image=len(images), audio=len(audios))
                    if row.get("messages"):
                        target = row["messages"][-1]
                        roles[target["role"]] += 1
                        if target.get("loss_scale") is not None:
                            loss_values[format(float(target["loss_scale"]), ".17g")] += 1
                chunks.append(pa.Table.from_pylist(output, schema=ctx.schema))
            if chunks:
                table = pa.concat_tables(chunks)
                turns.write_table(table, row_group_size=count)
    if offset != len(rows):
        raise ValueError("shard row plan mismatch")
    os.replace(temporary_media, media_path)
    os.replace(temporary_turn, turn_path)
    receipt = {
        "turns": part_record(turn_path),
        "media": part_record(media_path),
        "media_ids": sorted(seen),
        "references": sorted(references),
        "loss_values": dict(loss_values),
        "terminal_roles": dict(roles),
        "media_totals": dict(totals),
    }
    if receipt["turns"]["rows"] != len(rows):
        raise ValueError("written shard row count differs")
    write_json(done, receipt, atomic=True)
    shutil.rmtree(work)
    return receipt


def _deduplicate_media(partial: Path) -> set[str]:
    seen: set[str] = set()
    for path in sorted((partial / "media").glob("part-*.parquet")):
        with pq.ParquetFile(path) as media:
            ids = media.read(columns=["media_id"])["media_id"].to_pylist()
            keep = []
            for index, media_id in enumerate(ids):
                if media_id not in seen:
                    seen.add(media_id)
                    keep.append(index)
            if len(keep) == len(ids):
                continue
            temporary = path.with_suffix(".partial")
            with pq.ParquetWriter(temporary, MEDIA_TABLE_SCHEMA, compression="zstd", use_dictionary=True) as writer:
                for index in keep:
                    writer.write_table(media.read_row_group(index), row_group_size=1)
            # Receipt is the commit marker for finalization too. Rename and
            # receipt update are recovered by re-running this idempotent scan.
            os.replace(temporary, path)
    return seen


def _resolve_config(config: AgentParquetConfig | None, kwargs: dict[str, Any]) -> AgentParquetConfig:
    if config is not None and kwargs:
        raise ValueError("pass either AgentParquetConfig or keyword arguments, not both")
    if config is not None:
        return config
    return AgentParquetConfig(**kwargs)


def write_agent_parquet(
    rows: Iterable[dict[str, Any]],
    config: AgentParquetConfig | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Write normalized rows as shared Parquet with deterministic shard receipts.

    Accepts either an ``AgentParquetConfig`` instance or equivalent keyword
    arguments. ``prepare_rows(batch, cache_dir)`` may encode source media and
    add statistics; ``validate_row(embedded_row, context)`` provides
    source-specific checks. Only a bounded row window holds Python media bytes.
    """

    cfg = _resolve_config(config, kwargs)
    row_list = list(rows)
    output = Path(cfg.output).resolve()
    if not row_list or min(cfg.num_shards, cfg.row_group_size, cfg.workers, cfg.worker_count) < 1:
        raise ValueError("nonempty rows and positive shard/group/worker counts required")
    if not 0 <= cfg.worker_index < cfg.worker_count:
        raise ValueError("invalid shard worker partition")
    if output.exists() and not cfg.overwrite:
        raise FileExistsError(output)
    schema = turn_schema(cfg.schema, cfg.metadata)
    message_fields = schema.field("messages").type.value_type
    if (
        "loss_scale" in [field.name for field in message_fields]
        and message_fields.field("loss_scale").type != pa.float64()
    ):
        raise ValueError("message loss_scale must be float64")
    if cfg.group_ids is not None:
        if len(cfg.group_ids) != len(row_list):
            raise ValueError("group identity count differs from rows")
        layouts = plan_grouped_parquet_layout(
            cfg.group_ids,
            row_group_size=cfg.row_group_size,
            num_shards=cfg.num_shards,
        )
    else:
        partitions = shard_rows(row_list, cfg.num_shards, weights=[_embedding_work_units(row) for row in row_list])
        layouts = [
            [min(cfg.row_group_size, len(part) - start) for start in range(0, len(part), cfg.row_group_size)]
            for part in partitions
        ]
    partial = Path(cfg.partial).resolve() if cfg.partial else output.with_name(f".{output.name}.partial")
    if partial.parent != output.parent or partial == output:
        raise ValueError("partial must be beside and distinct from output")
    plan = _identity(row_list, cfg.source_id, schema, layouts, {"identity": cfg.identity, "contract": cfg.contract})
    state = partial / "build_state.json"
    partial.mkdir(parents=True, exist_ok=True)
    with (partial / ".plan.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if state.exists():
            if (not cfg.resume and cfg.worker_count == 1) or read_json(state) != plan:
                raise ValueError("resume input identity/schema/layout differs")
        elif any(path.name != ".plan.lock" for path in partial.iterdir()):
            raise ValueError("partial has no validated build identity")
        else:
            write_json(state, plan, atomic=True)
        for directory in (partial / "turns", partial / "media", partial / ".done"):
            directory.mkdir(exist_ok=True)
    cache = Path(cfg.cache_dir) if cfg.cache_dir else partial / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[int, list[int], list[dict[str, Any]]]] = []
    offset = 0
    for index, groups in enumerate(layouts):
        count = sum(groups)
        if index % cfg.worker_count == cfg.worker_index:
            jobs.append((index, groups, row_list[offset : offset + count]))
        offset += count
    # Lighter shards are paired in the executor queue; each worker owns one
    # shard and its temporary files, and results are committed in source order.
    jobs.sort(key=lambda job: -sum(_embedding_work_units(row) for row in job[2]))
    active_workers = min(cfg.workers, max(1, len(jobs)))
    shard_ctx = _ShardWriteContext(
        partial=partial,
        schema=schema,
        source_id=cfg.source_id,
        prepare_rows=cfg.prepare_rows,
        validate_row=cfg.validate_row,
        cache_dir=cache,
        batch_rows=max(1, cfg.row_group_size),
        payload_workers=max(1, cfg.workers // active_workers),
    )

    def build(job: tuple[int, list[int], list[dict[str, Any]]]) -> dict[str, Any]:
        index, groups, shard = job
        return _write_shard(shard, groups, index, shard_ctx)

    finalizing = partial / ".finalizing"
    if finalizing.exists():
        jobs = []
    with ThreadPoolExecutor(max_workers=active_workers) as executor:
        list(executor.map(build, jobs))
    if cfg.build_only:
        return {"rows": sum(sum(job[1]) for job in jobs), "partial": str(partial), "build_only": True}
    receipts = [read_json(partial / ".done" / f"{index:05d}.json") for index in range(len(layouts))]
    for index, receipt in enumerate(receipts):
        turn = partial / "turns" / receipt["turns"]["file"]
        with pq.ParquetFile(turn) as parquet:
            if (
                part_record(turn) != receipt["turns"]
                or parquet.schema_arrow != schema
                or parquet.schema_arrow.metadata != schema.metadata
                or [parquet.metadata.row_group(group).num_rows for group in range(parquet.metadata.num_row_groups)]
                != layouts[index]
            ):
                raise ValueError(f"completed turn shard differs from plan: {turn}")
        media = partial / "media" / receipt["media"]["file"]
        if not finalizing.exists() and part_record(media) != receipt["media"]:
            raise ValueError(f"completed media shard changed: {media}")
    if not finalizing.exists():
        finalizing.touch()
    media_ids = _deduplicate_media(partial)
    references = set().union(*(set(item["references"]) for item in receipts))
    if not references <= media_ids or media_ids != set().union(*(set(item["media_ids"]) for item in receipts)):
        raise ValueError("turns reference missing media")
    tables = {
        table: [part_record(path) for path in sorted((partial / table).glob("part-*.parquet"))]
        for table in ("turns", "media")
    }
    total_rows = sum(record["rows"] for record in tables["turns"])
    if total_rows != len(row_list):
        raise ValueError("completed shard set differs from planned rows")
    counts: dict[str, Counter[str]] = {key: Counter() for key in ("loss_values", "terminal_roles", "media_totals")}
    for item in receipts:
        for key, counter in counts.items():
            counter.update(item[key])
    validation = {
        "ok": True,
        "rows": total_rows,
        "turn_rows": total_rows,
        "media_rows": len(media_ids),
        "media_row_groups": len(media_ids),
        "embedded_media": False,
        "media_columns": "lazy_embedded_media_parquet",
        "turn_media_columns_empty": True,
        "turn_media_references_valid": True,
        "terminal_roles": dict(counts["terminal_roles"]),
        "media_totals": dict(counts["media_totals"]),
        "loss_scale": {"dtype": "float64", "values": dict(counts["loss_values"])},
    }
    result = {
        **(cfg.contract or {}),
        "schema_version": AGENT_SHARED_MEDIA_LAYOUT,
        "table_layout": AGENT_SHARED_MEDIA_LAYOUT,
        "embedded_media": False,
        "media_columns": "lazy_embedded_media_parquet",
        "source_id": cfg.source_id,
        "columns": schema.names,
        "rows": total_rows,
        "turn_rows": total_rows,
        "media_rows": len(media_ids),
        "tables": tables,
    }
    write_json(partial / "dataset_contract.json", result)
    write_json(partial / "validation.json", validation)
    write_json(
        partial / "manifest.json",
        {
            "pipeline": "agent_shared_writer_v1",
            "lifecycle": "locally_validated",
            "configuration": plan,
            "validation": validation,
            "tables": tables,
        },
    )
    if cfg.finalize_artifact:
        cfg.finalize_artifact(partial)
    if cfg.cache_dir is None:
        shutil.rmtree(cache)
    if output.exists():
        shutil.rmtree(output)
    partial.rename(output)
    shutil.rmtree(output / ".done")
    (output / "build_state.json").unlink()
    (output / ".finalizing").unlink()
    (output / ".plan.lock").unlink()
    return {**validation, "contract": result, "validation": validation, "tables": tables}


def write_agent_view(
    parent: str | Path,
    output: str | Path,
    *,
    select: Iterable[int] | None = None,
    updates: dict[int, dict[str, Any]] | None = None,
    contract: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select physical row indices and replace columns without touching media.

    ``updates`` maps physical indices to changed column values, including exact
    message loss weights. Unselected groups are skipped before Arrow reads;
    unchanged shards use hardlinks. The complete immutable parent media index
    stays available through a directory link, so IDs retain their old identity.
    """

    parent_path, output_path = Path(parent).resolve(), Path(output).resolve()
    source = read_json(parent_path / "dataset_contract.json")
    if contract and contract.get("source_id", source.get("source_id")) != source.get("source_id"):
        raise ValueError("views retain the parent source_id and media identity")
    if metadata and metadata.get("omni.source_id", source.get("source_id")) != source.get("source_id"):
        raise ValueError("views retain the parent source metadata")
    if source.get("table_layout", source.get("schema_version")) != AGENT_SHARED_MEDIA_LAYOUT:
        raise ValueError("views require a shared-media parent")
    if not read_json(parent_path / "validation.json").get("ok"):
        raise ValueError("parent validation failed")
    if output_path.exists():
        raise FileExistsError(output_path)
    selected = None if select is None else set(select)
    replacements = updates or {}
    total = source.get("turn_rows", source["rows"])
    if any(
        not isinstance(index, int) or index < 0 or index >= total
        for index in (set(replacements) | (selected or set()))
    ):
        raise ValueError("physical row index outside parent")
    partial = output_path.with_name(f".{output_path.name}.partial")
    (partial / "turns").mkdir(parents=True)
    try:
        (partial / "media").symlink_to(parent_path / "media", target_is_directory=True)
        offset, rows = 0, 0
        for spec in source["tables"]["turns"]:
            original = parent_path / "turns" / spec["file"]
            target = partial / "turns" / spec["file"]
            with pq.ParquetFile(original) as parquet:
                schema = parquet.schema_arrow
                if metadata:
                    schema = schema.with_metadata(
                        {
                            **(schema.metadata or {}),
                            **{str(key).encode(): str(value).encode() for key, value in metadata.items()},
                        }
                    )
                span = range(offset, offset + parquet.metadata.num_rows)
                whole = selected is None or all(index in selected for index in span)
                changed = any(index in replacements for index in span)
                if whole and not changed and schema.metadata == parquet.schema_arrow.metadata:
                    try:
                        os.link(original, target)
                    except OSError as error:
                        if error.errno != errno.EXDEV:
                            raise
                        shutil.copyfile(original, target)
                    offset += parquet.metadata.num_rows
                    rows += parquet.metadata.num_rows
                    continue
                writer = None
                try:
                    for group in range(parquet.metadata.num_row_groups):
                        count = parquet.metadata.row_group(group).num_rows
                        positions = [i for i in range(count) if selected is None or offset + i in selected]
                        if positions:
                            table = parquet.read_row_group(group, use_threads=False)
                            if len(positions) != count:
                                table = table.take(pa.array(positions, type=pa.int64()))
                            changes = [replacements.get(offset + i, {}) for i in positions]
                            for name in set().union(*(set(change) for change in changes)):
                                if name in {"images", "audios", "media_id", "omni_source_id"}:
                                    raise ValueError("views cannot change media references or payloads")
                                values = table[name].to_pylist()
                                values = [
                                    change.get(name, value) for change, value in zip(changes, values, strict=True)
                                ]
                                table = table.set_column(
                                    schema.get_field_index(name),
                                    schema.field(name),
                                    pa.array(values, type=schema.field(name).type),
                                )
                            if writer is None:
                                writer = pq.ParquetWriter(target, schema, compression="zstd", use_dictionary=True)
                            writer.write_table(
                                table.replace_schema_metadata(schema.metadata),
                                row_group_size=len(positions),
                            )
                            rows += len(positions)
                        offset += count
                finally:
                    if writer is not None:
                        writer.close()
        if not rows:
            raise ValueError("empty Agent view")
        tables = {
            "turns": [part_record(path) for path in sorted((partial / "turns").glob("*.parquet"))],
            "media": source["tables"]["media"],
        }
        result = {
            **source,
            **(contract or {}),
            "rows": rows,
            "turn_rows": rows,
            "schema_version": AGENT_SHARED_MEDIA_LAYOUT,
            "table_layout": AGENT_SHARED_MEDIA_LAYOUT,
            "embedded_media": False,
            "source_id": source.get("source_id"),
            "columns": source["columns"],
            "media_rows": source["media_rows"],
            "parent_artifact": str(parent_path),
            "tables": tables,
        }
        validation = {
            "ok": True,
            "rows": rows,
            "turn_rows": rows,
            "media_rows": source["media_rows"],
            "media_references_valid": True,
            "full_media_rescan": False,
        }
        write_json(partial / "dataset_contract.json", result)
        write_json(partial / "validation.json", validation)
        write_json(partial / "manifest.json", {"parent": str(parent_path), "tables": tables, "validation": validation})
        partial.rename(output_path)
        return result
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def write_agent_manifest(sources: Iterable[str | Path], output: str | Path) -> list[str]:
    """Compose validated shared artifacts without decoding or copying media."""

    paths: list[str] = []
    for source in sources:
        source_path = Path(source).resolve()
        contract = read_json(source_path / "dataset_contract.json")
        if contract.get("table_layout", contract.get("schema_version")) != AGENT_SHARED_MEDIA_LAYOUT:
            raise ValueError(f"not shared Parquet: {source_path}")
        if not read_json(source_path / "validation.json").get("ok"):
            raise ValueError(f"source validation failed: {source_path}")
        paths.append(str(source_path))
    Path(output).write_text(yaml.safe_dump({"dataset": paths}, sort_keys=False), encoding="utf-8")
    return paths
