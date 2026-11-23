"""JSON, JSONL, atomic-write, digest, and Arrow/Parquet I/O helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import Dataset, Features, Image, Sequence as HFSequence, Value

from omnicoach_sft.paths import PROJECT_ROOT

JsonRow = dict[str, Any]
GROUPED_DICTIONARY_PAGESIZE_LIMIT = 1024**3


def require_mapping(payload: Any, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a mapping")
    return payload


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def git_commit(cwd: str | Path = PROJECT_ROOT) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_mapping_json(path: str | Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def iter_jsonl(path: str | Path) -> Iterator[Any]:
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path: str | Path) -> list[Any]:
    return list(iter_jsonl(path))


def read_jsonl_objects(path: str | Path) -> list[dict[str, Any]]:
    """Read an object-per-line JSONL file with line-numbered errors."""

    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            rows.append(row)
    return rows


def atomic_write_bytes(
    path: str | Path,
    payload: bytes,
    *,
    parents: bool = False,
    skip_if_size_matches: bool = False,
) -> None:
    target = Path(path)
    if skip_if_size_matches and target.is_file() and target.stat().st_size == len(payload):
        return
    if parents:
        target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: str | Path, text: str, *, parents: bool = False) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), parents=parents)


def write_json(
    path: str | Path,
    value: Any,
    *,
    sort_keys: bool = False,
    indent: int | None = 2,
    atomic: bool = False,
    parents: bool = False,
    separators: tuple[str, str] | None = None,
) -> None:
    target = Path(path)
    text = json.dumps(value, ensure_ascii=False, sort_keys=sort_keys, indent=indent, separators=separators) + "\n"
    if atomic:
        atomic_write_text(target, text, parents=parents)
        return
    if parents:
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def write_jsonl(
    path: str | Path,
    rows: Iterable[Any],
    *,
    sort_keys: bool = False,
    separators: tuple[str, str] | None = None,
    parents: bool = False,
    mode: str = "w",
) -> int:
    target = Path(path)
    if parents:
        target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open(mode, encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=sort_keys, separators=separators) + "\n")
            count += 1
    return count


def write_jsonl_atomic(rows: Iterable[dict[str, Any]], output_path: str | Path) -> tuple[str, int]:
    """Atomically write canonical JSONL rows and return ``(sha256, byte_size)``."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as handle:
            for row in rows:
                encoded = (canonical_json(row) + "\n").encode("utf-8")
                handle.write(encoded)
                digest.update(encoded)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest(), target.stat().st_size


def write_dataset_parquet(
    dataset: Dataset,
    output_path: str | Path,
    *,
    parquet_metadata: dict[str, str] | None = None,
    row_group_size: int = 100,
) -> None:
    """Write an Arrow-backed dataset while preserving HF/custom metadata."""
    table = dataset.data.table
    if parquet_metadata:
        metadata = dict(table.schema.metadata or {})
        metadata.update({str(key).encode(): str(value).encode() for key, value in parquet_metadata.items()})
        table = table.replace_schema_metadata(metadata)
    pq.write_table(table, output_path, row_group_size=max(1, int(row_group_size)), use_dictionary=True)


def write_dataset_shards(
    dataset: Dataset,
    output_dir: str | Path,
    *,
    num_shards: int,
    row_group_size: int,
    parquet_metadata: dict[str, str] | None = None,
) -> None:
    """Write deterministic contiguous ``part-*.parquet`` shards."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    width = max(5, len(str(max(0, num_shards - 1))))
    for shard_index in range(num_shards):
        shard = dataset.shard(num_shards=num_shards, index=shard_index, contiguous=True)
        write_dataset_parquet(
            shard,
            output_dir / f"part-{shard_index:0{width}d}.parquet",
            parquet_metadata=parquet_metadata,
            row_group_size=row_group_size,
        )


def _embedding_work_units(row: JsonRow) -> int:
    image_tokens = row.get("omni_image_num_tokens")
    image_work = (
        int(image_tokens)
        if isinstance(image_tokens, (int, float)) and not isinstance(image_tokens, bool) and image_tokens > 0
        else len(row.get("images") or [])
    )
    audio_samples = row.get("omni_audio_num_samples")
    audio_work = (
        math.ceil(sum(max(0, int(value)) for value in audio_samples) / 1600)
        if isinstance(audio_samples, list)
        else len(row.get("audios") or [])
    )
    return max(1, image_work + audio_work)


def _contiguous_group_sizes(group_ids: list[str]) -> list[int]:
    if not group_ids:
        raise ValueError("row_group_ids must not be empty")
    sizes: list[int] = []
    seen: set[str] = set()
    current_id = ""
    for index, group_id in enumerate(group_ids):
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"row_group_ids[{index}] must be a non-empty string")
        if index == 0 or group_id != current_id:
            if group_id in seen:
                raise ValueError(f"row group id is not contiguous: {group_id!r}")
            seen.add(group_id)
            sizes.append(1)
            current_id = group_id
        else:
            sizes[-1] += 1
    return sizes


def _pack_atomic_groups(group_sizes: list[int], *, target_rows: int) -> list[int]:
    row_groups: list[int] = []
    pending = 0
    for group_size in group_sizes:
        if pending and pending + group_size > target_rows:
            row_groups.append(pending)
            pending = 0
        pending += group_size
    if pending:
        row_groups.append(pending)
    return row_groups


def _partition_contiguous_sizes(sizes: list[int], *, num_parts: int) -> list[list[int]]:
    num_parts = min(max(1, num_parts), len(sizes))
    parts: list[list[int]] = []
    start = 0
    consumed_rows = 0
    total_rows = sum(sizes)
    for part_index in range(num_parts):
        remaining_parts = num_parts - part_index
        if remaining_parts == 1:
            end = len(sizes)
        else:
            max_end = len(sizes) - (remaining_parts - 1)
            target = (total_rows - consumed_rows) / remaining_parts
            end = start
            part_rows = 0
            while end < max_end:
                next_rows = sizes[end]
                if end > start and part_rows + next_rows > target:
                    if abs(target - (part_rows + next_rows)) < abs(target - part_rows):
                        part_rows += next_rows
                        end += 1
                    break
                part_rows += next_rows
                end += 1
                if part_rows >= target:
                    break
        parts.append(sizes[start:end])
        consumed_rows += sum(sizes[start:end])
        start = end
    return parts


def plan_grouped_parquet_layout(
    group_ids: list[str],
    *,
    row_group_size: int,
    num_shards: int,
) -> list[list[int]]:
    """Plan shards/row groups without splitting any contiguous logical group."""

    atomic_sizes = _contiguous_group_sizes(group_ids)
    row_groups = _pack_atomic_groups(atomic_sizes, target_rows=max(1, int(row_group_size)))
    return _partition_contiguous_sizes(row_groups, num_parts=max(1, int(num_shards)))


def shard_rows(
    rows: list[JsonRow],
    num_shards: int,
    *,
    weights: list[int] | None = None,
) -> list[list[JsonRow]]:
    num_shards = min(max(1, num_shards), len(rows))
    if weights is not None:
        if len(weights) != len(rows):
            raise ValueError("row weights must have the same length as rows")
        if any(isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0 for weight in weights):
            raise ValueError("row weights must be positive integers")
        shards = []
        start = 0
        for part in _partition_contiguous_sizes(weights, num_parts=num_shards):
            end = start + len(part)
            shards.append(rows[start:end])
            start = end
        return shards

    base_size, remainder = divmod(len(rows), num_shards)
    shards: list[list[JsonRow]] = []
    start = 0
    for shard_index in range(num_shards):
        shard_size = base_size + (1 if shard_index < remainder else 0)
        end = start + shard_size
        shards.append(rows[start:end])
        start = end
    return shards


def build_features(
    embed: bool,
    *,
    embedded_image_struct: bool = False,
    embedded_large_binary: bool = False,
    include_contract_stats: bool = False,
    include_image_num_tokens: bool = False,
    include_tools: bool = False,
    include_videos: bool = False,
    include_channel: bool = False,
    include_message_loss: bool = False,
    include_message_loss_scale: bool = False,
) -> Features:
    message_fields = {"role": Value("string"), "content": Value("string")}
    if include_message_loss:
        message_fields["loss"] = Value("bool")
    if include_message_loss_scale:
        # float64 is intentional: per-question weights such as 1/3 must survive
        # JSONL -> Arrow -> Parquet without being rounded to float32.
        message_fields["loss_scale"] = Value("float64")
    fields: dict[str, Any] = {
        "messages": [message_fields],
    }
    if embed:
        media_binary_dtype = "large_binary" if embedded_large_binary else "binary"
        fields["images"] = (
            [{"bytes": Value(media_binary_dtype), "path": Value("string")}]
            if embedded_image_struct
            else HFSequence(Image())
        )
        fields["audios"] = HFSequence(Value(media_binary_dtype))
    else:
        fields["images"] = HFSequence(Value("string"))
        fields["audios"] = HFSequence(Value("string"))
    if include_videos:
        fields["videos"] = HFSequence(Value("string"))
    if include_tools:
        fields["tools"] = Value("string")
    if include_channel:
        fields["channel"] = Value("string")
    if include_contract_stats:
        fields["omni_source_id"] = Value("string")
        fields["omni_num_images"] = Value("int32")
        fields["omni_audio_num_samples"] = HFSequence(Value("int64"))
        fields["omni_last_frame_timestamp_ms"] = Value("int64")
        if include_image_num_tokens:
            fields["omni_image_num_tokens"] = Value("int64")
    return Features(fields)


def read_messages_jsonl(jsonl_paths: list[str]) -> list[JsonRow]:
    rows: list[JsonRow] = []
    for jsonl_path in jsonl_paths:
        with open(jsonl_path, encoding="utf-8") as src:
            for line_no, line in enumerate(src, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{jsonl_path}:{line_no} is not valid JSON: {exc}") from exc
                if not isinstance(row.get("messages"), list):
                    raise ValueError(f"{jsonl_path}:{line_no} missing list field: messages")
                rows.append(row)
    return rows
