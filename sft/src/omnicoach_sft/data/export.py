"""Build a locally validated Agent JSON export as shared-media Parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omnicoach_sft.data.evalkit import materialize_evalkit_rows
from omnicoach_sft.data.parquet import AgentParquetConfig, write_agent_parquet
from omnicoach_sft.data.schema import (
    ALLOWED_ROLES,
    CONTROL_PREFIXES,
    FRAME_TIMESTAMP_RE,
    MEDIA_KEYS,
    PACK_AUDIO_SAMPLE_RATE,
    PACK_IMAGE_MAX_PIXELS,
    PACK_IMAGE_MIN_PIXELS,
    PACK_IMAGE_QUALITY,
    PACK_IMAGE_RESIZE_FACTOR,
    PACK_IMAGE_WEBP_METHOD,
    agent_media_encoding,
    agent_pack_length_stats,
    image_token_stats,
    validate_agent_row,
)
from omnicoach_sft.io import (
    build_features,
    read_mapping_json as load_json,
    read_messages_jsonl,
)
from omnicoach_sft.media.cache import (
    add_pack_length_stats,
    audio_num_samples_16khz,
    cache_base_wav,
    cache_base_webp,
    embedded_image_num_tokens,
    prepare_embedded_media_paths,
    probe_audio_sample_counts,
    probe_image_token_counts,
)
from omnicoach_sft.media.image_tokens import (
    qwen_image_token_contribution as _qwen_image_token_contribution,
    webp_dimensions as _webp_dimensions,
)

__all__ = [
    "ALLOWED_ROLES",
    "CONTROL_PREFIXES",
    "FRAME_TIMESTAMP_RE",
    "MEDIA_KEYS",
    "PACK_AUDIO_SAMPLE_RATE",
    "PACK_IMAGE_MAX_PIXELS",
    "PACK_IMAGE_MIN_PIXELS",
    "PACK_IMAGE_QUALITY",
    "PACK_IMAGE_RESIZE_FACTOR",
    "PACK_IMAGE_WEBP_METHOD",
    "_qwen_image_token_contribution",
    "_webp_dimensions",
    "add_pack_length_stats",
    "agent_media_encoding",
    "agent_pack_length_stats",
    "audio_num_samples_16khz",
    "cache_base_wav",
    "cache_base_webp",
    "embedded_image_num_tokens",
    "group_rows_by_base_provenance",
    "image_token_stats",
    "main",
    "parse_args",
    "prepare_agent_rows",
    "prepare_embedded_media_paths",
    "probe_audio_sample_counts",
    "probe_image_token_counts",
    "validate_agent_row",
    "validate_source",
]


def validate_source(source_export: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    train_jsonl = source_export / "train.jsonl"
    source_contract = load_json(source_export / "dataset_contract.json")
    source_validation = load_json(source_export / "validation.json")
    if not source_validation.get("ok"):
        raise ValueError(f"source validation failed: {source_export}")
    protocol = source_contract.get("protocol")
    supported_protocols = {
        "native-tool-v45-seconds-v2",
        "native-tool-v45-seconds-v1",
        "native-tool-v2",
        "native-tool-v3-seconds",
        "native-tool-v3-seconds-think-only",
    }
    if protocol not in supported_protocols:
        raise ValueError(
            f"unsupported source protocol {protocol!r}; expected one of {sorted(supported_protocols)}: {source_export}"
        )
    if not train_jsonl.is_file():
        raise FileNotFoundError(train_jsonl)
    return train_jsonl, source_contract, source_validation


def group_rows_by_base_provenance(
    source_export: Path,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    provenance_path = source_export / "provenance.jsonl"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"--group-by-base requires {provenance_path}")

    grouped_indices: dict[str, list[int]] = {}
    with provenance_path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            try:
                provenance = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{provenance_path}:{row_index + 1} is not valid JSON: {exc}") from exc
            if not isinstance(provenance, dict):
                raise ValueError(f"{provenance_path}:{row_index + 1} must be an object")
            if provenance.get("row_idx") != row_index:
                raise ValueError(
                    f"{provenance_path}:{row_index + 1} row_idx must be {row_index}, "
                    f"got {provenance.get('row_idx')!r}"
                )
            base_sample_id = provenance.get("base_sample_id")
            if not isinstance(base_sample_id, str) or not base_sample_id:
                raise ValueError(f"{provenance_path}:{row_index + 1} missing base_sample_id")
            grouped_indices.setdefault(base_sample_id, []).append(row_index)
    if sum(len(indices) for indices in grouped_indices.values()) != len(rows):
        raise ValueError(
            f"train/provenance row mismatch: train={len(rows)} "
            f"provenance={sum(len(indices) for indices in grouped_indices.values())}"
        )

    order = [index for indices in grouped_indices.values() for index in indices]
    grouped_rows = [rows[index] for index in order]
    group_ids = [base_sample_id for base_sample_id, indices in grouped_indices.items() for _ in indices]
    rows_per_base = [len(indices) for indices in grouped_indices.values()]
    stats = {
        "key": "base_sample_id",
        "base_groups": len(grouped_indices),
        "min_rows_per_base": min(rows_per_base),
        "max_rows_per_base": max(rows_per_base),
        "source_order_changed": order != list(range(len(rows))),
        "provenance_path": str(provenance_path),
    }
    return grouped_rows, group_ids, stats


def prepare_agent_rows(
    rows: list[dict[str, Any]],
    *,
    media_resolver: dict[str, Any] | None,
    cache_dir: Path,
    media_workers: int,
    embed: bool,
    source_id: str,
    invalid_image_report: Path | None = None,
) -> list[dict[str, Any]]:
    """Materialize, normalize, and probe one independently disposable row batch."""

    if media_resolver is not None:
        rows = materialize_evalkit_rows(
            rows,
            resolver=media_resolver,
            output_root=cache_dir / "evalkit_media",
            workers=media_workers,
        )
    if embed:
        rows = prepare_embedded_media_paths(
            rows,
            cache_root=cache_dir / "embedded_media",
            num_proc=media_workers,
            direct_compatible_webp=True,
        )
    audio_sample_counts = probe_audio_sample_counts(rows, num_proc=media_workers)
    image_token_counts = probe_image_token_counts(
        rows,
        num_proc=media_workers,
        invalid_report_path=invalid_image_report,
    )
    return add_pack_length_stats(
        rows,
        source_id=source_id,
        audio_sample_counts=audio_sample_counts,
        image_token_counts=image_token_counts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("source-export", "out"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--source-id")
    for option, default in (
        ("num-shards", 1),
        ("row-group-size", 100),
        ("num-proc", 1),
        ("media-workers", 0),
        ("shard-worker-index", 0),
        ("shard-worker-count", 1),
        ("limit-rows", 0),
    ):
        parser.add_argument("--" + option, type=int, default=default)
    for option in ("cache-dir", "resume-partial", "invalid-image-report"):
        parser.add_argument("--" + option, type=Path)
    for option in ("build-only", "group-by-base", "overwrite"):
        parser.add_argument("--" + option, action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train, source, validation = validate_source(args.source_export.resolve())
    rows = read_messages_jsonl([str(train)])
    if len(rows) != int(validation["train_rows"]):
        raise ValueError("train.jsonl row count differs from source validation")
    if args.limit_rows < 0 or args.media_workers < 0:
        raise ValueError("negative row limit or media worker count")
    groups, grouping = None, None
    if args.group_by_base:
        rows, groups, grouping = group_rows_by_base_provenance(args.source_export, rows)
    if args.limit_rows:
        rows = rows[: args.limit_rows]
        groups = groups[: args.limit_rows] if groups is not None else None
    source_id = args.source_id or source.get("source_id") or f"agent:{args.source_export.name}"
    schema = build_features(
        True,
        embedded_image_struct=True,
        embedded_large_binary=True,
        include_contract_stats=True,
        include_image_num_tokens=True,
        include_tools=True,
        include_channel=True,
        include_message_loss=True,
        include_message_loss_scale=True,
    ).arrow_schema

    def prepare(batch: list[dict[str, Any]], cache: Path) -> list[dict[str, Any]]:
        return prepare_agent_rows(
            batch,
            media_resolver=source.get("media_resolver"),
            cache_dir=cache,
            media_workers=max(1, (args.media_workers or args.num_proc) // min(args.num_proc, args.num_shards)),
            embed=True,
            source_id=source_id,
            invalid_image_report=args.invalid_image_report,
        )

    contract = {
        **source,
        "format": "ms-swift-multimodal-parquet",
        "model_family": "qwen3_omni",
        "purpose": "agentic_turn_sft",
        "channel": "agent",
        "source_export": str(args.source_export.resolve()),
        "row_granularity": "one supervised assistant turn",
        "pack_length_stats": agent_pack_length_stats(),
        "media_encoding": {**agent_media_encoding(), "layout": "shared_media_table"},
        "training_template": source.get("training_template")
        or {
            "template_backend": "swift",
            "agent_template": "hermes",
            "loss_scale": source.get("supervision", {}).get("required_loss_scale", "last_round"),
        },
    }
    if grouping:
        contract["physical_layout"] = {
            **grouping,
            "atomic_base_groups": True,
            "row_group_target_rows": args.row_group_size,
        }
    config = AgentParquetConfig(
        output=args.out,
        source_id=source_id,
        schema=schema,
        num_shards=args.num_shards,
        row_group_size=args.row_group_size,
        group_ids=groups,
        workers=args.num_proc,
        cache_dir=args.cache_dir,
        partial=args.resume_partial,
        resume=args.resume_partial is not None,
        worker_index=args.shard_worker_index,
        worker_count=args.shard_worker_count,
        build_only=args.build_only,
        overwrite=args.overwrite,
        prepare_rows=prepare,
        validate_row=lambda row, context: validate_agent_row(row, context=context, embedded_media=True),
        contract=contract,
        identity={"protocol": source["protocol"], "media_encoding": agent_media_encoding()},
        metadata={
            "omni.format": "agent_turn_shared_media_v1",
            "omni.protocol": source["protocol"],
            "omni.source_id": source_id,
        },
    )
    result = write_agent_parquet(rows, config)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "tables"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
