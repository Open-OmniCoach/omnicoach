"""Inspect or validate an Agent shared-media Parquet dataset without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from omnicoach_sft.data.parquet import TURN_TABLE_DIR
from omnicoach_sft.media.parquet_pool import MEDIA_TABLE_DIR
from omnicoach_sft.train.dataset import (
    AgentSharedMediaRegistry,
    expand_agent_shared_turn_parts,
    load_agent_contract,
    load_agent_parquet_parts,
    resolve_agent_shared_source_root,
    validate_agent_training_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Agent Parquet root containing dataset_contract.json, turns/, and media/.")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate contract/turns/media IDs without loading full binary media into memory.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Resolve one turn row and report its message and media payload counts.",
    )
    return parser.parse_args()


def parquet_summary(directory: Path) -> dict[str, object]:
    paths = sorted(directory.glob("part-*.parquet"))
    if not paths:
        raise ValueError(f"no parquet shards under {directory}")
    rows = 0
    row_groups = 0
    for path in paths:
        metadata = pq.ParquetFile(path).metadata
        rows += metadata.num_rows
        row_groups += metadata.num_row_groups
    return {
        "shards": len(paths),
        "rows": rows,
        "row_groups": row_groups,
        "columns": pq.read_schema(paths[0]).names,
    }


def _validate_and_index_source(root: Path, *, sample_index: int | None) -> dict[str, Any]:
    parts = expand_agent_shared_turn_parts(root)
    dataset = load_agent_parquet_parts(parts, num_proc=1)
    validate_agent_training_artifact(dataset, root)
    registry = AgentSharedMediaRegistry()
    try:
        store = registry.store_for_root(root)
        referenced = {str(value) for value in dataset["media_id"] if value}
        missing = sorted(referenced - store.media_ids)
        if missing:
            raise ValueError(f"Agent turns reference missing media_id(s): {missing[:3]}")
        details: dict[str, Any] = {
            "validated_turn_rows": len(dataset),
            "indexed_media_rows": len(store.media_ids),
        }
        if sample_index is not None:
            if not -len(dataset) <= sample_index < len(dataset):
                raise IndexError(f"sample index {sample_index} is outside a {len(dataset)}-row turn table")
            row = registry.resolve_row(dataset[sample_index])
            details["sample"] = {
                "channel": row.get("channel"),
                "messages": len(row["messages"]),
                "frames": len(row["images"]),
                "audio_tracks": len(row["audios"]),
            }
        return details
    finally:
        registry.close()


def inspect_source(source: str | Path, *, validate: bool, sample_index: int | None) -> dict[str, object]:
    root = resolve_agent_shared_source_root(source)
    if root is None:
        raise ValueError(f"not an Agent shared-media source root: {source}")
    contract = load_agent_contract(root)
    if contract is None:
        raise ValueError(f"unsupported Agent dataset contract: {root}")
    result: dict[str, object] = {
        "source_root": str(root),
        "source_id": contract.get("source_id"),
        "protocol": contract.get("protocol"),
        "table_layout": contract.get("table_layout"),
        "turns": parquet_summary(root / TURN_TABLE_DIR),
        "media": parquet_summary(root / MEDIA_TABLE_DIR),
    }
    if validate or sample_index is not None:
        result.update(_validate_and_index_source(root, sample_index=sample_index))
    return result


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            inspect_source(args.source, validate=args.validate, sample_index=args.sample_index),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
