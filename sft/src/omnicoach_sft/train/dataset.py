"""Agent shared-media contract checks, schema alignment, and lazy row resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml
from datasets import Dataset, Features, Value, load_dataset as hf_load_dataset

from omnicoach_sft.data.parquet import (
    AGENT_SHARED_MEDIA_LAYOUT,
    AGENT_SHARED_MEDIA_SCHEMA_VERSION,
    MEDIA_TABLE_SCHEMA,
    TURN_TABLE_DIR,
)
from omnicoach_sft.media.parquet_pool import (
    MEDIA_TABLE_DIR,
    IndexedMediaStore,
    LazyMediaDataset,
    MediaRegistry,
    ParquetFilePool,
    _parquet_parts,
)

_MESSAGE_FEATURES = Features(
    {
        "messages": [
            {
                "role": Value("string"),
                "content": Value("string"),
                "loss": Value("bool"),
                "loss_scale": Value("float64"),
            }
        ]
    }
)

_METADATA_FEATURES = {
    "omni_source_id": Value("string"),
    "omni_num_images": Value("int64"),
    "omni_image_num_tokens": Value("int64"),
    "omni_audio_num_samples": [Value("int64")],
    "omni_last_frame_timestamp_ms": Value("int64"),
}

AGENT_STATS_COLUMNS = set(_METADATA_FEATURES) - {"omni_image_num_tokens"}
AGENT_REQUIRED_COLUMNS = AGENT_STATS_COLUMNS | {"messages", "channel"}

__all__ = [
    "AGENT_REQUIRED_COLUMNS",
    "AGENT_STATS_COLUMNS",
    "AgentMediaLocation",
    "AgentSharedMediaDataset",
    "AgentSharedMediaRegistry",
    "AgentSharedMediaStore",
    "IndexedMediaStore",
    "LazyMediaDataset",
    "MediaRegistry",
    "ParquetFilePool",
    "_parquet_parts",
    "expand_agent_manifests",
    "expand_agent_shared_turn_parts",
    "load_agent_contract",
    "load_agent_parquet_parts",
    "normalize_mixed_training_columns",
    "pq",
    "prepare_agent_length_dataset",
    "resolve_agent_shared_source_root",
    "validate_agent_training_artifact",
]


def load_agent_contract(source: str | Path) -> dict[str, Any] | None:
    path = Path(source).expanduser() / "dataset_contract.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        isinstance(payload, dict)
        and payload.get("schema_version") == AGENT_SHARED_MEDIA_SCHEMA_VERSION
        and payload.get("table_layout") == AGENT_SHARED_MEDIA_LAYOUT
    ):
        return payload
    return None


def load_agent_parquet_parts(
    parts: list[str],
    *,
    num_proc: int,
    cache_dir: str | Path | None = None,
) -> Dataset:
    """Load contracted Agent shards as one HF Parquet train split."""

    if not parts:
        raise ValueError("Agent Parquet parts must not be empty")
    worker_count = min(max(1, int(num_proc)), len(parts))
    load_kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        load_kwargs["cache_dir"] = str(cache_dir)
    return hf_load_dataset(
        "parquet",
        data_files={"train": list(parts)},
        split="train",
        num_proc=worker_count,
        **load_kwargs,
    )


def validate_agent_training_artifact(dataset: Any, source: str | Path) -> None:
    """Trust the exporter validation receipt after checking its artifact metadata."""

    source_path = Path(source).expanduser()
    contract = load_agent_contract(source_path)
    if contract is None:
        raise ValueError(f"unsupported Agent dataset contract: {source_path}")
    validation_path = source_path / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not isinstance(validation, dict) or validation.get("ok") is not True:
        raise ValueError(f"Agent dataset validation is missing or failed: {validation_path}")
    validated_rows = validation.get("rows", validation.get("turn_rows", -1))
    if int(validated_rows) != len(dataset):
        raise ValueError(
            f"Agent dataset row count does not match validation: "
            f"dataset={len(dataset)} validation={validated_rows}"
        )

    required_columns = set(AGENT_REQUIRED_COLUMNS) | {"images", "audios", "tools"}
    declared_columns = set(contract.get("columns") or [])
    missing = sorted(required_columns - declared_columns)
    if missing:
        raise ValueError(f"Agent dataset contract is missing columns: {missing}")
    missing = sorted(required_columns - set(dataset.column_names))
    if missing:
        raise ValueError(f"Agent Parquet is missing training columns: {missing}")

    if contract.get("embedded_media") or validation.get("embedded_media"):
        raise ValueError(f"Agent shared-media turns must not declare embedded media: {source_path}")
    if contract.get("media_columns") != "lazy_embedded_media_parquet":
        raise ValueError(f"Agent shared-media contract has an invalid media column mode: {source_path}")


def normalize_mixed_training_columns(dataset: Any, *, channel: str) -> Any:
    """Align lightweight columns without rewriting embedded media buffers."""

    if "messages" not in dataset.column_names:
        raise ValueError("training source is missing messages")
    if channel not in {"base", "agent"}:
        raise ValueError(f"unsupported training channel: {channel}")
    if getattr(dataset, "_indices", None) is not None:
        raise ValueError("training schema normalization must run before row filtering")

    def replace_column(name: str, column: Any, feature: Any) -> None:
        nonlocal dataset
        # remove_columns and horizontal add_column reuse every untouched Arrow
        # buffer. This avoids Dataset.cast/map re-serializing image/audio bytes.
        dataset = dataset.remove_columns(name)
        dataset = dataset.add_column(name, column, feature=feature)

    def cast_column(name: str, feature: Any) -> None:
        target_type = Features({name: feature}).arrow_schema.field(name).type
        replace_column(name, dataset.data.table.column(name).cast(target_type), feature)

    target_message_feature = _MESSAGE_FEATURES["messages"]
    if dataset.features["messages"] != target_message_feature:
        cast_column("messages", target_message_feature)

    for name, feature in _METADATA_FEATURES.items():
        if name not in dataset.column_names or dataset.features[name] == feature:
            continue
        cast_column(name, feature)

    if "channel" not in dataset.column_names:
        dataset = dataset.add_column("channel", [channel] * len(dataset), feature=Value("string"))
    elif dataset.features["channel"] != Value("string"):
        cast_column("channel", Value("string"))

    if set(dataset["channel"]) != {channel}:
        raise ValueError(f"training source contains rows outside channel={channel}")
    return dataset


def prepare_agent_length_dataset(dataset: Any, *, source: str, channel: str) -> Any:
    """Apply the fingerprint-affecting Agent transforms used before length caching."""

    if "dataset" not in dataset.column_names:
        dataset = dataset.add_column("dataset", [str(source)] * len(dataset))
    return normalize_mixed_training_columns(dataset, channel=channel)


@dataclass(frozen=True)
class AgentMediaLocation:
    path: Path
    row_group: int


def resolve_agent_shared_source_root(source: str | Path) -> Path | None:
    path = Path(source).expanduser()
    if not path.is_dir() or not (path / TURN_TABLE_DIR).is_dir() or not (path / MEDIA_TABLE_DIR).is_dir():
        return None
    return path.resolve() if load_agent_contract(path) is not None else None


def expand_agent_manifests(sources: list[str] | tuple[str, ...]) -> list[str]:
    """Expand a YAML ``dataset`` list when every entry is an Agent shared-media root."""

    expanded: list[str] = []
    for source in sources:
        path = Path(source).expanduser()
        if path.suffix.lower() not in {".yaml", ".yml"} or not path.is_file():
            expanded.append(source)
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        entries = payload.get("dataset") if isinstance(payload, dict) else None
        if not isinstance(entries, list) or not entries:
            expanded.append(source)
            continue
        resolved_entries = [str(item) for item in entries]
        if any(resolve_agent_shared_source_root(item) is None for item in resolved_entries):
            expanded.append(source)
            continue
        expanded.extend(resolved_entries)
    return expanded


def expand_agent_shared_turn_parts(source: str | Path) -> list[str]:
    # HF Datasets given the directory would also pick up validation.json and
    # return a validation-only split; Swift needs explicit train shards.
    root = resolve_agent_shared_source_root(source)
    if root is None:
        raise ValueError(f"not an agent shared-media source: {source}")
    return [str(path.resolve()) for path in _parquet_parts(root / TURN_TABLE_DIR)]


class AgentSharedMediaStore(IndexedMediaStore):
    """Index only media IDs; read one binary media bundle when a row is fetched."""

    media_label = "Agent "
    index_count_error = "Agent media ID count does not match row groups"
    media_schema = MEDIA_TABLE_SCHEMA

    def __init__(
        self,
        source_root: str | Path,
        *,
        max_open_files: int = 32,
        file_pool: ParquetFilePool | None = None,
    ) -> None:
        root = resolve_agent_shared_source_root(source_root)
        if root is None:
            raise ValueError(f"not an agent shared-media source: {source_root}")
        super().__init__(root, max_open_files=max_open_files, file_pool=file_pool)

    def _location(self, path: Path, row_group: int, media_id: str, row: dict[str, Any]) -> AgentMediaLocation:
        return AgentMediaLocation(path, row_group)


class AgentSharedMediaRegistry(MediaRegistry):
    store_class = AgentSharedMediaStore
    # Content-addressed media can be shared across publication shards.
    reuse_media_ids = True

    def resolve_row(self, row: dict[str, Any]) -> dict[str, Any]:
        media_id = row.get("media_id")
        if media_id is None or media_id == "":
            return row
        media_id = str(media_id)
        store = self.store_for(media_id)
        media = store.read(media_id)
        resolved = dict(row)
        resolved.pop("media_id", None)
        resolved["images"] = list(media["images"] or [])
        resolved["audios"] = list(media["audios"] or [])
        indices = resolved.pop("omni_image_indices", None)
        if indices is not None:
            if any(type(index) is not int for index in indices):
                raise ValueError("omni_image_indices must contain integer frame indices")
            if indices != sorted(set(indices)):
                raise ValueError("omni_image_indices must be unique and strictly increasing")
            if any(index < 0 or index >= len(resolved["images"]) for index in indices):
                raise ValueError("omni_image_indices contains an out-of-range frame index")
            resolved["images"] = [resolved["images"][index] for index in indices]
            if len(resolved["images"]) != int(resolved["omni_num_images"]):
                raise ValueError("selected image count does not match omni_num_images")
        return resolved


class AgentSharedMediaDataset(LazyMediaDataset):
    """Dataset facade that resolves Agent media only for requested examples."""

    resolved_columns = ("images", "audios")
