"""Packing-length profiles, atomic caches and model-specific CPU estimation."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import logging
import math
import os
import re
import sys
import time
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from datasets import Dataset
from tqdm import tqdm

try:
    import torch.distributed as dist
except ImportError:  # pragma: no cover - optional in CPU-only unit environments
    dist = None

from omnicoach_sft.data.parquet import TURN_TABLE_DIR
from omnicoach_sft.io import require_mapping
from omnicoach_sft.media.image_tokens import QWEN_IMAGE_TOKEN_COUNT_BASIS
from omnicoach_sft.paths import project_path
from omnicoach_sft.train.dataset import AGENT_STATS_COLUMNS, load_agent_contract


@dataclass(frozen=True)
class PackLengthProfile:
    path: Path
    schema_version: str
    model_family: str = "qwen3_omni"
    tokens_per_image: int = 192
    max_image_pixels: int = 384 * 384
    audio_tokens_per_second: int = 13
    audio_tail_tokens: int = 2
    audio_sample_rate: int = 16000
    audio_hop_length: int = 160
    safety_tokens: int = 128
    round_up_to: int = 256
    exact_encode_threshold: int = 60000
    exact_encode_upper_bound: int | None = None
    overflow: str = "filter"
    map_batch_size: int = 256
    cache_dir: Path | None = None
    lock_stale_seconds: int = 7200
    profile_hash: str = ""


def load_pack_length_profile(raw_path: str | Path) -> PackLengthProfile:
    path = project_path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"pack-length profile not found: {path}")

    payload = require_mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "pack-length profile")
    schema_version = str(payload.get("schema_version") or "")
    if schema_version != "omni_pack_length/v1":
        raise ValueError(f"unsupported pack-length schema_version: {schema_version!r}")

    numeric_fields = [field for field in fields(PackLengthProfile) if isinstance(field.default, int)]
    allowed_keys = {
        "schema_version",
        "mode",
        "model_family",
        "image",
        "audio",
        "exact_encode_upper_bound",
        "overflow",
        "cache_dir",
    }
    allowed_keys.update(
        field.name
        for field in numeric_fields
        if not field.name.startswith("audio_") and field.name not in {"tokens_per_image", "max_image_pixels"}
    )
    unknown_keys = sorted(set(payload) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"unknown pack-length profile keys: {unknown_keys}")

    mode = payload.get("mode")
    if mode != "contract_stats":
        raise ValueError(f"unsupported pack-length mode: {mode!r}")
    model_family = str(payload.get("model_family") or "qwen3_omni")
    if model_family not in {"qwen3_omni", "qwen2_5_omni"}:
        raise ValueError(f"unsupported pack-length model_family: {model_family!r}")

    image = require_mapping(payload.get("image") or {}, "image")
    audio = require_mapping(payload.get("audio") or {}, "audio")
    values: dict[str, Any] = {}
    for field in numeric_fields:
        name, default = field.name, field.default
        if name.startswith("audio_"):
            value = audio.get(name.removeprefix("audio_"), default)
        elif name in {"tokens_per_image", "max_image_pixels"}:
            value = image.get("max_pixels" if name == "max_image_pixels" else name, default)
        else:
            value = payload.get(name, default)
        values[name] = int(value)
        minimum = 0 if name in {"safety_tokens", "audio_tail_tokens"} else 1
        if values[name] < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {values[name]}")
    values.update(
        exact_encode_upper_bound=(
            int(payload["exact_encode_upper_bound"]) if payload.get("exact_encode_upper_bound") is not None else None
        ),
        overflow=str(payload.get("overflow", "filter")),
        cache_dir=project_path(payload["cache_dir"]) if payload.get("cache_dir") else None,
    )
    if (
        values["exact_encode_upper_bound"] is not None
        and values["exact_encode_upper_bound"] < values["exact_encode_threshold"]
    ):
        raise ValueError("exact_encode_upper_bound must be >= exact_encode_threshold")
    if values["overflow"] != "filter":
        raise ValueError(f"only overflow=filter is supported, got {values['overflow']!r}")
    if model_family == "qwen2_5_omni" and any(
        values[name] != expected
        for name, expected in (
            ("audio_tokens_per_second", 25),
            ("audio_tail_tokens", 2),
            ("audio_sample_rate", 16000),
            ("audio_hop_length", 160),
        )
    ):
        raise ValueError("Qwen2.5-Omni requires 16 kHz / hop 160 audio, 25 tokens/s and two wrapper tokens")

    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    profile_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return PackLengthProfile(
        path=path,
        schema_version=schema_version,
        model_family=model_family,
        **values,
        profile_hash=profile_hash,
    )


@dataclass(frozen=True)
class CachedLengths:
    lengths: np.ndarray
    keep_indices: np.ndarray
    metadata: dict[str, Any]


def safe_cache_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value[:100] or "dataset"


def cache_paths(cache_dir: Path, source_name: str, cache_key: str) -> tuple[Path, Path, Path]:
    stem = f"{safe_cache_name(source_name)}.{cache_key[:20]}"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.json", cache_dir / f"{stem}.lock"


def load_cached_lengths(npz_path: Path, json_path: Path, expected: dict[str, Any]) -> CachedLengths | None:
    try:
        with np.load(npz_path, allow_pickle=False) as payload:
            # New entries are one atomic artifact. Old NPZ + JSON entries stay
            # readable with the same cache key; no warm-cache invalidation.
            if "metadata" in payload:
                metadata = json.loads(payload["metadata"].tobytes().decode("utf-8"))
            else:
                metadata = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                return None
            for key, value in expected.items():
                if metadata.get(key) != value:
                    return None
            lengths = np.asarray(payload["lengths"], dtype=np.int64)
            keep_indices = np.asarray(payload["keep_indices"], dtype=np.int64)
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None
    if len(lengths) != len(keep_indices):
        return None
    if int(metadata.get("kept_rows", -1)) != len(lengths):
        return None
    return CachedLengths(lengths=lengths, keep_indices=keep_indices, metadata=metadata)


def save_cached_lengths(
    npz_path: Path,
    json_path: Path,
    *,
    lengths: np.ndarray,
    keep_indices: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Publish arrays and metadata together using one same-directory rename.

    Concurrent jobs may compute the same cold key. Each publishes a complete,
    independently valid artifact; neither deletes another publisher's files.
    json_path is retained for callers/legacy reads, but new writes need no sidecar.
    """
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = npz_path.with_name(f".{npz_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        # FUSE may not support seekable writes: finalize the ZIP in memory.
        buffer = io.BytesIO()
        metadata_bytes = json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8")
        np.savez_compressed(
            buffer,
            lengths=lengths,
            keep_indices=keep_indices,
            metadata=np.frombuffer(metadata_bytes, dtype=np.uint8),
        )
        with tmp_npz.open("wb") as handle:
            handle.write(buffer.getbuffer())
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                # Some distributed/FUSE mounts do not implement fsync.
                pass
        os.replace(tmp_npz, npz_path)
    finally:
        tmp_npz.unlink(missing_ok=True)


logger = logging.getLogger(__name__)
ESTIMATOR_VERSION = "qwen3_omni_fast_pack_v9"
QWEN25_IMAGE_TOKEN_COUNT_BASIS = "qwen2_5_omni_patch14_merge2_plus_vision_wrappers"


@dataclass(frozen=True)
class ContractLengthStats:
    contract_hash: str
    source_id: str
    image_max_pixels: int
    audio_sample_rate: int
    is_agent: bool
    has_exact_media_lengths: bool


def _info(message: str) -> None:
    if "swift.utils" in sys.modules:
        from swift.utils import get_logger

        get_logger().info(message)
    else:
        logger.info(message)


def _as_int_length(value: Any) -> int:
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(sum(int(item) for item in value))
    return int(value)


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + multiple - 1) // multiple) * multiple


def _message_contents(messages: list[dict[str, Any]]) -> list[str]:
    return [str(message.get("content") or "") for message in messages]


def _strip_media_placeholders(
    messages: list[dict[str, Any]],
    *,
    model_family: str = "qwen3_omni",
) -> list[dict[str, Any]]:
    stripped = copy.deepcopy(messages)
    # Keep tokenizer boundaries for Qwen2.5. Removing the tag completely can
    # merge neighboring timestamp/text tokens once per image; fixed safety
    # margins cannot cover that for arbitrary later tool observations.
    image_boundary = "<|vision_bos|><|vision_eos|>" if model_family == "qwen2_5_omni" else ""
    audio_boundary = "<|audio_bos|><|audio_eos|>" if model_family == "qwen2_5_omni" else ""
    for message in stripped:
        content = str(message.get("content") or "")
        message["content"] = (
            content.replace("<image>", image_boundary)
            .replace("<audio>", audio_boundary)
            .replace("<video>", "")
        )
    return stripped


def _placeholder_count(messages: list[dict[str, Any]], placeholder: str) -> int:
    return sum(content.count(placeholder) for content in _message_contents(messages))


def _contract_source_root(source: str | Path) -> Path:
    """Resolve a configured Parquet source back to its artifact directory."""

    raw = str(source)
    sampled = re.fullmatch(r"(.+)#([1-9]\d*)", raw)
    if sampled is not None:
        candidate = Path(sampled.group(1)).expanduser()
        if candidate.is_file():
            raw = sampled.group(1)
    source_path = Path(raw).expanduser()
    if source_path.is_file():
        parent = source_path.parent
        if parent.name == TURN_TABLE_DIR and (parent.parent / "dataset_contract.json").is_file():
            return parent.parent
        return parent
    return source_path


def _load_contract_stats(source: str | Path, *, model_family: str = "qwen3_omni") -> ContractLengthStats:
    """Load the metadata needed to trust precomputed multimodal length stats."""

    source_path = _contract_source_root(source)
    contract_path = source_path / "dataset_contract.json"
    payload = load_agent_contract(source_path)
    if payload is None:
        raise ValueError(f"unsupported or missing Agent dataset contract: {contract_path}")

    for field in ("purpose", "protocol", "source_id"):
        if not str(payload.get(field) or "").strip():
            raise ValueError(f"Agent dataset contract is missing {field}: {contract_path}")
    source_id = str(payload["source_id"]).strip()

    stats = payload.get("pack_length_stats")
    if not isinstance(stats, dict):
        raise ValueError(f"Agent dataset contract is missing pack_length_stats: {contract_path}")
    required = {"image_max_pixels", "audio_sample_rate", "audio_sample_count_basis"}
    missing = sorted(required - set(stats))
    if missing:
        raise ValueError(f"Agent pack_length_stats is missing fields {missing}: {contract_path}")
    if stats["audio_sample_count_basis"] != "resampled_16khz":
        raise ValueError(
            f"Agent omni_audio_num_samples must use audio_sample_count_basis='resampled_16khz': {contract_path}"
        )
    image_basis = stats.get("image_token_count_basis")
    image_column = stats.get("image_token_count_column")
    if (image_basis is None) != (image_column is None):
        raise ValueError(
            "Agent exact image-token stats must declare both image_token_count_basis and "
            f"image_token_count_column: {contract_path}"
        )
    has_exact_media_lengths = image_basis is not None
    expected_image_basis = (
        QWEN25_IMAGE_TOKEN_COUNT_BASIS if model_family == "qwen2_5_omni" else QWEN_IMAGE_TOKEN_COUNT_BASIS
    )
    if model_family == "qwen2_5_omni" and not has_exact_media_lengths:
        raise ValueError(f"Qwen2.5-Omni requires exact image-token contract statistics: {contract_path}")
    if has_exact_media_lengths and (
        image_basis != expected_image_basis or image_column != "omni_image_num_tokens"
    ):
        raise ValueError(
            f"unsupported Agent exact image-token stats basis={image_basis!r} column={image_column!r}: {contract_path}"
        )
    contract_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    return ContractLengthStats(
        contract_hash=contract_hash,
        source_id=source_id,
        image_max_pixels=int(stats["image_max_pixels"]),
        audio_sample_rate=int(stats["audio_sample_rate"]),
        is_agent=True,
        has_exact_media_lengths=has_exact_media_lengths,
    )


class _MessageLengthMapper:
    def __init__(self, template: Any, profile: PackLengthProfile):
        self.template = template
        self.profile = profile

    def __call__(self, batch: dict[str, list[Any]]) -> dict[str, list[int]]:
        output: list[int] = []
        for index, messages in enumerate(batch["messages"]):
            tools = batch["tools"][index] if "tools" in batch else None
            image_count = int(batch["_omni_image_count"][index])
            image_num_tokens = batch["_omni_image_num_tokens"][index] if "_omni_image_num_tokens" in batch else None
            audio_count = int(batch["_omni_audio_count"][index])
            audio_samples = batch["_omni_audio_num_samples"][index]
            placeholder_images = _placeholder_count(messages, "<image>")
            placeholder_audios = _placeholder_count(messages, "<audio>")
            if _placeholder_count(messages, "<video>"):
                raise ValueError("Agent messages must contain expanded <image>/<audio> placeholders, not <video>")
            for kind, placeholders, count in (
                ("image", placeholder_images, image_count),
                ("audio", placeholder_audios, audio_count),
            ):
                if placeholders != count:
                    raise ValueError(
                        f"{kind} placeholder/media count mismatch: placeholders={placeholders}, media={count}"
                    )

            text_row: dict[str, Any] = {
                "messages": _strip_media_placeholders(messages, model_family=self.profile.model_family)
            }
            if tools is not None:
                text_row["tools"] = tools
            text_length = _as_int_length(self.template.encode(text_row, return_length=True)["lengths"])

            image_length = (
                image_count * self.profile.tokens_per_image if image_num_tokens is None else int(image_num_tokens)
            )
            if image_num_tokens is not None:
                if image_length < 0:
                    raise ValueError(f"omni_image_num_tokens must be >= 0, got {image_length}")
                if bool(image_count) != bool(image_length):
                    raise ValueError(f"image count/token presence mismatch: images={image_count} tokens={image_length}")
            if self.profile.model_family == "qwen2_5_omni":
                # These wrapper tokens were already encoded in text_row.
                image_length -= 2 * image_count

            audio_length = 0
            samples = [int(value) for value in (audio_samples or [])]
            if len(samples) != audio_count:
                raise ValueError(f"audio sample-count mismatch: stats={len(samples)}, media={audio_count}")
            for num_samples in samples:
                frames = math.ceil(num_samples / self.profile.audio_hop_length)
                if self.profile.model_family == "qwen2_5_omni":
                    audio_length += ((frames - 1) // 2 + 1 - 2) // 2 + 1
                else:
                    full_seconds, remainder = divmod(frames, 100)
                    audio_length += (
                        full_seconds * self.profile.audio_tokens_per_second
                        + math.ceil(remainder / 8)
                        + self.profile.audio_tail_tokens
                    )

            estimated = text_length + image_length + audio_length + self.profile.safety_tokens
            output.append(_round_up(estimated, self.profile.round_up_to))
        return {"lengths": output}


def _build_light_dataset(dataset: Dataset, profile: PackLengthProfile) -> Dataset:
    light_columns = [name for name in ("messages", "tools") if name in dataset.column_names]
    light = dataset.select_columns(light_columns)
    required = {"omni_num_images", "omni_audio_num_samples", "omni_source_id"}
    missing = sorted(required - set(dataset.column_names))
    if missing:
        raise ValueError(f"contract_stats mode requires parquet columns: {missing}")
    image_counts = [int(value) for value in dataset["omni_num_images"]]
    audio_samples = [[int(item) for item in (value or [])] for value in dataset["omni_audio_num_samples"]]
    audio_counts = [len(value) for value in audio_samples]
    columns: dict[str, list[Any]] = {"_omni_image_count": image_counts}
    if "omni_image_num_tokens" in dataset.column_names:
        columns["_omni_image_num_tokens"] = [int(value) for value in dataset["omni_image_num_tokens"]]
    columns.update(_omni_audio_count=audio_counts, _omni_audio_num_samples=audio_samples)
    for name, values in columns.items():
        light = light.add_column(name, values)
    return light


def _estimate_lengths(dataset: Dataset, template: Any, profile: PackLengthProfile, num_proc: int) -> list[int]:
    light = _build_light_dataset(dataset, profile)
    mapped = light.map(
        _MessageLengthMapper(template, profile),
        batched=True,
        batch_size=profile.map_batch_size,
        # Do not fork a Swift template from inside an initialized torchrun
        # worker; Datasets reports only "subprocess has abruptly died".
        num_proc=None,
        remove_columns=light.column_names,
        load_from_cache_file=False,
        desc="Fast pack-length estimate",
    )
    return [int(value) for value in mapped["lengths"]]


def _distributed_cache_miss(needs_cache: bool, identity: tuple[Any, ...], error: str | None = None) -> bool:
    """All ranks must take the same branch, including nodes with warm caches."""
    states: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(states, (needs_cache, identity, error))
    errors = [state[2] for state in states if state[2] is not None]
    if errors:
        raise RuntimeError(f"fast pack-length cache lookup failed: {errors[:3]}")
    if any(state[1] != identity for state in states):
        raise ValueError("distributed pack-length sources/contracts differ across ranks")
    return any(state[0] for state in states)


def _estimate_lengths_distributed(
    dataset: Dataset,
    template: Any,
    profile: PackLengthProfile,
    num_proc: int,
) -> list[int]:
    """Use existing ranks as CPU workers; never fork an initialized CUDA process."""
    from datasets.utils import logging as datasets_logging

    rank, world_size = dist.get_rank(), dist.get_world_size()
    start = len(dataset) * rank // world_size
    end = len(dataset) * (rank + 1) // world_size
    if rank == 0:
        _info(f"Distributed fast pack-length estimate: rows={len(dataset)} ranks={world_size}")
    progress_enabled = datasets_logging.is_progress_bar_enabled()
    original_model = getattr(template, "model", None)
    template.model = None
    if rank != 0:
        datasets_logging.disable_progress_bar()
    try:
        lengths = (
            _estimate_lengths(dataset.select(range(start, end)), template, profile, num_proc) if end > start else []
        )
        result: tuple[str | None, list[int]] = (None, lengths)
    except Exception as exc:  # Broadcast any worker error so peer ranks do not hang.
        result = (f"rank {rank}: {type(exc).__name__}: {exc}", [])
    finally:
        template.model = original_model
        if progress_enabled:
            datasets_logging.enable_progress_bar()

    results: list[Any] = [None] * world_size
    dist.all_gather_object(results, result)
    errors = [error for error, _ in results if error is not None]
    if errors:
        raise RuntimeError(f"distributed pack-length estimation failed: {errors[:3]}")
    lengths = [length for _, chunk in results for length in chunk]
    if len(lengths) != len(dataset):
        raise RuntimeError("distributed pack-length result does not cover the dataset")
    return lengths


def _exact_boundary_lengths(
    dataset: Dataset,
    template: Any,
    estimated: list[int],
    *,
    source_name: str,
    profile: PackLengthProfile,
    packing_length: int,
    row_resolver: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    trust_contract_lengths: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    lengths = np.asarray(estimated, dtype=np.int64)
    keep = np.ones(len(dataset), dtype=bool)
    overflow_records: list[dict[str, Any]] = []

    def discard(
        index: int,
        exact: int | None,
        source_row: dict[str, Any],
        reason: str,
        *,
        media_row: dict[str, Any] | None = None,
    ) -> None:
        keep[index] = False
        if media_row is None:
            image_count = int(source_row.get("omni_num_images") or 0)
            audio_count = len(source_row.get("omni_audio_num_samples") or [])
        else:
            image_count = len(media_row.get("images") or []) or int(media_row.get("omni_num_images") or 0)
            audio_count = len(media_row.get("audios") or []) or len(media_row.get("omni_audio_num_samples") or [])
        overflow_records.append(
            {
                "source": source_name,
                "source_index": int(index),
                "estimated_length": int(lengths[index]),
                "exact_length": exact,
                "image_count": image_count,
                "audio_count": audio_count,
                "reason": reason,
            }
        )

    upper_bound = packing_length if trust_contract_lengths else profile.exact_encode_upper_bound
    if upper_bound is not None:
        rejected = np.flatnonzero(lengths > upper_bound).tolist()
        if rejected:
            # Rejected rows need only report statistics, never binary media.
            summary_columns = [
                name for name in ("messages", "omni_num_images", "omni_audio_num_samples") if name in dataset.column_names
            ]
            summaries = dataset.select_columns(summary_columns)
            reason = "contract_length_upper_bound" if trust_contract_lengths else "estimate_above_exact_window"
            for index in rejected:
                exact = int(lengths[index]) if trust_contract_lengths else None
                discard(index, exact, summaries[index], reason)
    if not trust_contract_lengths:
        boundary_mask = lengths >= profile.exact_encode_threshold
        if upper_bound is not None:
            boundary_mask &= lengths <= upper_bound
        boundary_indices = np.flatnonzero(boundary_mask).tolist()
        for index in tqdm(boundary_indices, desc=f"Exact boundary lengths ({source_name})", unit="row"):
            source_row = dataset[int(index)]
            row = row_resolver(source_row) if row_resolver is not None else source_row
            overflow_reason = "exact_length"
            try:
                exact = _as_int_length(template.encode(row, return_length=True)["lengths"])
            except Exception as exc:  # Swift MaxLengthError may not be imported here
                if exc.__class__.__name__ != "MaxLengthError":
                    raise
                exact = packing_length + 1
                overflow_reason = "max_length_error"
            if exact > packing_length:
                discard(index, exact, source_row, overflow_reason, media_row=row)
            else:
                lengths[index] = exact
    keep_indices = np.flatnonzero(keep).astype(np.int64)
    return lengths[keep], keep_indices, overflow_records


def _estimator_version(profile: PackLengthProfile) -> str:
    return "qwen2_5_omni_full_audio_fast_pack_v1" if profile.model_family == "qwen2_5_omni" else ESTIMATOR_VERSION


def _cache_metadata(
    dataset: Dataset,
    *,
    source: str,
    model_identity: str,
    profile: PackLengthProfile,
    packing_length: int,
    contract_hash: str,
) -> dict[str, Any]:
    payload = {
        "estimator_version": _estimator_version(profile),
        "dataset_fingerprint": dataset._fingerprint,
        "rows": len(dataset),
        "source": source,
        "model_identity": model_identity,
        "profile_hash": profile.profile_hash,
        "packing_length": packing_length,
        "contract_hash": contract_hash,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    metadata: dict[str, Any] = {**payload, "cache_key": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}
    metadata["original_rows"] = metadata.pop("rows")
    return metadata


def _validate_contract_rows(dataset: Dataset, source: str, contract_stats: ContractLengthStats) -> None:
    """Validate rows once on a cold cache; warm entries already own their identity."""

    source_ids = set(dataset["omni_source_id"])
    if source_ids != {contract_stats.source_id}:
        raise ValueError(
            f"parquet omni_source_id values {sorted(source_ids)} "
            f"do not match contract source_id={contract_stats.source_id}"
        )


def _validate_contract_compatibility(
    dataset: Dataset,
    contract_stats: ContractLengthStats,
    profile: PackLengthProfile,
) -> None:
    required_columns = set(AGENT_STATS_COLUMNS)
    if contract_stats.has_exact_media_lengths:
        required_columns.add("omni_image_num_tokens")
    missing_columns = sorted(required_columns - set(dataset.column_names))
    if missing_columns:
        raise ValueError(f"contract_stats mode requires parquet columns: {missing_columns}")
    if contract_stats.image_max_pixels <= 0 or contract_stats.image_max_pixels > profile.max_image_pixels:
        raise ValueError(
            f"dataset max_pixels={contract_stats.image_max_pixels} is outside profile limit={profile.max_image_pixels}"
        )
    if contract_stats.audio_sample_rate != profile.audio_sample_rate:
        raise ValueError(
            f"dataset audio sample_rate={contract_stats.audio_sample_rate} does not match profile={profile.audio_sample_rate}"
        )


def _run_as_writer(
    *,
    is_writer: bool,
    distributed: bool,
    operation: Callable[[], Any],
) -> tuple[str | None, Any]:
    if not is_writer:
        return None, None
    try:
        return None, operation()
    except Exception as exc:  # propagate rank-0 error collectively in distributed mode
        if not distributed:
            raise
        return f"rank 0: {type(exc).__name__}: {exc}", None


def _lookup_cached_lengths(
    dataset: Dataset,
    *,
    source: str,
    contract_stats: ContractLengthStats,
    npz_path: Path,
    json_path: Path,
    expected: dict[str, Any],
) -> CachedLengths | None:
    value = load_cached_lengths(npz_path, json_path, expected)
    if value is None:
        _validate_contract_rows(dataset, source, contract_stats)
    return value


def _build_and_save_cache(
    dataset: Dataset,
    *,
    source_name: str,
    template: Any,
    profile: PackLengthProfile,
    packing_length: int,
    num_proc: int,
    row_resolver: Callable[[dict[str, Any]], dict[str, Any]] | None,
    has_exact_media_lengths: bool,
    expected: dict[str, Any],
    npz_path: Path,
    json_path: Path,
    cached: CachedLengths | None,
    distributed_estimated: list[int] | None,
    distributed_started_at: float | None,
) -> CachedLengths:
    # A concurrent job may have published while ranks were estimating.
    value = cached if cached is not None else load_cached_lengths(npz_path, json_path, expected)
    if value is not None:
        _info(f"Fast pack-length cache hit for {source_name}: {npz_path}")
        return value
    _info(f"Building fast pack-length cache for {source_name}: {npz_path}")
    started_at = distributed_started_at or time.time()
    original_model = getattr(template, "model", None)
    template.model = None
    try:
        estimated = (
            distributed_estimated
            if distributed_estimated is not None
            else _estimate_lengths(dataset, template, profile, num_proc)
        )
        lengths, keep_indices, overflow_records = _exact_boundary_lengths(
            dataset,
            template,
            estimated,
            source_name=source_name,
            profile=profile,
            packing_length=packing_length,
            row_resolver=row_resolver,
            trust_contract_lengths=has_exact_media_lengths,
        )
    finally:
        template.model = original_model
    metadata = {
        **expected,
        "created_at": time.time(),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "kept_rows": len(keep_indices),
        "overflow_records": overflow_records,
    }
    save_cached_lengths(npz_path, json_path, lengths=lengths, keep_indices=keep_indices, metadata=metadata)
    value = load_cached_lengths(npz_path, json_path, expected)
    if value is None:
        raise RuntimeError(f"published fast pack-length cache is missing or invalid: {npz_path}")
    _info(
        f"Fast pack-length cache written for {source_name}: "
        f"rows={len(dataset)} kept={len(keep_indices)} path={npz_path}"
    )
    return value


def _attach_lengths_column(dataset: Dataset, cached: CachedLengths) -> Dataset:
    keeps_every_row = len(cached.keep_indices) == len(dataset) and np.array_equal(
        cached.keep_indices,
        np.arange(len(dataset), dtype=np.int64),
    )
    if keeps_every_row:
        dataset = dataset.add_column("lengths", cached.lengths.tolist())
    else:
        # Add the lightweight column before creating an indexed selection.
        # Dataset.add_column calls flatten_indices() for an indexed Dataset,
        # which would otherwise serialize every embedded image/audio payload.
        full_lengths = np.full(len(dataset), -1, dtype=np.int64)
        full_lengths[cached.keep_indices] = cached.lengths
        dataset = dataset.add_column("lengths", full_lengths.tolist())
        dataset = dataset.select(cached.keep_indices.tolist())
    return dataset


def prepare_source_lengths(
    dataset: Dataset,
    *,
    source: str,
    template: Any,
    profile: PackLengthProfile,
    model_identity: str,
    packing_length: int,
    num_proc: int,
    row_resolver: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[Dataset, list[dict[str, Any]]]:
    if not isinstance(dataset, Dataset):
        raise TypeError("fast pack-length estimation requires a non-streaming Hugging Face Dataset")
    if profile.cache_dir is None:
        raise ValueError("pack-length cache_dir was not resolved by the training entrypoint")
    source_name = Path(source.rstrip("/")).name or "dataset"
    contract_stats = _load_contract_stats(source, model_family=profile.model_family)
    _validate_contract_compatibility(dataset, contract_stats, profile)

    expected = _cache_metadata(
        dataset,
        source=source,
        model_identity=model_identity,
        profile=profile,
        packing_length=packing_length,
        contract_hash=contract_stats.contract_hash,
    )
    cache_key = expected["cache_key"]
    npz_path, json_path, _ = cache_paths(profile.cache_dir, source_name, cache_key)

    distributed = dist is not None and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    is_writer = not distributed or dist.get_rank() == 0

    lookup_error, cached = _run_as_writer(
        is_writer=is_writer,
        distributed=distributed,
        operation=lambda: _lookup_cached_lengths(
            dataset,
            source=source,
            contract_stats=contract_stats,
            npz_path=npz_path,
            json_path=json_path,
            expected=expected,
        ),
    )
    needs_cache = cached is None
    if distributed:
        # Rank 0 alone touches storage. All ranks agree on identity/errors and
        # branch together even if they cannot see the writer's cache mount.
        needs_cache = _distributed_cache_miss(is_writer and needs_cache, (cache_key,), lookup_error)

    distributed_estimated = None
    distributed_started_at = None
    if needs_cache and contract_stats.has_exact_media_lengths and distributed:
        distributed_started_at = time.time()
        distributed_estimated = _estimate_lengths_distributed(dataset, template, profile, num_proc)

    error, cached = _run_as_writer(
        is_writer=is_writer,
        distributed=distributed,
        operation=lambda: _build_and_save_cache(
            dataset,
            source_name=source_name,
            template=template,
            profile=profile,
            packing_length=packing_length,
            num_proc=num_proc,
            row_resolver=row_resolver,
            has_exact_media_lengths=contract_stats.has_exact_media_lengths,
            expected=expected,
            npz_path=npz_path,
            json_path=json_path,
            cached=cached,
            distributed_estimated=distributed_estimated,
            distributed_started_at=distributed_started_at,
        ),
    )
    if distributed:
        # Both success and ordinary I/O/encoding failures follow this collective.
        # Peers never wait at a barrier while the publisher exits with an error.
        result: list[Any] = [(error, cached)] if is_writer else [None]
        dist.broadcast_object_list(result, src=0)
        error, cached = result[0]
        if error is not None:
            raise RuntimeError(f"fast pack-length cache preparation failed: {error}")
    if cached is None:
        raise RuntimeError(f"fast pack-length cache is missing or invalid: {npz_path}")

    dataset = _attach_lengths_column(dataset, cached)
    overflow_records = list(cached.metadata.get("overflow_records") or [])
    return dataset, overflow_records
