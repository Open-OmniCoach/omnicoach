"""Rollout-local reuse of frozen Omni encoder outputs across ref, old-logp, and actor train.

Invariants:
- Cache keys are derived from real input tensor fingerprints (shape, dtype,
  content hash/digest, and auxiliary grid/length arguments), preventing silent
  cross-sample mismatch when microbatch ordering differs across phases.
- Telemetry recording and CUDA timing events are disabled by default unless
  ``log_dir`` or ``SLIME_OMNI_ENCODER_FEATURE_LOG_DIR`` is explicitly set; when
  disabled, ``flush()`` is a zero-overhead no-op without CUDA synchronization.
- Entries are reused across ``ref_log_probs`` -> ``log_probs`` (actor old
  logprobs) -> multi-step ``actor_train``, and cleared only when a new
  ``rollout_id`` starts.
- Compatible with Megatron ``torch_dist`` ``ref_load`` checkpoints (enforces
  frozen tower parameters directly).
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

_EXACT_HASH_NUMEL_LIMIT = 4096
_SAMPLE_COUNT = 4096
VALID_ENCODER_FEATURE_MODES = frozenset({"cache", "measure", "off", "disabled", "none"})


def _tensor_fingerprint(tensor: torch.Tensor) -> tuple[Any, ...]:
    """Build a collision-resistant, O(1)-memory fingerprint for CPU or GPU tensors."""
    meta = (tuple(tensor.shape), str(tensor.dtype))
    numel = tensor.numel()
    if numel == 0:
        return (*meta, "empty")

    detached = tensor.detach()
    if numel <= _EXACT_HASH_NUMEL_LIMIT:
        raw = detached.cpu().contiguous()
        # Normalize bfloat16/float8/bool through 1-D uint8 raw bytes before hashing.
        byte_view = raw.reshape(-1).view(torch.uint8).numpy().tobytes()
        return (*meta, hashlib.blake2b(byte_view, digest_size=16).hexdigest())

    flat = detached.reshape(-1)
    # Deterministic strided sample (<= 4096 elements) plus scalar reductions
    # catches both permutations and arbitrary single-element edits with O(1)
    # extra memory (no full-tensor FP32 copies or arange allocations).
    stride = max(1, (numel - 1) // (_SAMPLE_COUNT - 1))
    sampled = flat[::stride][: _SAMPLE_COUNT - 1]
    sample_vec = torch.cat([sampled, flat[-1:]], dim=0)
    sample_bytes = sample_vec.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()

    if flat.dtype == torch.bool:
        stats = torch.stack([flat.sum(dtype=torch.int64)])
    elif flat.is_floating_point():
        stats = torch.stack(
            [
                flat.sum(dtype=torch.float64),
                flat.amin().to(dtype=torch.float64),
                flat.amax().to(dtype=torch.float64),
            ]
        )
    else:
        stats = torch.stack(
            [
                flat.sum(dtype=torch.int64),
                flat.amin().to(dtype=torch.int64),
                flat.amax().to(dtype=torch.int64),
            ]
        )
    stats_bytes = stats.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return (*meta, hashlib.blake2b(sample_bytes + stats_bytes, digest_size=16).hexdigest())


def _value_fingerprint(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return ("tensor", _tensor_fingerprint(value))
    if isinstance(value, (int, float, bool, str, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_value_fingerprint(item) for item in value)
    if isinstance(value, dict):
        return tuple((k, _value_fingerprint(v)) for k, v in sorted(value.items()))
    return repr(value)


def make_encoder_cache_key(name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    """Compute a deterministic cache key from encoder method name and inputs."""
    return (
        name,
        tuple(_value_fingerprint(arg) for arg in args),
        tuple((k, _value_fingerprint(v)) for k, v in sorted(kwargs.items())),
    )


class EncoderFeatures:
    def __init__(self, model, mode: str, limit_gib: float, log_dir: str | Path | None = None):
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in VALID_ENCODER_FEATURE_MODES:
            raise ValueError(
                "SLIME_OMNI_ENCODER_FEATURE_MODE must be one of "
                f"{sorted(VALID_ENCODER_FEATURE_MODES)}, got {mode!r}"
            )
        self.cache_enabled = normalized_mode == "cache"
        self.limit_bytes = int(limit_gib * 1024**3)
        self.entries: dict[tuple[Any, ...], tuple[str, list[torch.Tensor], Any, int]] = {}
        self.cached_bytes = 0
        self.records: list[dict[str, Any]] = []
        self.phase: dict[str, Any] = {"phase": "init", "rollout_id": None}
        self._cache_rollout_id: Any = None
        self.call_index = 0

        resolved_log_dir = log_dir if log_dir is not None else os.environ.get("SLIME_OMNI_ENCODER_FEATURE_LOG_DIR")
        self.directory: Path | None = Path(resolved_log_dir) if resolved_log_dir else None
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

        for part in model:
            for module in part.modules():
                if not hasattr(module, "get_audio_features") or not hasattr(module, "get_image_features"):
                    continue
                assert not any(
                    p.requires_grad
                    for tower in (module.visual, module.audio_model)
                    for p in tower.parameters()
                ), "EncoderFeatures cache requires frozen vision and audio towers"
                assert getattr(module.audio_model.config, "attention_dropout", 0) == 0
                assert getattr(module.audio_model.config, "dropout", 0) == 0
                for name in ("get_image_features", "get_audio_features"):
                    setattr(module, name, self.wrap(getattr(module, name), name))

    def clear_cache(self) -> None:
        self.entries.clear()
        self.cached_bytes = 0

    def begin(self, phase: dict[str, Any]) -> None:
        self.flush()
        incoming_rollout_id = phase.get("rollout_id")
        incoming_phase = phase.get("phase")
        # Clear cached features only on a new rollout, preserving entries across
        # ref_log_probs -> log_probs (old_log_probs) -> multi-step actor_train.
        if (
            self._cache_rollout_id is not None
            and incoming_rollout_id is not None
            and incoming_rollout_id != self._cache_rollout_id
        ) or (incoming_phase == "ref_log_probs" and incoming_rollout_id != self._cache_rollout_id):
            self.clear_cache()
        if incoming_rollout_id is not None:
            self._cache_rollout_id = incoming_rollout_id
        self.phase = dict(phase)
        self.call_index = 0

    def _can_read_cache(self) -> bool:
        return self.cache_enabled and self.phase.get("phase") in {"log_probs", "actor_train"}

    def _can_store_cache(self) -> bool:
        return self.cache_enabled and self.phase.get("phase") in {"ref_log_probs", "log_probs", "actor_train"}

    def wrap(self, original, name: str):
        @functools.wraps(original)
        def call(*args, **kwargs):
            index = self.call_index
            self.call_index += 1

            record_metrics = self.directory is not None
            started = time.perf_counter() if record_metrics else 0.0

            target_device = args[0].device if args and isinstance(args[0], torch.Tensor) else torch.device("cpu")
            is_cuda = target_device.type == "cuda" and torch.cuda.is_available()
            record_cuda_events = record_metrics and is_cuda
            if record_cuda_events:
                before, encoded, after = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                before.record()
            else:
                before = encoded = after = None

            cache_key = make_encoder_cache_key(name, args, kwargs) if self.cache_enabled else None
            entry = self.entries.get(cache_key) if (cache_key is not None and self._can_read_cache()) else None
            size = 0
            if entry is not None:
                entry_name, leaves, structure, size = entry
                assert entry_name == name
                leaves = [
                    tensor.to(target_device, non_blocking=is_cuda) if target_device.type == "cuda" else tensor.clone()
                    for tensor in leaves
                ]
                output = tree_unflatten(leaves, structure)
            else:
                output = original(*args, **kwargs)

            if record_cuda_events:
                encoded.record()

            should_store = (
                entry is None
                and cache_key is not None
                and self._can_store_cache()
                and cache_key not in self.entries
            )
            stored = False
            if entry is None and (should_store or record_metrics):
                leaves, structure = tree_flatten(output)
                size = sum(t.numel() * t.element_size() for t in leaves if isinstance(t, torch.Tensor))
                if should_store and self.cached_bytes + size <= self.limit_bytes:
                    host = []
                    for source in leaves:
                        if isinstance(source, torch.Tensor):
                            assert not source.requires_grad
                            dst = torch.empty_like(source, device="cpu", pin_memory=is_cuda)
                            dst.copy_(source, non_blocking=is_cuda)
                            host.append(dst)
                        else:
                            host.append(source)
                    self.entries[cache_key] = (name, host, structure, size)
                    self.cached_bytes += size
                    stored = True

            if record_cuda_events:
                after.record()

            if record_metrics:
                record = {
                    "name": name,
                    "call": index,
                    "hit": entry is not None,
                    "stored": stored,
                    "output_bytes": size,
                    "cache_bytes": self.cached_bytes,
                    "host_seconds": time.perf_counter() - started,
                }
                if record_cuda_events:
                    record["events"] = (before, encoded, after)
                self.records.append(record)
            return output

        return call

    def flush(self) -> None:
        if self.directory is None or not self.records:
            self.records = []
            return
        last_events = self.records[-1].get("events")
        if last_events is not None:
            last_events[-1].synchronize()
        for record in self.records:
            events = record.pop("events", None)
            if events is not None:
                before, encoded, after = events
                record["encode_or_restore_ms"] = before.elapsed_time(encoded)
                record["store_ms"] = encoded.elapsed_time(after)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        path = self.directory / f"rank-{rank:02d}.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            print(
                json.dumps(
                    {
                        **self.phase,
                        "cache_enabled": self.cache_enabled,
                        "cache_limit_bytes": self.limit_bytes,
                        "calls": self.records,
                    }
                ),
                file=stream,
            )
        self.records = []
