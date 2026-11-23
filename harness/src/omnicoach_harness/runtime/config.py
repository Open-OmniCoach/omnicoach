"""`RunConfig`, `BatchOptions`, `MediaOptions` and `OperationalConfig` for YAML run files."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .sampling import MAX_PROVIDER_SEED

_DEFAULT_REPLACEMENT_BATCH_SIZE = 128
_DEFAULT_MEDIA_QUEUE_SIZE = 8
_DEFAULT_MEDIA_DECODE_BYTES = 2 * 1024**3

_TOP_LEVEL_KEYS = frozenset({"stage", "route", "io", "provider", "execution", "sampling", "system", "operational"})
_ROUTE_KEYS = frozenset({"entrypoint"})
_IO_KEYS = frozenset({"input", "output_dir"})
_EXECUTION_KEYS = frozenset({"workers", "rollouts", "attempts", "retry_errors"})
_SAMPLING_KEYS = frozenset({"base_seed", "temperature", "top_p"})
_SYSTEM_KEYS = frozenset({"ffmpeg", "ffprobe"})
_PROVIDER_FIXED_KEYS = frozenset(
    {"model", "backend", "max_tokens", "retries", "retry_429_forever", "request_timeout_seconds"}
)
_OPERATIONAL_KEYS = frozenset(
    {
        "task_filter_path",
        "run_key_filter_path",
        "defer_run_keys_path",
        "media_cache_dir",
        "work_dir",
        "intake_stop_path",
        "max_concurrent_starts",
        "retry_replacement_batch_size",
        "fast_resume",
        "defer_scoring",
        "media_workers",
        "media_queue_size",
        "media_decode_bytes",
        "provider_attempt_budget",
        "provider_time_budget_seconds",
        "telemetry",
        "telemetry_dir",
    }
)


def _check_keys(section: str, mapping: Mapping[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unsupported {section} options: {unknown}")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


@dataclass(frozen=True, slots=True, kw_only=True)
class MediaOptions:
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    media_cache_dir: str | Path | None = None
    work_dir: str | Path | None = None
    workers: int | None = None
    queue_size: int = _DEFAULT_MEDIA_QUEUE_SIZE
    decode_bytes: int = _DEFAULT_MEDIA_DECODE_BYTES

    def __post_init__(self) -> None:
        if not self.ffmpeg.strip() or not self.ffprobe.strip():
            raise ValueError("ffmpeg and ffprobe must not be empty")
        if self.workers is not None and self.workers < 1:
            raise ValueError("media workers must be >= 1 when provided")
        if self.queue_size < 0:
            raise ValueError("media queue_size must be >= 0")
        if self.decode_bytes < 1:
            raise ValueError("media decode_bytes must be >= 1")


@dataclass(frozen=True, slots=True, kw_only=True)
class BatchOptions:
    output_dir: str | Path
    workers: int = 1
    rollouts: int = 1
    attempts: int = 1
    retry_errors: bool = False
    fast_resume: bool = False
    sampling_seed_base: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_concurrent_starts: int | None = None
    retry_replacement_batch_size: int = _DEFAULT_REPLACEMENT_BATCH_SIZE
    run_key_filter: frozenset[str] | None = None
    defer_run_keys: frozenset[str] | None = None
    intake_stop_path: str | Path | None = None
    defer_scoring: bool = False

    def __post_init__(self) -> None:
        for name in ("workers", "rollouts", "attempts", "retry_replacement_batch_size"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.fast_resume and self.retry_errors:
            raise ValueError("fast_resume is append-only; retry errors in a successor")
        if self.max_concurrent_starts is not None and self.max_concurrent_starts < 1:
            raise ValueError("max_concurrent_starts must be >= 1 when provided")
        if self.sampling_seed_base is not None and not 0 <= self.sampling_seed_base <= MAX_PROVIDER_SEED:
            raise ValueError("sampling.base_seed must be between 0 and 2**31-1")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("sampling.temperature must be >= 0")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("sampling.top_p must be in (0, 1]")


@dataclass(frozen=True, slots=True, kw_only=True)
class OperationalConfig:
    task_filter_path: str | None = None
    run_key_filter_path: str | None = None
    defer_run_keys_path: str | None = None
    media_cache_dir: str | None = None
    work_dir: str | None = None
    intake_stop_path: str | None = None
    max_concurrent_starts: int | None = None
    retry_replacement_batch_size: int = _DEFAULT_REPLACEMENT_BATCH_SIZE
    fast_resume: bool = False
    defer_scoring: bool = False
    media_workers: int | None = None
    media_queue_size: int = _DEFAULT_MEDIA_QUEUE_SIZE
    media_decode_bytes: int = _DEFAULT_MEDIA_DECODE_BYTES
    provider_attempt_budget: int | None = None
    provider_time_budget_seconds: float | None = None
    telemetry: bool = False
    telemetry_dir: str | None = None

    def __post_init__(self) -> None:
        if self.max_concurrent_starts is not None and self.max_concurrent_starts < 1:
            raise ValueError("operational.max_concurrent_starts must be >= 1")
        if self.retry_replacement_batch_size < 1:
            raise ValueError("operational.retry_replacement_batch_size must be >= 1")
        if self.media_workers is not None and self.media_workers < 1:
            raise ValueError("operational.media_workers must be >= 1")
        if self.media_queue_size < 0:
            raise ValueError("operational.media_queue_size must be >= 0")
        if self.media_decode_bytes < 1:
            raise ValueError("operational.media_decode_bytes must be >= 1")
        if self.provider_attempt_budget is not None and self.provider_attempt_budget < 1:
            raise ValueError("operational.provider_attempt_budget must be >= 1")
        if self.provider_time_budget_seconds is not None and self.provider_time_budget_seconds <= 0:
            raise ValueError("operational.provider_time_budget_seconds must be > 0")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> OperationalConfig:
        if not raw:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("operational section must be a mapping")
        _check_keys("operational", raw, _OPERATIONAL_KEYS)
        return cls(
            task_filter_path=_optional_str(raw.get("task_filter_path")),
            run_key_filter_path=_optional_str(raw.get("run_key_filter_path")),
            defer_run_keys_path=_optional_str(raw.get("defer_run_keys_path")),
            media_cache_dir=_optional_str(raw.get("media_cache_dir")),
            work_dir=_optional_str(raw.get("work_dir")),
            intake_stop_path=_optional_str(raw.get("intake_stop_path")),
            max_concurrent_starts=_optional_int(raw.get("max_concurrent_starts")),
            retry_replacement_batch_size=int(
                raw["retry_replacement_batch_size"]
                if raw.get("retry_replacement_batch_size") is not None
                else _DEFAULT_REPLACEMENT_BATCH_SIZE
            ),
            fast_resume=bool(raw.get("fast_resume", False)),
            defer_scoring=bool(raw.get("defer_scoring", False)),
            media_workers=_optional_int(raw.get("media_workers")),
            media_queue_size=int(
                raw["media_queue_size"]
                if raw.get("media_queue_size") is not None
                else _DEFAULT_MEDIA_QUEUE_SIZE
            ),
            media_decode_bytes=int(
                raw["media_decode_bytes"]
                if raw.get("media_decode_bytes") is not None
                else _DEFAULT_MEDIA_DECODE_BYTES
            ),
            provider_attempt_budget=_optional_int(raw.get("provider_attempt_budget")),
            provider_time_budget_seconds=_optional_float(raw.get("provider_time_budget_seconds")),
            telemetry=bool(raw.get("telemetry", False)),
            telemetry_dir=_optional_str(raw.get("telemetry_dir")),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RunConfig:
    stage: str
    route_entrypoint: str
    input_path: str
    output_dir: str
    model: str
    provider_backend: str = "openai_compat"
    provider_options: Mapping[str, Any] = field(default_factory=dict)
    max_tokens: int = 32768
    retries: int = 10
    retry_429_forever: bool = True
    request_timeout_seconds: float = 300.0
    workers: int = 1
    rollouts: int = 1
    attempts: int = 1
    retry_errors: bool = False
    sampling_seed_base: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    operational: OperationalConfig = field(default_factory=OperationalConfig)

    def __post_init__(self) -> None:
        if self.stage != "actor":
            raise ValueError("stage must be actor")
        if not self.provider_backend.strip():
            raise ValueError("provider.backend must not be empty")
        if ":" not in self.route_entrypoint:
            raise ValueError("route_entrypoint must use module:function syntax")
        for name in ("max_tokens", "retries", "workers", "rollouts", "attempts"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be > 0")
        if self.sampling_seed_base is not None and not 0 <= self.sampling_seed_base <= MAX_PROVIDER_SEED:
            raise ValueError("sampling.base_seed must be between 0 and 2**31-1")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("sampling.temperature must be >= 0")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("sampling.top_p must be in (0, 1]")

    def media_options(self) -> MediaOptions:
        return MediaOptions(
            ffmpeg=self.ffmpeg,
            ffprobe=self.ffprobe,
            media_cache_dir=self.operational.media_cache_dir,
            work_dir=self.operational.work_dir,
            workers=self.operational.media_workers,
            queue_size=self.operational.media_queue_size,
            decode_bytes=self.operational.media_decode_bytes,
        )

    def batch_options(
        self,
        *,
        run_key_filter: set[str] | frozenset[str] | None = None,
        defer_run_keys: set[str] | frozenset[str] | None = None,
    ) -> BatchOptions:
        return BatchOptions(
            output_dir=self.output_dir,
            workers=self.workers,
            rollouts=self.rollouts,
            attempts=self.attempts,
            retry_errors=self.retry_errors,
            fast_resume=self.operational.fast_resume,
            sampling_seed_base=self.sampling_seed_base,
            temperature=self.temperature,
            top_p=self.top_p,
            max_concurrent_starts=self.operational.max_concurrent_starts,
            retry_replacement_batch_size=self.operational.retry_replacement_batch_size,
            run_key_filter=frozenset(run_key_filter) if run_key_filter is not None else None,
            defer_run_keys=frozenset(defer_run_keys) if defer_run_keys is not None else None,
            intake_stop_path=self.operational.intake_stop_path,
            defer_scoring=self.operational.defer_scoring,
        )

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
        if not isinstance(value, dict):
            raise ValueError("run config must contain a YAML object")
        _check_keys("top-level", value, _TOP_LEVEL_KEYS)
        route = value.get("route")
        if not isinstance(route, dict):
            raise ValueError("run config requires route.entrypoint")
        _check_keys("route", route, _ROUTE_KEYS)
        provider = value.get("provider") or {}
        execution = value.get("execution") or {}
        sampling = value.get("sampling") or {}
        io = value.get("io") or {}
        system = value.get("system") or {}
        for name, section, allowed in (
            ("io", io, _IO_KEYS),
            ("execution", execution, _EXECUTION_KEYS),
            ("sampling", sampling, _SAMPLING_KEYS),
            ("system", system, _SYSTEM_KEYS),
        ):
            if not isinstance(section, Mapping):
                raise ValueError(f"{name} section must be a mapping")
            _check_keys(name, section, allowed)
        if not isinstance(provider, Mapping):
            raise ValueError("provider section must be a mapping")
        return cls(
            stage=str(value.get("stage") or ""),
            route_entrypoint=str(route.get("entrypoint") or ""),
            input_path=str(io.get("input") or ""),
            output_dir=str(io.get("output_dir") or ""),
            model=str(provider.get("model") or ""),
            provider_backend=str(provider.get("backend") or "openai_compat"),
            provider_options={
                str(key): item
                for key, item in provider.items()
                if key not in _PROVIDER_FIXED_KEYS
            },
            max_tokens=int(provider.get("max_tokens", 32768)),
            retries=int(provider.get("retries", 10)),
            retry_429_forever=bool(provider.get("retry_429_forever", True)),
            request_timeout_seconds=float(provider.get("request_timeout_seconds", 300.0)),
            workers=int(execution.get("workers", 1)),
            rollouts=int(execution.get("rollouts", 1)),
            attempts=int(execution.get("attempts", 1)),
            retry_errors=bool(execution.get("retry_errors", False)),
            sampling_seed_base=_optional_int(sampling.get("base_seed")),
            temperature=_optional_float(sampling.get("temperature")),
            top_p=_optional_float(sampling.get("top_p")),
            ffmpeg=str(system.get("ffmpeg") or "ffmpeg"),
            ffprobe=str(system.get("ffprobe") or "ffprobe"),
            operational=OperationalConfig.from_mapping(value.get("operational")),
        )
