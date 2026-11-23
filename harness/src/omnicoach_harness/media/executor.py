"""`MediaExecutor`, which runs decode work in a bounded subprocess pool."""

from __future__ import annotations

import importlib
import multiprocessing
import threading
from collections.abc import Callable
from functools import wraps
from typing import Any

_executor: MediaExecutor | None = None


def _execute(module: str, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    return getattr(importlib.import_module(module), name).__wrapped__(*args, **kwargs)


class MediaExecutor:
    def __init__(self, *, workers: int, queue_size: int, decode_bytes: int) -> None:
        if workers < 1 or queue_size < 0 or decode_bytes < 1:
            raise ValueError("invalid media executor budget")
        self.capacity = decode_bytes
        self.used = 0
        self.active = 0
        self.waiting = 0
        self.condition = threading.Condition()
        self.slots = threading.BoundedSemaphore(workers + queue_size)
        # One job per child provides deterministic native-allocator reclamation.
        self.pool = multiprocessing.get_context("spawn").Pool(workers, maxtasksperchild=1)

    def _estimate_weight(
        self,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> int:
        from .probe import probe_video

        weight = 64 * 1024**2
        if function.__name__ not in {"extract_frame", "extract_frame_batch"}:
            return weight
        source = probe_video(
            args[0] if args else kwargs["video_path"],
            ffprobe=str(kwargs.get("ffprobe", "ffprobe")),
        )
        per_frame_bytes = max(1, source.width * source.height * 9)
        if function.__name__ == "extract_frame":
            frame_count = 1
        else:
            batch_size = int(kwargs.get("batch_size", 16))
            timestamps = kwargs.get("timestamps")
            frame_count = min(batch_size, max(1, len(timestamps))) if timestamps is not None else batch_size
        weight = max(weight, per_frame_bytes * frame_count)
        if weight > self.capacity and function.__name__ == "extract_frame_batch":
            kwargs["batch_size"] = max(1, self.capacity // per_frame_bytes)
            weight = per_frame_bytes * kwargs["batch_size"]
        return weight

    def dispatch(self, function: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        weight = self._estimate_weight(function, args, kwargs)
        if weight > self.capacity:
            raise ValueError("one source frame exceeds configured decode byte budget")
        with self.condition:
            self.waiting += 1
        with self.slots:
            with self.condition:
                self.condition.wait_for(lambda: self.used + weight <= self.capacity)
                self.used += weight
                self.active += 1
                self.waiting -= 1
            try:
                return self.pool.apply(_execute, (function.__module__, function.__name__, args, kwargs))
            finally:
                with self.condition:
                    self.used -= weight
                    self.active -= 1
                    self.condition.notify_all()

    def snapshot(self) -> dict[str, int]:
        with self.condition:
            return {"submitted": self.active, "waiting": self.waiting, "decode_bytes": self.used}

    def close(self) -> None:
        global _executor
        self.pool.close()
        self.pool.join()
        if _executor is self:
            _executor = None


def configure_media_executor(**kwargs: Any) -> MediaExecutor:
    global _executor
    if _executor is not None:
        raise RuntimeError("media executor already configured")
    _executor = MediaExecutor(**kwargs)
    return _executor


def offload_to_media_pool(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if _executor is None:
            return function(*args, **kwargs)
        if "timestamps" in kwargs:
            kwargs["timestamps"] = list(kwargs["timestamps"])
        return _executor.dispatch(function, args, kwargs)

    return wrapped
