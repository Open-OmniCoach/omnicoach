"""Attempt and time counters for one run, and the `ProviderBudgetExhausted` they raise."""

from __future__ import annotations

import threading
import time
from collections import Counter

_lock = threading.Lock()
_counts: Counter[str] = Counter()
_limit: int | None = None
_deadline: float | None = None


class ProviderBudgetExhausted(RuntimeError):
    """A run provider budget is exhausted; do not retry the same request."""


def configure_budget(*, attempts: int | None = None, seconds: float | None = None) -> None:
    global _limit, _deadline
    if attempts is not None and attempts < 1:
        raise ValueError("provider attempt budget must be >= 1 when provided")
    if seconds is not None and seconds <= 0:
        raise ValueError("provider time budget seconds must be > 0 when provided")
    with _lock:
        _limit = attempts
        _deadline = time.monotonic() + seconds if seconds is not None else None


def reset() -> None:
    """Clear telemetry counters and budget limits (intended for tests and fresh runs)."""

    global _limit, _deadline
    with _lock:
        _counts.clear()
        _limit = None
        _deadline = None


def record(model: str, event: str, *, entry: str | None = None) -> None:
    with _lock:
        if event == "attempt":
            if (_limit is not None and _counts["attempt"] >= _limit) or (
                _deadline is not None and time.monotonic() >= _deadline
            ):
                raise ProviderBudgetExhausted("campaign provider attempt/time budget exhausted")
            _counts["attempt"] += 1
        _counts[f"{model}:{event}"] += 1
        if entry is not None:
            # Only configured, non-secret entry names; one attempt budget debit.
            _counts[f"{model}:entry:{entry}:{event}"] += 1


def snapshot() -> dict[str, int]:
    with _lock:
        return dict(_counts)
