"""`rollout_seed`, deriving a deterministic seed per task and rollout."""

from __future__ import annotations

MAX_PROVIDER_SEED = 2**31 - 1


def _fnv1a64(value: bytes) -> int:
    result = 14695981039346656037
    for byte in value:
        result ^= byte
        result = (result * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return result


def rollout_seed(*, base_seed: int, task_id: str, rollout_index: int, rollouts: int) -> int:
    """Map one campaign seed to distinct stable seeds within each task."""

    if not 0 <= base_seed <= MAX_PROVIDER_SEED:
        raise ValueError("base_seed must be between 0 and 2**31-1")
    if not task_id:
        raise ValueError("task_id must be non-empty")
    if not 1 <= rollouts <= MAX_PROVIDER_SEED + 1:
        raise ValueError("rollouts is outside the supported seed space")
    if not 0 <= rollout_index < rollouts:
        raise ValueError("rollout_index must be in [0, rollouts)")

    buckets = (MAX_PROVIDER_SEED + 1) // rollouts
    task_bucket = _fnv1a64(f"{base_seed}\0{task_id}".encode()) % buckets
    return task_bucket * rollouts + rollout_index
