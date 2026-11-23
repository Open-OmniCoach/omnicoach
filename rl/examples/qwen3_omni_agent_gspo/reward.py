"""Pure outcome GSPO reward, filtering, conversion, and logging for Example 2.

Uses only ``slime_ext.agent.rewards``; zero cross-example imports.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from slime_ext.agent.rewards import (
    compute_outcome_group_advantages,
    convert_agent_samples_to_train_data,
    filter_outcome_group,
    log_agent_metrics,
    reject_runtime_failures,
    score_outcome_sample,
)


async def score(args: Any, sample: Any, **_kwargs: Any) -> float | list[float]:
    """Slime ``--custom-rm-path``: pure outcome (+ optional format reward)."""
    cfg = args.agent_config
    fmt = float(cfg.get("format_reward", 0.0))
    gate = bool(cfg.get("tg_complete_event_gate", True))
    if isinstance(sample, list):
        return [score_outcome_sample(s, format_reward=fmt, complete_event_gate=gate) for s in sample]
    return score_outcome_sample(sample, format_reward=fmt, complete_event_gate=gate)


def convert_samples_to_train_data(args: Any, samples: Sequence[Any]) -> dict[str, Any]:
    """Slime ``--custom-convert-samples-to-train-data-path`` using pure outcome GSPO."""
    return convert_agent_samples_to_train_data(
        args,
        samples,
        advantage_shaper=compute_outcome_group_advantages,
    )


def filter_group(args: Any, samples: Sequence[Any], **kwargs: Any):
    """Slime ``--dynamic-sampling-filter-path``: runtime failure check + nonzero outcome std."""
    return filter_outcome_group(args, samples, **kwargs)


def log_metrics(
    rollout_id: int,
    args: Any,
    samples: Sequence[Any],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float,
) -> bool:
    """Slime ``--custom-rollout-log-function-path``."""
    return log_agent_metrics(rollout_id, args, samples, rollout_extra_metrics, rollout_time)


__all__ = [
    "convert_samples_to_train_data",
    "filter_group",
    "log_metrics",
    "reject_runtime_failures",
    "score",
]
