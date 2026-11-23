"""Outcome rewards, runtime filtering, and GSPO batch conversion for Omni agents.

Provides:
- Pure outcome scoring for MCQ and ``span_set`` temporal grounding tasks
  (using shared interval & 1-D Wasserstein primitives from ``intervals.py``,
  with optional signed ``format_reward`` defaulting to 0.0).
- Explicit group-indexed pure outcome GSPO advantage calculation.
- Runtime failure filtering rejecting ``CONTEXT_LENGTH_EXCEEDED`` and
  ``MEDIA_TOOL_FAILED`` groups prior to reward-variance filtering.
- Candidate-level metric logging safe against ``rollout_extra_metrics=None``
  and invariant to per-trajectory turn counts.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .intervals import merge_intervals, overlap_duration, temporal_iou, w1_distance
from .rollout import _status_name, ensure_valid_top_p_fields

_SPAN_INTERVAL_RE = r"\[\s*(\d+(?:\.\d+)?)s\s*,\s*(\d+(?:\.\d+)?)s\s*\]"
_RUNTIME_FAILURE_REASONS = frozenset({"CONTEXT_LENGTH_EXCEEDED", "MEDIA_TOOL_FAILED"})


def _parse_seconds_value(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean time value")
    fields = [float(part) for part in str(value).strip().split(":")]
    if len(fields) == 1:
        return fields[0]
    if len(fields) == 2:
        return fields[0] * 60.0 + fields[1]
    if len(fields) == 3:
        return fields[0] * 3600.0 + fields[1] * 60.0 + fields[2]
    raise ValueError(f"invalid time value {value!r}")


def _parse_structured_spans(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("invalid span collection")
    if len(value) == 2 and all(
        isinstance(item, (int, float, str)) and not isinstance(item, bool) for item in value
    ):
        value = [value]
    spans: list[tuple[float, float]] = []
    for item in value:
        if isinstance(item, Mapping):
            start = item.get("start_seconds", item.get("start"))
            end = item.get("end_seconds", item.get("end"))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2:
            start, end = item
        else:
            raise ValueError("invalid span item")
        sf, ef = _parse_seconds_value(start), _parse_seconds_value(end)
        if not (math.isfinite(sf) and math.isfinite(ef)) or sf < 0.0 or ef <= sf:
            raise ValueError(f"invalid interval ({sf}, {ef})")
        spans.append((sf, ef))
    if not spans:
        raise ValueError("empty span collection")
    return spans


def _parse_prediction_spans(answer: str) -> list[tuple[float, float]]:
    full_pattern = _SPAN_INTERVAL_RE + r"(?:\s*;\s*" + _SPAN_INTERVAL_RE + r")*"
    if re.fullmatch(full_pattern, answer.strip()) is None:
        raise ValueError("unparseable span_set answer")
    pairs = [[float(a), float(b)] for a, b in re.findall(_SPAN_INTERVAL_RE, answer)]
    return _parse_structured_spans(pairs)


def score_mcq(answer: str, label: Any) -> dict[str, Any]:
    if re.fullmatch(r"[A-Za-z]", answer.strip()) is None:
        raise ValueError("MCQ answer must be a single option letter")
    cleaned = answer.strip()
    correct = cleaned.upper() == str(label).strip().upper()
    return {
        "outcome_score": float(correct),
        "correct": correct,
        "canonical_format": cleaned == cleaned.upper(),
    }


def score_span_set(
    answer: str,
    label: Any,
    duration_s: float,
    *,
    complete_event_gate: bool = False,
) -> dict[str, Any]:
    pred = _parse_prediction_spans(answer)
    if any(end > duration_s + 1e-6 for _, end in pred):
        raise ValueError("prediction interval exceeds media duration")
    target = _parse_structured_spans(label)
    pred_set, target_set = merge_intervals(pred), merge_intervals(target)
    target_dur = math.fsum(e - s for s, e in target_set)
    set_tiou = temporal_iou(pred_set, target_set)
    dist = w1_distance(pred_set, target_set)
    tw = min(1.0, max(0.0, math.exp(-dist / (target_dur + 1e-12))))
    coverages = [overlap_duration(pred_set, [(s, e)]) / (e - s) for s, e in target]
    min_cov = min(coverages)
    quality = 0.5 * (set_tiou + tw)
    complete_pass = set_tiou >= 0.5 and min_cov >= 0.5
    outcome = (0.8 * float(complete_pass) + 0.2 * quality) if complete_event_gate else quality
    return {
        "outcome_score": float(outcome),
        "correct": complete_pass,
        "canonical_format": True,
        "metrics": {
            "set_tiou": set_tiou,
            "temporal_wasserstein": tw,
            "temporal_wasserstein_distance": dist,
            "min_gt_span_coverage": min_cov,
            "quality_score": quality,
        },
    }


def score_outcome_sample(
    sample: Any,
    *,
    format_reward: float = 0.0,
    complete_event_gate: bool = False,
) -> float:
    """Score one completed agent trajectory (pure outcome + optional signed format reward)."""
    metadata = sample.metadata
    components: dict[str, Any] = {
        "outcome_score": 0.0,
        "correct": False,
        "canonical_format": False,
    }
    answer = metadata.get("answer")
    if _status_name(sample.status) == "COMPLETED" and answer is not None:
        answer_type = metadata.get("answer_type", "mcq")
        try:
            if answer_type == "mcq":
                components = score_mcq(str(answer), sample.label)
            elif answer_type == "span_set":
                components = score_span_set(
                    str(answer),
                    sample.label,
                    float(metadata["media_duration_seconds"]),
                    complete_event_gate=complete_event_gate,
                )
            else:
                raise ValueError(f"unsupported answer_type {answer_type!r}")
        except ValueError:
            pass

    format_ok = bool(
        components["canonical_format"]
        and int(metadata.get("invalid_actions", 0)) == 0
        and int(metadata.get("no_progress", 0)) == 0
    )
    mag = float(format_reward)
    signed_fmt = (mag if format_ok else -mag) if mag > 0.0 else 0.0
    total = float(components["outcome_score"]) + signed_fmt
    metadata["reward_components"] = {
        **components,
        "format_ok": format_ok,
        "format_reward": signed_fmt,
        "total": total,
    }
    sample.reward = total
    for turn in metadata.get("turns", ()):
        turn.reward = total
    return total


# ---------------------------------------------------------------------------
# Group outcome advantages, runtime filtering & Slime BatchBuilder conversion
# ---------------------------------------------------------------------------


def compute_outcome_group_advantages(
    args: Any,
    samples: Sequence[Any],
) -> tuple[list[float], list[list[float]]]:
    """Pure outcome GSPO group centering broadcast uniformly across each trajectory's turns."""
    raw_rewards = [float(sample.get_reward_value(args)) for sample in samples]
    groups: dict[Any, list[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        gid = getattr(sample, "group_index", None)
        if gid is None:
            raise ValueError("group outcome advantage requires an explicit group_index")
        groups[gid].append(idx)

    n_per_prompt = int(getattr(args, "n_samples_per_prompt", 0) or 0)
    rewards_norm = bool(getattr(args, "rewards_normalization", True))
    std_norm = bool(getattr(args, "grpo_std_normalization", False))

    turn_advantages: list[list[float]] = [[] for _ in samples]
    for indices in groups.values():
        if n_per_prompt > 0 and len(indices) != n_per_prompt:
            raise ValueError(f"expected group size {n_per_prompt}, got {len(indices)}")
        task_ids = {samples[i].metadata.get("task_id") for i in indices}
        if len(task_ids) > 1:
            raise ValueError(f"candidate group mixes multiple task_ids: {sorted(task_ids)}")

        vals = [raw_rewards[i] for i in indices]
        if len(indices) < 2:
            advs = [0.0] * len(indices)
        elif rewards_norm:
            mean_v = math.fsum(vals) / len(vals)
            centered = [v - mean_v for v in vals]
            if std_norm:
                var = math.fsum(c * c for c in centered) / len(centered)
                std = math.sqrt(max(0.0, var))
                advs = [c / (std + 1e-6) for c in centered]
            else:
                advs = centered
        else:
            advs = vals

        for idx, adv in zip(indices, advs, strict=True):
            turns = samples[idx].metadata.get("turns", ())
            turn_advantages[idx] = [float(adv)] * len(turns)
    return raw_rewards, turn_advantages


@dataclass
class FilterResult:
    keep: bool
    reason: str | None = None


def _make_filter_output(keep: bool, reason: str | None = None):
    try:
        from slime.rollout.filter_hub.base_types import DynamicFilterOutput

        return DynamicFilterOutput(keep=keep, reason=reason)
    except ImportError:
        return FilterResult(keep=keep, reason=reason)


def reject_runtime_failures(args: Any, samples: Sequence[Any], **_kwargs: Any):
    """Reject groups hit by context overflow, media decode failure, or empty turns."""
    del args
    for sample in samples:
        reason = sample.metadata.get("failure_reason")
        if reason in _RUNTIME_FAILURE_REASONS:
            return _make_filter_output(False, str(reason).lower())
        if not sample.metadata.get("turns"):
            return _make_filter_output(False, "empty_turns")
    return _make_filter_output(True, None)


def filter_outcome_group(args: Any, samples: Sequence[Any], **kwargs: Any):
    """Reject runtime failures first, then require non-zero outcome reward std."""
    runtime = reject_runtime_failures(args, samples, **kwargs)
    if not runtime.keep:
        return runtime
    rewards = [float(sample.get_reward_value(args)) for sample in samples]
    if len(rewards) < 2:
        return _make_filter_output(False, "insufficient_samples")
    mean_r = math.fsum(rewards) / len(rewards)
    var = math.fsum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    keep = math.sqrt(max(0.0, var)) > 1e-6
    return _make_filter_output(keep, None if keep else f"zero_std_{round(rewards[0], 2)}")


def convert_agent_samples_to_train_data(
    args: Any,
    samples: Sequence[Any],
    *,
    advantage_shaper: Callable[[Any, Sequence[Any]], tuple[list[float], list[list[float]]]] | None = None,
    extra_turn_fields_fn: Callable[[Any, Sequence[Any]], dict[str, list[Any]]] | None = None,
) -> dict[str, Any]:
    """Expand multi-turn carrier Samples into Slime ``BatchBuilder`` training dict."""
    shaper = advantage_shaper or compute_outcome_group_advantages
    raw_rewards, per_sample_turn_advs = shaper(args, samples)

    turns: list[Any] = []
    rewards: list[float] = []
    turn_raw_rewards: list[float] = []
    for carrier, turn_advs, raw_r in zip(samples, per_sample_turn_advs, raw_rewards, strict=True):
        cand_turns = carrier.metadata.get("turns", ())
        if len(cand_turns) != len(turn_advs):
            raise ValueError(
                f"turn count ({len(cand_turns)}) != advantage count ({len(turn_advs)})"
            )
        turns.extend(cand_turns)
        rewards.extend(float(a) for a in turn_advs)
        turn_raw_rewards.extend([float(raw_r)] * len(cand_turns))

    masks: list[list[int]] = []
    for turn in turns:
        ensure_valid_top_p_fields(args, turn)
        mask = turn.loss_mask
        if mask is None:
            mask = [1] * int(turn.response_length)
        if getattr(turn, "remove_sample", False):
            mask = [0] * int(turn.response_length)
        masks.append(list(mask))

    mask_sums: dict[Any, int] = defaultdict(int)
    for turn, mask in zip(turns, masks, strict=True):
        mask_sums[turn.rollout_id] += sum(mask)

    data: dict[str, Any] = {
        "tokens": [turn.tokens for turn in turns],
        "response_lengths": [int(turn.response_length) for turn in turns],
        "rewards": rewards,
        "raw_reward": turn_raw_rewards,
        "truncated": [int(_status_name(turn.status) == "TRUNCATED") for turn in turns],
        "sample_indices": [turn.index for turn in turns],
        "rollout_ids": [turn.rollout_id for turn in turns],
        "loss_masks": masks,
        "rollout_mask_sums": [mask_sums[turn.rollout_id] for turn in turns],
        "multimodal_train_inputs": [turn.multimodal_train_inputs for turn in turns],
        "source_names": [turn.metadata.get("source_name", "unknown") for turn in turns],
    }
    if getattr(args, "use_rollout_logprobs", False):
        data["rollout_log_probs"] = [turn.rollout_log_probs for turn in turns]
    if float(getattr(args, "rollout_top_p", 1.0)) != 1.0:
        data["rollout_top_p_token_ids"] = [turn.rollout_top_p_token_ids for turn in turns]
        data["rollout_top_p_token_offsets"] = [turn.rollout_top_p_token_offsets for turn in turns]
    if getattr(args, "use_rollout_routing_replay", False):
        data["rollout_routed_experts"] = [turn.rollout_routed_experts for turn in turns]

    if extra_turn_fields_fn is not None:
        extra = extra_turn_fields_fn(args, samples)
        for key, values in extra.items():
            if len(values) != len(turns):
                raise ValueError(f"extra field {key!r} length {len(values)} != {len(turns)}")
            data[key] = values
    return data


def log_agent_metrics(
    rollout_id: int,
    args: Any,
    samples: Sequence[Any],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float,
    *,
    extra_metrics_fn: Callable[[Any, Sequence[Any]], dict[str, float]] | None = None,
) -> bool:
    """Record unbiased candidate-level agent rollout metrics."""
    del rollout_id
    if rollout_extra_metrics is None or not samples:
        return False

    n_samples = len(samples)
    turns = [t for s in samples for t in s.metadata.get("turns", ())]
    total_resp_tokens = sum(int(t.response_length) for t in turns)
    rollout_gpus = max(1, int(getattr(args, "rollout_num_gpus", 1) or 1))
    safe_time = max(1e-6, float(rollout_time))

    # Candidate-level outcome reward mean (independent of turn count).
    cand_rewards = [float(s.get_reward_value(args)) for s in samples]
    rollout_extra_metrics.update(
        {
            "agent/candidate_reward_mean": math.fsum(cand_rewards) / n_samples,
            "agent/turns_per_candidate": len(turns) / n_samples,
            "agent/response_tokens": total_resp_tokens,
            "agent/tokens_per_gpu_per_sec": total_resp_tokens / safe_time / rollout_gpus,
            "agent/invalid_actions_per_candidate": (
                sum(int(s.metadata.get("invalid_actions", 0)) for s in samples) / n_samples
            ),
            "agent/no_progress_per_candidate": (
                sum(int(s.metadata.get("no_progress", 0)) for s in samples) / n_samples
            ),
            "agent/correct_fraction": (
                sum(bool(s.metadata.get("reward_components", {}).get("correct")) for s in samples)
                / n_samples
            ),
            "agent/media_failure_fraction": (
                sum(s.metadata.get("failure_reason") == "MEDIA_TOOL_FAILED" for s in samples)
                / n_samples
            ),
            "agent/budget_used_mean": (
                sum(float(s.metadata.get("budget_used", 0.0)) for s in samples) / n_samples
            ),
        }
    )
    if extra_metrics_fn is not None:
        rollout_extra_metrics.update(extra_metrics_fn(args, samples))
    return False


__all__ = [
    "FilterResult",
    "compute_outcome_group_advantages",
    "convert_agent_samples_to_train_data",
    "filter_outcome_group",
    "log_agent_metrics",
    "reject_runtime_failures",
    "score_mcq",
    "score_outcome_sample",
    "score_span_set",
]
