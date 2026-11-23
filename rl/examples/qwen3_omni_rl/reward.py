"""Single-turn Qwen3-Omni rule-based reward functions (MCQ, AV match, Temporal IoU).

STRICT ISOLATION: ZERO ``slime_ext.agent`` or cross-example imports.
Matches Slime ``--custom-rm-path`` async contract for single Sample or batch.
"""

from __future__ import annotations

import re
from typing import Any

_ANSWER_TAG_RE = re.compile(r"<answer\b[^>]*>([\s\S]*?)</answer>", re.IGNORECASE)
_MCQ_RE = re.compile(r"\b([A-Za-z])\b")
_SPAN_RE = re.compile(
    r"[\[(]\s*(\d+(?:\.\d+)?)\s*s?\s*,\s*(\d+(?:\.\d+)?)\s*s?\s*[\])]"
    r"|(\d+(?:\.\d+)?)\s*s?\s*(?:to|\-|–|—)\s*(\d+(?:\.\d+)?)\s*s?",
    re.IGNORECASE,
)
_AV_LABELS = frozenset({"match", "mismatch"})


def _extract_answer_region(text: str) -> str:
    matches = list(_ANSWER_TAG_RE.finditer(str(text or "")))
    if matches:
        return matches[-1].group(1).strip()
    return str(text or "").strip()


def score_mcq_response(response: str, label: Any) -> float:
    region = _extract_answer_region(response)
    cleaned = re.sub(r"<\|.*?\|>", " ", region).strip(" \t\r\n\"'`.,:;!?()[]{}<>")
    if re.fullmatch(r"[A-Za-z]", cleaned):
        pred = cleaned.upper()
    else:
        m = _MCQ_RE.search(cleaned)
        pred = m.group(1).upper() if m else None
    target = str(label).strip().upper()
    return 1.0 if (pred is not None and pred == target) else 0.0


def score_av_match_response(response: str, label: Any) -> float:
    region = _extract_answer_region(response).lower()
    region = re.sub(r"<\|.*?\|>", " ", region)
    words = re.findall(r"[a-z]+", region)
    pred = words[-1] if words and words[-1] in _AV_LABELS else None
    target_words = re.findall(r"[a-z]+", str(label).strip().lower())
    target = target_words[0] if target_words and target_words[0] in _AV_LABELS else None
    if pred is None or target is None:
        return 0.0
    return 1.0 if pred == target else 0.0


def parse_temporal_span(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            s, e = float(value[0]), float(value[1])
            return (s, e) if (s >= 0.0 and e > s) else None
        except (TypeError, ValueError):
            return None
    region = _extract_answer_region(str(value or ""))
    matches = list(_SPAN_RE.finditer(region))
    for m in reversed(matches):
        a = m.group(1) if m.group(1) is not None else m.group(3)
        b = m.group(2) if m.group(2) is not None else m.group(4)
        s, e = float(a), float(b)
        if s >= 0.0 and e > s:
            return s, e
    return None


def score_temporal_iou_response(response: str, label: Any) -> float:
    pred = parse_temporal_span(response)
    gt = parse_temporal_span(label)
    if pred is None or gt is None:
        return 0.0
    inter = max(0.0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return (inter / union) if union > 0.0 else 0.0


def _infer_reward_type(sample: Any, configured: str) -> str:
    if configured != "auto":
        return configured
    meta = getattr(sample, "metadata", None) or {}
    if meta.get("answer_type") in {"mcq", "av_match", "temporal_iou"}:
        return str(meta["answer_type"])
    label = getattr(sample, "label", None)
    if isinstance(label, (list, tuple)) and len(label) == 2:
        return "temporal_iou"
    if isinstance(label, str) and label.strip().lower() in _AV_LABELS:
        return "av_match"
    return "mcq"


def score_one(args: Any, sample: Any) -> float:
    cfg = getattr(args, "qwen3_omni_rl", None) or {}
    reward_type = _infer_reward_type(sample, str(cfg.get("reward_type", "auto")))
    response = getattr(sample, "response", "") or ""
    label = getattr(sample, "label", None)
    if reward_type == "av_match":
        return score_av_match_response(response, label)
    if reward_type == "temporal_iou":
        return score_temporal_iou_response(response, label)
    return score_mcq_response(response, label)


async def score(args: Any, sample_or_samples: Any, **_kwargs: Any) -> float | list[float]:
    """Slime ``--custom-rm-path`` entrypoint supporting single sample or batch."""
    if isinstance(sample_or_samples, list):
        return [score_one(args, s) for s in sample_or_samples]
    return score_one(args, sample_or_samples)


__all__ = [
    "parse_temporal_span",
    "score",
    "score_av_match_response",
    "score_mcq_response",
    "score_one",
    "score_temporal_iou_response",
]
