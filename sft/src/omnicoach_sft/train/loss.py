"""Project-local loss-scale fixes for weighted Agent turns."""

from __future__ import annotations

import math
from typing import Any


def _explicit_final_response_scale(messages: list[dict[str, Any]]) -> float | None:
    if not messages or messages[-1].get("role") != "assistant":
        return None
    raw_scale = messages[-1].get("loss_scale")
    values = raw_scale if isinstance(raw_scale, list) else [raw_scale]
    explicit = [float(value) for value in values if value is not None]
    if not explicit:
        return None
    if any(not math.isfinite(value) or value <= 0 for value in explicit):
        raise ValueError(f"final assistant has invalid loss_scale: {raw_scale!r}")
    if any(not math.isclose(value, explicit[0], rel_tol=0, abs_tol=1e-12) for value in explicit[1:]):
        raise ValueError(f"final assistant parts have inconsistent loss_scale: {raw_scale!r}")
    return explicit[0]


def _expanded_context_count(context: Any) -> int:
    # Match Swift LossScale.__call__: a list of token ids is one context, while
    # a list of response parts is expanded into independent contexts.
    if not isinstance(context, list) or (context and isinstance(context[0], int)):
        return 1
    return len(context)


def _context_kind(context_type: Any) -> str:
    return str(getattr(context_type, "name", context_type)).rsplit(".", 1)[-1].upper()


class ExactAgentSuffixLossScale:
    """Make an explicitly weighted Agent row supervise only its current target.

    Swift handles fractional response weights and tool rounds, but assigns
    suffixes the base weight and can retain explicitly supervised history.
    Apply the final message's weight only to its RESPONSE and SUFFIX.
    Unweighted rows retain Swift's policy.
    """

    def __init__(self, delegate: Any) -> None:
        if getattr(delegate, "base_strategy", None) != "last_round":
            raise ValueError("ExactAgentSuffixLossScale requires loss_scale=last_round")
        self.delegate = delegate
        self.base_strategy = delegate.base_strategy

    is_binary_loss_scale = False

    def __call__(self, context_list, context_types, messages, **kwargs):
        contexts, weights = self.delegate(context_list, context_types, messages, **kwargs)
        explicit_scale = _explicit_final_response_scale(messages)
        if explicit_scale is None:
            return contexts, weights

        cursor = 0
        last_response: range | None = None
        last_suffix: range | None = None
        for context, context_type in zip(context_list, context_types, strict=True):
            count = _expanded_context_count(context)
            kind = _context_kind(context_type)
            if kind == "RESPONSE":
                last_response = range(cursor, cursor + count)
            elif kind == "SUFFIX":
                last_suffix = range(cursor, cursor + count)
            cursor += count
        if cursor != len(weights):
            raise ValueError(
                "Swift loss-scale delegate changed context cardinality; "
                f"input_expanded={cursor}, output={len(weights)}"
            )
        if last_response is None or last_suffix is None:
            raise ValueError("final assistant response has explicit loss_scale but no RESPONSE/SUFFIX context")
        weights[:] = [0.0] * len(weights)
        for target_range in (last_response, last_suffix):
            for index in target_range:
                weights[index] = explicit_scale
        return contexts, weights


def install_exact_agent_suffix_loss_scale(template: Any) -> None:
    """Install the one-target suffix wrapper for ``last_round`` templates only.

    Multi-target segment datasets use native ms-swift message ``loss`` /
    ``loss_scale`` with the ``default`` base strategy and must keep the native
    loss-scale implementation.
    """

    delegate = template.loss_scale
    if isinstance(delegate, ExactAgentSuffixLossScale) or getattr(delegate, "base_strategy", None) != "last_round":
        return
    template._loss_scale_cache[template._loss_scale] = ExactAgentSuffixLossScale(delegate)
