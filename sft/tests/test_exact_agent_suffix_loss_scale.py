from __future__ import annotations

import pickle
from enum import Enum, auto

import pytest

from omnicoach_sft.train.loss import ExactAgentSuffixLossScale, install_exact_agent_suffix_loss_scale


class ContextType(Enum):
    RESPONSE = auto()
    SUFFIX = auto()
    OTHER = auto()


class FakeLastRoundLossScale:
    base_strategy = "last_round"

    def __call__(self, context_list, context_types, messages, **kwargs):
        contexts = []
        weights = []
        for context, context_type in zip(context_list, context_types, strict=True):
            parts = context if isinstance(context, list) and not (context and isinstance(context[0], int)) else [context]
            contexts.extend(parts)
            if context_type == ContextType.SUFFIX:
                weights.extend([1.0] * len(parts))
            elif context_type == ContextType.RESPONSE:
                raw = messages[-1].get("loss_scale")
                values = raw if isinstance(raw, list) else [raw]
                weights.extend([float(value or 1.0) for value in values])
            else:
                weights.extend([0.0] * len(parts))
        return contexts, weights


def test_fractional_answer_scales_only_final_suffix():
    wrapper = ExactAgentSuffixLossScale(FakeLastRoundLossScale())
    contexts, weights = wrapper(
        ["history", "<|im_end|>", "answer", "<|im_end|>"],
        [ContextType.OTHER, ContextType.OTHER, ContextType.RESPONSE, ContextType.SUFFIX],
        [{"role": "assistant", "content": "<answer>A</answer>", "loss_scale": 0.5}],
    )
    assert contexts[-2:] == ["answer", "<|im_end|>"]
    assert weights == [0.0, 0.0, 0.5, 0.5]


def test_native_tool_history_response_is_masked_even_when_swift_marks_it_last_round():
    wrapper = ExactAgentSuffixLossScale(FakeLastRoundLossScale())
    _, weights = wrapper(
        ["historical assistant tool call", "historical suffix", "final answer", "final suffix"],
        [ContextType.RESPONSE, ContextType.SUFFIX, ContextType.RESPONSE, ContextType.SUFFIX],
        [{"role": "assistant", "content": "final answer", "loss_scale": 0.5}],
    )
    assert weights == [0.0, 0.0, 0.5, 0.5]


def test_explicit_unit_weight_agent_row_still_masks_history():
    wrapper = ExactAgentSuffixLossScale(FakeLastRoundLossScale())
    _, weights = wrapper(
        ["historical response", "historical suffix", "final response", "final suffix"],
        [ContextType.RESPONSE, ContextType.SUFFIX, ContextType.RESPONSE, ContextType.SUFFIX],
        [{"role": "assistant", "content": "final answer", "loss_scale": 1.0}],
    )
    assert weights == [0.0, 0.0, 1.0, 1.0]


def test_merged_tool_parts_and_suffix_share_exact_weight():
    wrapper = ExactAgentSuffixLossScale(FakeLastRoundLossScale())
    _, weights = wrapper(
        [["observation-think", "tool-call"], "<|im_end|>"],
        [ContextType.RESPONSE, ContextType.SUFFIX],
        [
            {
                "role": "assistant",
                "content": ["observation-think", "tool-call"],
                "loss_scale": [1 / 3, 1 / 3],
            }
        ],
    )
    assert weights == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_pinned_swift_string_suffix_context_is_recognized():
    wrapper = ExactAgentSuffixLossScale(FakeLastRoundLossScale())
    _, weights = wrapper(
        ["answer", "<|im_end|>"],
        [ContextType.RESPONSE, "suffix"],
        [{"role": "assistant", "content": "answer", "loss_scale": 0.5}],
    )
    assert weights == [0.5, 0.5]


def test_base_row_without_explicit_weight_is_unchanged():
    wrapper = ExactAgentSuffixLossScale(FakeLastRoundLossScale())
    _, weights = wrapper(
        ["answer", "<|im_end|>"],
        [ContextType.RESPONSE, ContextType.SUFFIX],
        [{"role": "assistant", "content": "answer", "loss_scale": None}],
    )
    assert weights == [1.0, 1.0]


def test_install_replaces_cached_last_round_strategy_once():
    class FakeTemplate:
        _loss_scale = "last_round"

        def __init__(self):
            self._loss_scale_cache = {"last_round": FakeLastRoundLossScale()}

        @property
        def loss_scale(self):
            return self._loss_scale_cache[self._loss_scale]

    template = FakeTemplate()
    install_exact_agent_suffix_loss_scale(template)
    first = template.loss_scale
    install_exact_agent_suffix_loss_scale(template)
    assert isinstance(first, ExactAgentSuffixLossScale)
    assert template.loss_scale is first


def test_install_keeps_native_default_strategy_for_segment_rows():
    class FakeDefaultLossScale:
        base_strategy = "default"

    class FakeTemplate:
        _loss_scale = "default"

        def __init__(self):
            self._loss_scale_cache = {"default": FakeDefaultLossScale()}

        @property
        def loss_scale(self):
            return self._loss_scale_cache[self._loss_scale]

    template = FakeTemplate()
    native = template.loss_scale

    install_exact_agent_suffix_loss_scale(template)

    assert template.loss_scale is native


def test_wrapper_pickle_round_trip_restores_delegate_without_recursion():
    wrapper = ExactAgentSuffixLossScale(FakeLastRoundLossScale())

    restored = pickle.loads(pickle.dumps(wrapper))

    assert isinstance(restored, ExactAgentSuffixLossScale)
    assert restored.base_strategy == "last_round"
    _, weights = restored(
        ["answer", "<|im_end|>"],
        [ContextType.RESPONSE, ContextType.SUFFIX],
        [{"role": "assistant", "content": "answer", "loss_scale": 0.5}],
    )
    assert weights == [0.5, 0.5]
