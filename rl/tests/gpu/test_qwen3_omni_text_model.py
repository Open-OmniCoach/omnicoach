from copy import deepcopy

import torch
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import TransformerLayer

from slime_ext.models.qwen3_omni.text import (
    Qwen3OmniMultimodalRotaryEmbedding,
    Qwen3OmniTransformerLayer,
    _with_deepstack_layers,
)


def test_qwen3_mrope_interleaves_temporal_height_and_width():
    rope = Qwen3OmniMultimodalRotaryEmbedding.__new__(
        Qwen3OmniMultimodalRotaryEmbedding
    )
    torch.nn.Module.__init__(rope)
    rope.inv_freq = torch.ones(4)
    rope.rotary_interleaved = False
    rope.seq_len_interpolation_factor = None
    rope.cp_group = None
    position_ids = torch.tensor([[[10]], [[20]], [[30]]])

    output = rope(position_ids, [2, 1, 1])

    assert output[:, 0, 0, :4].tolist() == [[10.0, 20.0, 30.0, 10.0]]
    assert output[:, 0, 0, 4:].tolist() == [[10.0, 20.0, 30.0, 10.0]]


def test_deepstack_layer_injects_after_decoder_layer(monkeypatch):
    def decoder_forward(self, hidden_states, **kwargs):
        return hidden_states + 1, None

    monkeypatch.setattr(TransformerLayer, "forward", decoder_forward)
    layer = Qwen3OmniTransformerLayer.__new__(Qwen3OmniTransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.layer_number = 1
    hidden_states = torch.zeros(3, 1, 2)
    visual_mask = torch.tensor([[False, True, True]])
    deepstack = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])

    output, returned_deepstack = layer.forward(
        hidden_states,
        context=deepstack,
        context_mask=visual_mask,
    )

    assert output[:, 0].tolist() == [[1.0, 1.0], [11.0, 21.0], [31.0, 41.0]]
    assert returned_deepstack is deepstack


def test_deepstack_spec_is_copied_before_replacing_layers():
    original = TransformerBlockSubmodules(
        layer_specs=[ModuleSpec(module=TransformerLayer)]
    )
    snapshot = deepcopy(original)

    updated = _with_deepstack_layers(original)

    assert updated.layer_specs[0].module is Qwen3OmniTransformerLayer
    assert original.layer_specs[0].module is snapshot.layer_specs[0].module
