from types import SimpleNamespace

import torch

from slime_ext.models.qwen3_omni.weights.hf_to_megatron import qwen3_omni_hf_tensor
from slime_ext.models.qwen3_omni.weights.megatron_to_hf import convert_qwen3omni_to_hf


class Reader:
    def __init__(self, tensors):
        self.tensors = tensors

    def __contains__(self, name):
        return name in self.tensors

    def get_tensor(self, name):
        return self.tensors[name]


_ARGS = SimpleNamespace(
    kv_channels=2,
    hidden_size=8,
    num_attention_heads=4,
    num_query_groups=2,
    vocab_size=16,
)
_CONFIG = SimpleNamespace(
    thinker_config=SimpleNamespace(
        text_config=SimpleNamespace(
            hidden_size=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            tie_word_embeddings=False,
        )
    )
)


def _round_trip(name, parameter):
    hf_tensors = dict(convert_qwen3omni_to_hf(_ARGS, name, parameter))
    return qwen3_omni_hf_tensor(name, Reader(hf_tensors), _CONFIG)


def test_language_qkv_round_trip():
    parameter = torch.arange(16 * 8).reshape(16, 8)
    name = "module.module.thinker.language_model.decoder.layers.0.self_attention.linear_qkv.weight"

    assert torch.equal(_round_trip(name, parameter), parameter)


def test_global_and_local_expert_round_trip():
    parameter = torch.arange(12 * 8).reshape(12, 8)
    names = (
        "module.module.thinker.language_model.decoder.layers.0.mlp.experts.linear_fc1.weight9",
        "module.module.thinker.language_model.decoder.layers.0.mlp.experts.local_experts.9.linear_fc1.weight",
    )

    for name in names:
        assert torch.equal(_round_trip(name, parameter), parameter)


def test_multimodal_towers_keep_their_hf_layout():
    visual = torch.randn(2, 3)
    audio = torch.randn(3, 4)
    reader = Reader(
        {
            "thinker.visual.patch_embed.weight": visual,
            "thinker.audio_tower.conv.weight": audio,
        }
    )

    assert qwen3_omni_hf_tensor("thinker.visual.patch_embed.weight", reader, _CONFIG) is visual
    assert qwen3_omni_hf_tensor("thinker.audio_model.conv.weight", reader, _CONFIG) is audio
