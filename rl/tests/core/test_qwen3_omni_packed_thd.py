from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
EXT_ROOT = REPO_ROOT
SLIME_ROOT = Path(os.environ.get("SLIME_DIR", str(REPO_ROOT / "third_party/slime")))
sys.path.insert(0, str(SLIME_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from slime_ext.backends.megatron.data import _make_get_batch
from slime_ext.backends.megatron.packed import (
    REAL_CU_SEQLENS_Q_ATTR,
    count_audio_token_runs,
    pack_padded_batch_position_ids,
    packed_attention_mask,
    packed_thd_to_padded_batch,
)


def _load_rope_module():
    rope_path = (
        EXT_ROOT
        / "slime_ext/models/qwen3_omni/rope.py"
    )
    spec = importlib.util.spec_from_file_location("qwen3omni_rope_for_test", rope_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rope_module = _load_rope_module()
get_qwen3_omni_audio_output_lengths = rope_module.get_qwen3_omni_audio_output_lengths
get_rope_index = rope_module.get_rope_index


NUM_GPUS = 0


class FakeMpu:
    @staticmethod
    def get_tensor_model_parallel_world_size():
        return 4

    @staticmethod
    def get_context_parallel_world_size():
        return 1

    @staticmethod
    def get_context_parallel_rank():
        return 0


class FakePackedSeqParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeDataIterator:
    def get_next(self, keys):
        return {
            "tokens": [
                torch.tensor([10, 11, 12], dtype=torch.long),
                torch.tensor([20, 21], dtype=torch.long),
            ],
            "loss_masks": [
                torch.tensor([1], dtype=torch.long),
                torch.tensor([1], dtype=torch.long),
            ],
            "total_lengths": [3, 2],
            "response_lengths": [1, 1],
            "multimodal_train_inputs": None,
        }


def fake_slice_with_cp(tensor, pad_value):
    return tensor


def test_thd_get_batch_records_real_cu_seqlens_before_tail_padding():
    get_batch = _make_get_batch(FakeMpu, torch.nn.functional, FakePackedSeqParams, fake_slice_with_cp)

    batch = get_batch(
        FakeDataIterator(),
        ["tokens", "loss_masks", "total_lengths", "response_lengths"],
        pad_multiplier=2,
    )

    assert batch["tokens"].tolist() == [[10, 11, 12, 20, 21, 0, 0, 0]]
    assert batch["packed_seq_params"].cu_seqlens_q.tolist() == [0, 3, 5, 8]
    assert getattr(batch["packed_seq_params"], REAL_CU_SEQLENS_Q_ATTR).tolist() == [0, 3, 5]
    assert batch["full_loss_masks"].shape == batch["tokens"].shape


def test_packed_helpers_exclude_padding_from_audio_runs_and_repack_positions():
    audio_token_id = 7
    real_cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    packed_input_ids = torch.tensor([[1, audio_token_id, audio_token_id, 2, audio_token_id]], dtype=torch.long)

    unpacked = packed_thd_to_padded_batch(packed_input_ids, real_cu_seqlens, pad_value=audio_token_id)
    mask = packed_attention_mask(real_cu_seqlens, max_seq_len=unpacked.shape[1])

    assert unpacked.tolist() == [[1, 7, 7], [2, 7, 7]]
    assert mask.tolist() == [[True, True, True], [True, True, False]]
    assert count_audio_token_runs(unpacked, audio_token_id, attention_mask=mask).tolist() == [2, 1]

    base = torch.tensor([[0, 1, 2], [0, 1, 1]], dtype=torch.long)
    position_ids = torch.stack([base, base + 10, base + 20], dim=0)
    packed_position_ids = pack_padded_batch_position_ids(position_ids, real_cu_seqlens, total_length=8)

    assert packed_position_ids.shape == (3, 1, 8)
    assert packed_position_ids[:, 0, :5].tolist() == [
        [0, 1, 2, 0, 1],
        [10, 11, 12, 10, 11],
        [20, 21, 22, 20, 21],
    ]
    assert packed_position_ids[:, 0, 5:].tolist() == [[1, 1, 1], [11, 11, 11], [21, 21, 21]]


def test_rope_index_uses_attention_mask_before_audio_run_counting():
    audio_start_token_id = 5
    audio_token_id = 7
    real_cu_seqlens = torch.tensor([0, 4, 7], dtype=torch.int32)
    packed_input_ids = torch.tensor(
        [[10, audio_start_token_id, audio_token_id, audio_token_id, 20, audio_start_token_id, audio_token_id]],
        dtype=torch.long,
    )
    unpacked = packed_thd_to_padded_batch(packed_input_ids, real_cu_seqlens, pad_value=audio_token_id)
    mask = packed_attention_mask(real_cu_seqlens, max_seq_len=unpacked.shape[1])

    position_ids, _ = get_rope_index(
        spatial_merge_size=2,
        image_token_id=101,
        video_token_id=102,
        audio_token_id=audio_token_id,
        vision_start_token_id=103,
        audio_start_token_id=audio_start_token_id,
        position_id_per_seconds=1,
        input_ids=unpacked,
        attention_mask=mask,
        audio_seqlens=torch.tensor([20, 20], dtype=torch.long),
    )

    assert unpacked.tolist() == [[10, 5, 7, 7], [20, 5, 7, 7]]
    assert mask.tolist() == [[True, True, True, True], [True, True, True, False]]
    assert position_ids.shape == (3, 2, 4)
    assert position_ids[:, 1, 3].tolist() == [0, 0, 0]


def test_qwen3_omni_audio_output_lengths_use_thinker_formula():
    lengths = torch.tensor([1, 20, 100, 500], dtype=torch.long)

    assert get_qwen3_omni_audio_output_lengths(lengths).tolist() == [1, 3, 13, 65]


def test_qwen3_omni_audio_output_lengths_differ_from_encoder_instance_method():
    lengths = torch.tensor([500], dtype=torch.long)
    encoder_input_lengths = (lengths - 1) // 2 + 1
    encoder_output_lengths = (encoder_input_lengths - 2) // 2 + 1

    assert get_qwen3_omni_audio_output_lengths(lengths).item() == 65
    assert encoder_output_lengths.item() == 125


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
