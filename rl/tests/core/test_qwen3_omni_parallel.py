import torch

from slime_ext.models.qwen3_omni.parallel import (
    mark_replicated_parameters,
    shard_deepstack_for_model_parallel,
    split_data_for_context_parallel,
)


def test_replicated_tower_parameters_are_explicit():
    tower = torch.nn.Linear(2, 3)

    mark_replicated_parameters(tower)

    for parameter in tower.parameters():
        assert parameter.tensor_model_parallel is False
        assert parameter.partition_dim == -1
        assert parameter.partition_stride == 1
        assert parameter.average_gradients_across_tp_domain is True


def test_context_parallel_split_is_zigzag():
    values = torch.arange(8).view(8, 1, 1)

    assert split_data_for_context_parallel(values, cp_size=2, cp_rank=0).flatten().tolist() == [0, 1, 6, 7]
    assert split_data_for_context_parallel(values, cp_size=2, cp_rank=1).flatten().tolist() == [2, 3, 4, 5]


def test_deepstack_features_follow_sequence_parallel_shard():
    mask = torch.tensor([[True, False, True, False, False, True, True, False]])
    features = [torch.tensor([[10], [20], [30], [40]])]

    rank0_mask, rank0_features = shard_deepstack_for_model_parallel(
        mask, features, tp_size=2, tp_rank=0, cp_size=1, cp_rank=0, sequence_parallel=True
    )
    rank1_mask, rank1_features = shard_deepstack_for_model_parallel(
        mask, features, tp_size=2, tp_rank=1, cp_size=1, cp_rank=0, sequence_parallel=True
    )

    assert rank0_mask.tolist() == [[True, False, True, False]]
    assert rank1_mask.tolist() == [[False, True, True, False]]
    assert rank0_features[0].flatten().tolist() == [10, 20]
    assert rank1_features[0].flatten().tolist() == [30, 40]
