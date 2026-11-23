from array import array
from types import SimpleNamespace


def test_slime_weight_update_excludes_encoder_group():
    from slime.ray.rollout import RolloutManager

    def group(worker_type, engines, *, gpu_count, gpu_offset, num_new):
        return SimpleNamespace(
            worker_type=worker_type,
            engines=engines,
            num_gpus_per_engine=gpu_count,
            gpu_offset=gpu_offset,
            num_new_engines=num_new,
            parallel_config=lambda: {"worker_type": worker_type},
        )

    encoder = group("encoder", ["encoder-0", "encoder-1"], gpu_count=1, gpu_offset=0, num_new=2)
    regular = group("regular", ["regular-0", "regular-1"], gpu_count=2, gpu_offset=2, num_new=1)
    server = SimpleNamespace(update_weights=True, server_groups=[encoder, regular])
    manager_class = RolloutManager.__ray_metadata__.modified_class
    manager = manager_class.__new__(manager_class)
    manager.servers = {"actor": server}
    manager.rollout_engine_lock = object()

    engines, lock, num_new, gpu_counts, gpu_offsets, parallel_configs = (
        manager.get_updatable_engines_and_lock()
    )

    assert engines == ["regular-0", "regular-1"]
    assert lock is manager.rollout_engine_lock
    assert num_new == 1
    assert gpu_counts == [2, 2]
    assert gpu_offsets == [2, 4]
    assert parallel_configs == [
        {"worker_type": "regular"},
        {"worker_type": "regular"},
    ]


def test_ep_language_engines_get_node_local_unique_bootstrap_ports(monkeypatch):
    import slime.backends.sglang_utils.engine_group as engine_group_module

    class RemoteMethod:
        def __init__(self, host):
            self.host = host

        def remote(self, *, start_port=None, consecutive=None):
            del consecutive
            return self.host, start_port or 20000

    class Engine:
        def __init__(self, host):
            self._get_current_node_ip_and_free_port = RemoteMethod(host)

    monkeypatch.setattr(engine_group_module.ray, "get", lambda value: value)
    args = SimpleNamespace(
        num_gpus_per_node=8,
        rollout_num_gpus_per_engine=2,
        sglang_dp_size=1,
    )
    engines = [
        (2, Engine("node-a")),
        (3, Engine("node-a")),
        (4, Engine("node-a")),
        (5, Engine("node-b")),
    ]

    allocated, cursors = engine_group_module._allocate_rollout_engine_addr_and_ports_normal(
        args=args,
        rollout_engines=engines,
        worker_type="regular",
        allocate_encoder_bootstrap=True,
        num_gpus_per_engine=2,
        rank_offset=2,
    )

    assert [allocated[rank]["host"] for rank in range(2, 6)] == [
        "node-a",
        "node-a",
        "node-a",
        "node-b",
    ]
    assert [allocated[rank]["encoder_bootstrap_port"] for rank in (2, 3, 4)] == [
        15002,
        15036,
        15070,
    ]
    assert allocated[5]["encoder_bootstrap_port"] == 15002
    assert cursors == {"node-a": 15102, "node-b": 15034}


def test_sglang_engine_propagates_encoder_bootstrap_port():
    from slime.backends.sglang_utils.sglang_engine import _compute_server_args

    args = SimpleNamespace(
        hf_checkpoint="/model",
        seed=1234,
        offload_rollout=False,
        num_gpus_per_node=8,
        rollout_num_gpus_per_engine=2,
        sglang_pp_size=1,
        sglang_dp_size=1,
        sglang_ep_size=1,
        use_rollout_routing_replay=False,
        fp16=False,
    )
    server_args, _ = _compute_server_args(
        args,
        rank=2,
        dist_init_addr="node-a:15003",
        nccl_port=15001,
        host="node-a",
        port=15000,
        worker_type="regular",
        encoder_bootstrap_port=15002,
        base_gpu_id=0,
        sglang_overrides={"language_only": True},
        num_gpus_per_engine=2,
    )

    assert server_args["language_only"] is True
    assert server_args["encoder_bootstrap_port"] == 15002


def test_sglang_epd_preserves_pretokenized_array():
    from sglang.srt.multimodal.processors.base_processor import (
        BaseMultimodalProcessor,
    )

    class RejectingTokenizer:
        def encode(self, prompt):
            raise AssertionError(f"pretokenized input was re-tokenized: {prompt!r}")

    class ProbeProcessor(BaseMultimodalProcessor):
        async def process_mm_data_async(self, *args, **kwargs):
            raise NotImplementedError

    processor = ProbeProcessor.__new__(ProbeProcessor)
    processor._tokenizer = RejectingTokenizer()
    processor.hf_config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=1)
    )
    processor.IM_TOKEN_ID = None
    processor.VIDEO_TOKEN_ID = None
    processor.audio_token_id = None

    input_ids, offsets, modalities = processor.build_input_ids(
        array("q", [101, 102, 103])
    )

    assert input_ids == [101, 102, 103]
    assert offsets == []
    assert modalities == []
