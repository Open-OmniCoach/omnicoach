from __future__ import annotations

import json
from dataclasses import replace

import pytest
from datasets import Dataset

from omnicoach_sft.train.lengths import (
    _estimate_lengths,
    _exact_boundary_lengths,
    load_pack_length_profile,
    prepare_source_lengths,
)

AGENT_SHARED_CONTRACT = {
    "schema_version": "agent_turn_shared_media_v1",
    "table_layout": "agent_turn_shared_media_v1",
}


class FakeTemplate:
    model = object()

    def __init__(self, *, exact_length: int = 32):
        self.exact_length = exact_length

    def encode(self, row, return_length=False):
        assert isinstance(row, dict)
        content = "".join(message["content"] for message in row["messages"])
        if "<image>" in content or "<audio>" in content:
            return {"lengths": self.exact_length}
        return {"lengths": 32}


class ToolAwareFakeTemplate(FakeTemplate):
    def encode(self, row, return_length=False):
        return {"lengths": 532 if row.get("tools") else 32}


class DistinctLengthTemplate(FakeTemplate):
    def __init__(self, *, fail=False):
        super().__init__()
        self.calls = 0
        self.fail = fail

    def encode(self, row, return_length=False):
        self.calls += 1
        if self.fail:
            raise ValueError("intentional encoding failure")
        index = int(row["messages"][-1]["content"].split("-")[-1])
        return {"lengths": 32 + index * 256}


def make_distributed_dataset():
    rows = []
    for index in range(7):
        row = make_agent_dataset()[0]
        row["messages"][-1]["content"] = f"answer-{index}"
        row["row_id"] = index
        rows.append(row)
    return Dataset.from_list(rows)


def _distributed_length_worker(rank, root):
    import os
    from datetime import timedelta
    from pathlib import Path

    dist = pytest.importorskip("torch.distributed")

    from omnicoach_sft.train import lengths as estimator

    root = Path(root)
    os.environ["LOCAL_RANK"] = str(rank % 2)
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(),
                            rank=rank, world_size=4, timeout=timedelta(seconds=30))
    try:
        dataset = make_distributed_dataset()
        results = {}
        original_estimate = estimator._estimate_lengths
        original_load = estimator.load_cached_lengths
        original_save = estimator.save_cached_lengths
        writes = []

        def writer_only_save(*args, **kwargs):
            assert rank == 0, "only global rank 0 may publish"
            writes.append(str(args[0]))
            return original_save(*args, **kwargs)

        def writer_only_load(*args, **kwargs):
            assert rank == 0, "peers must receive lengths without reading storage"
            return original_load(*args, **kwargs)

        estimator.save_cached_lengths = writer_only_save
        estimator.load_cached_lengths = writer_only_load

        def reject_recompute(*args, **kwargs):
            raise AssertionError("a fully warm cache must not encode any rows")

        for phase in ("cold", "warm", "mixed", "shared"):
            node = rank // 2
            cache = root / ("shared-cache" if phase == "shared" else f"node-{node}")
            if phase == "mixed" and node == 1:
                cache = root / "replacement-node"
            profile = replace(load_pack_length_profile("config/pack_length/qwen3_omni_contract_v1.yaml"),
                              cache_dir=cache)
            estimator._estimate_lengths = reject_recompute if phase in {"warm", "mixed"} else original_estimate
            template = DistinctLengthTemplate()
            original_model = template.model
            prepared, overflow = prepare_source_lengths(
                dataset, source=str(root / "agent"), template=template, profile=profile,
                model_identity="fake-model", packing_length=1024, num_proc=16,
            )
            assert template.model is original_model
            results[phase] = {"rows": list(prepared["row_id"]), "lengths": list(prepared["lengths"]),
                              "overflow": overflow, "calls": template.calls}

        results["writes"] = writes
        estimator._estimate_lengths = original_estimate

        def fail_io(*args, **kwargs):
            raise OSError("injected cache I/O failure")

        for failure in ("read", "write"):
            estimator.load_cached_lengths = fail_io if failure == "read" else writer_only_load
            estimator.save_cached_lengths = fail_io if failure == "write" else writer_only_save
            failure_profile = replace(profile, cache_dir=root / f"failure-{failure}")
            try:
                prepare_source_lengths(
                    dataset, source=str(root / "agent"), template=DistinctLengthTemplate(),
                    profile=failure_profile, model_identity="fake-model", packing_length=1024, num_proc=1,
                )
            except RuntimeError as exc:
                assert "rank 0: OSError: injected cache I/O failure" in str(exc)
                results[f"{failure}_failure_propagated"] = True
            else:
                raise AssertionError("all ranks must receive the storage failure")
        estimator.load_cached_lengths = writer_only_load
        estimator.save_cached_lengths = writer_only_save
        try:
            prepare_source_lengths(
                dataset, source=str(root / "agent"), template=DistinctLengthTemplate(),
                profile=profile, model_identity=f"different-model-{rank}", packing_length=1024, num_proc=1,
            )
        except ValueError as exc:
            assert "differ across ranks" in str(exc)
            results["identity_mismatch_rejected"] = True
        else:
            raise AssertionError("different cache identities must not share lengths")
        # Fewer rows than ranks must still participate in all collectives.
        results["sparse"] = estimator._estimate_lengths_distributed(
            dataset.select([0]), DistinctLengthTemplate(), profile, 16)
        try:
            estimator._estimate_lengths_distributed(
                dataset, DistinctLengthTemplate(fail=rank == 1), profile, 16)
        except RuntimeError as exc:
            assert "rank 1: ValueError: intentional encoding failure" in str(exc)
            results["error_propagated"] = True
        else:
            raise AssertionError("all ranks must receive the encoding failure")
        (root / f"result-{rank}.json").write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


def test_distributed_lengths_preserve_order_filtering_and_cache_reuse(tmp_path, monkeypatch):
    import multiprocessing as mp

    dist = pytest.importorskip("torch.distributed")

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU Gloo process groups are unavailable")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    source = tmp_path / "agent"
    write_agent_contract(source)
    profile = replace(load_pack_length_profile("config/pack_length/qwen3_omni_contract_v1.yaml"),
                      cache_dir=tmp_path / "reference")
    prepared, overflow = prepare_source_lengths(
        make_distributed_dataset(), source=str(source), template=DistinctLengthTemplate(),
        profile=profile, model_identity="fake-model", packing_length=1024, num_proc=1,
    )
    expected = {"rows": list(prepared["row_id"]), "lengths": list(prepared["lengths"]), "overflow": overflow}
    assert 0 < len(prepared) < 7
    # Spawn only our own bounded CPU workers; no signal/process cleanup calls.
    context = mp.get_context("spawn")
    workers = [context.Process(target=_distributed_length_worker, args=(rank, str(tmp_path))) for rank in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=60)
    assert [worker.exitcode for worker in workers] == [0] * 4
    results = [json.loads((tmp_path / f"result-{rank}.json").read_text()) for rank in range(4)]
    for phase in ("cold", "warm", "mixed", "shared"):
        for result in results:
            assert {key: result[phase][key] for key in expected} == expected
        assert sum(result[phase]["calls"] for result in results) == (0 if phase in {"warm", "mixed"} else 7)
    assert all(result["sparse"] == [256] and result["error_propagated"] for result in results)
    assert len(list((tmp_path / "shared-cache").glob("*.npz"))) == 1
    assert [len(result["writes"]) for result in results] == [2, 0, 0, 0]
    assert all(result["read_failure_propagated"] and result["write_failure_propagated"]
               and result["identity_mismatch_rejected"] for result in results)


def test_placeholder_mismatch_fails():
    dataset = make_agent_dataset().remove_columns("omni_num_images").add_column("omni_num_images", [2])
    profile = load_pack_length_profile("config/pack_length/qwen3_omni_contract_v1.yaml")
    with pytest.raises(ValueError, match="image placeholder/media count mismatch"):
        _estimate_lengths(dataset, FakeTemplate(), profile, 1)


def test_estimator_includes_agent_tool_definition_tokens():
    dataset = make_agent_dataset().add_column("tools", ['[{"type":"function"}]'])
    profile = load_pack_length_profile("config/pack_length/qwen3_omni_contract_v1.yaml")

    assert _estimate_lengths(dataset, ToolAwareFakeTemplate(), profile, 1) == [768]


def test_exact_boundary_overflow_is_filtered_without_flattening_media(tmp_path, monkeypatch):
    source = tmp_path / "agent"
    write_agent_contract(source, image_token_count_basis=None, image_token_count_column=None)
    profile = replace(
        load_pack_length_profile("config/pack_length/qwen3_omni_contract_v1.yaml"),
        cache_dir=tmp_path / "cache",
        exact_encode_threshold=1,
    )

    def fail_flatten(*args, **kwargs):
        raise AssertionError("adding packing lengths must not flatten an indexed media dataset")

    monkeypatch.setattr(Dataset, "flatten_indices", fail_flatten)
    filtered, overflow = prepare_source_lengths(
        make_agent_dataset(),
        source=str(source),
        template=FakeTemplate(exact_length=70000),
        profile=profile,
        model_identity="fake-model",
        packing_length=65536,
        num_proc=1,
    )
    assert len(filtered) == 0
    assert overflow[0]["exact_length"] == 70000


def test_contract_upper_bound_skips_media_exact_encode():
    dataset = make_distributed_dataset().select(range(2))
    profile = replace(
        load_pack_length_profile("config/pack_length/qwen3_omni_contract_v1.yaml"),
        exact_encode_threshold=1,
    )
    estimated = [65536, 65792]

    lengths, keep_indices, overflow = _exact_boundary_lengths(
        dataset,
        template=None,
        estimated=estimated,
        source_name="fixture",
        profile=profile,
        packing_length=65536,
        trust_contract_lengths=True,
    )

    assert lengths.tolist() == [65536]
    assert keep_indices.tolist() == [0]
    assert overflow[0]["source_index"] == 1
    assert overflow[0]["reason"] == "contract_length_upper_bound"


def make_agent_dataset(*, source_id: str = "agent:test") -> Dataset:
    return Dataset.from_dict(
        {
            "messages": [
                [
                    {"role": "user", "content": "<audio><0.0 seconds><image> question"},
                    {"role": "assistant", "content": "<answer>A</answer>", "loss_scale": 0.5},
                ]
            ],
            "images": [["frame.webp"]],
            "audios": [["audio.m4a"]],
            "omni_source_id": [source_id],
            "omni_num_images": [1],
            "omni_image_num_tokens": [47],
            "omni_audio_num_samples": [[16000]],
            "omni_last_frame_timestamp_ms": [0],
        }
    )


def write_agent_contract(
    path, *, source_id: str = "agent:test", protocol: str = "native-tool-v2", **stats_overrides
) -> None:
    stats = {
        "image_max_pixels": 147456,
        "image_token_count_basis": "qwen3_omni_v3_patch16_merge2_plus_vision_wrappers",
        "image_token_count_column": "omni_image_num_tokens",
        "audio_sample_rate": 16000,
        "audio_sample_count_basis": "resampled_16khz",
        **stats_overrides,
    }
    path.mkdir()
    (path / "dataset_contract.json").write_text(
        json.dumps(
            {
                **AGENT_SHARED_CONTRACT,
                "purpose": "agentic_turn_sft",
                "protocol": protocol,
                "source_id": source_id,
                "pack_length_stats": {key: value for key, value in stats.items() if value is not None},
            }
        ),
        encoding="utf-8",
    )


def test_agent_contract_stats_use_fast_length_path(tmp_path):
    source = tmp_path / "agent"
    write_agent_contract(source)
    profile = replace(
        load_pack_length_profile("config/pack_length/qwen3_omni_contract_v1.yaml"),
        cache_dir=tmp_path / "cache",
    )
    prepared, overflow = prepare_source_lengths(
        make_agent_dataset(),
        source=str(source),
        template=FakeTemplate(),
        profile=profile,
        model_identity="fake-model",
        packing_length=65536,
        num_proc=1,
    )
    assert len(prepared) == 1
    # Exact image tokens (47), not the legacy fixed image allowance (152).
    assert prepared["lengths"] == [256]
    assert overflow == []


def test_agent_protocol_is_descriptive_not_an_estimator_allowlist(tmp_path):
    source = tmp_path / "agent"
    write_agent_contract(source, protocol="new-valid-agent-protocol-v1")
    profile = replace(
        load_pack_length_profile("config/pack_length/qwen3_omni_contract_v1.yaml"),
        cache_dir=tmp_path / "cache",
    )
    prepared, overflow = prepare_source_lengths(
        make_agent_dataset(),
        source=str(source),
        template=FakeTemplate(),
        profile=profile,
        model_identity="fake-model",
        packing_length=65536,
        num_proc=1,
    )
    assert prepared["lengths"] == [256]
    assert overflow == []


def test_agent_contract_requires_media_stats(tmp_path):
    source = tmp_path / "agent"
    source.mkdir()
    (source / "dataset_contract.json").write_text(
        json.dumps(
            {
                **AGENT_SHARED_CONTRACT,
                "purpose": "agentic_turn_sft",
                "protocol": "new-valid-agent-protocol-v1",
                "source_id": "agent:test",
            }
        ),
        encoding="utf-8",
    )
    profile = replace(
        load_pack_length_profile("config/pack_length/qwen3_omni_contract_v1.yaml"),
        cache_dir=tmp_path / "cache",
    )
    with pytest.raises(ValueError, match="missing pack_length_stats"):
        prepare_source_lengths(
            make_agent_dataset(),
            source=str(source),
            template=FakeTemplate(),
            profile=profile,
            model_identity="fake-model",
            packing_length=65536,
            num_proc=1,
        )


def test_seconds_agent_contract_stats_use_fast_length_path(tmp_path):
    source = tmp_path / "agent"
    write_agent_contract(source, protocol="native-tool-v3-seconds")
    profile = replace(
        load_pack_length_profile("config/pack_length/qwen3_omni_contract_v1.yaml"),
        cache_dir=tmp_path / "cache",
    )
    prepared, overflow = prepare_source_lengths(
        make_agent_dataset(),
        source=str(source),
        template=FakeTemplate(),
        profile=profile,
        model_identity="fake-model",
        packing_length=65536,
        num_proc=1,
    )
    assert prepared["lengths"] == [256]
    assert overflow == []


def test_seconds_think_only_agent_contract_stats_use_fast_length_path(tmp_path):
    source = tmp_path / "agent"
    write_agent_contract(source, protocol="native-tool-v3-seconds-think-only")
    profile = replace(
        load_pack_length_profile("config/pack_length/qwen3_omni_contract_v1.yaml"),
        cache_dir=tmp_path / "cache",
    )
    prepared, overflow = prepare_source_lengths(
        make_agent_dataset(),
        source=str(source),
        template=FakeTemplate(),
        profile=profile,
        model_identity="fake-model",
        packing_length=65536,
        num_proc=1,
    )
    assert prepared["lengths"] == [256]
    assert overflow == []


def test_agent_contract_stats_require_all_parquet_columns(tmp_path):
    source = tmp_path / "agent"
    write_agent_contract(source)
    profile = replace(
        load_pack_length_profile("config/pack_length/qwen3_omni_contract_v1.yaml"),
        cache_dir=tmp_path / "cache",
    )
    dataset = make_agent_dataset().remove_columns("omni_last_frame_timestamp_ms")
    with pytest.raises(ValueError, match="omni_last_frame_timestamp_ms"):
        prepare_source_lengths(
            dataset,
            source=str(source),
            template=FakeTemplate(),
            profile=profile,
            model_identity="fake-model",
            packing_length=65536,
            num_proc=1,
        )


def test_agent_contract_rejects_non_resampled_audio_counts(tmp_path):
    source = tmp_path / "agent"
    write_agent_contract(source, audio_sample_count_basis="source_native")
    profile = replace(
        load_pack_length_profile("config/pack_length/qwen3_omni_contract_v1.yaml"),
        cache_dir=tmp_path / "cache",
    )
    with pytest.raises(ValueError, match="resampled_16khz"):
        prepare_source_lengths(
            make_agent_dataset(),
            source=str(source),
            template=FakeTemplate(),
            profile=profile,
            model_identity="fake-model",
            packing_length=65536,
            num_proc=1,
        )


def test_cache_reads_legacy_pair_and_atomic_entry(tmp_path):
    import numpy as np

    from omnicoach_sft.train.lengths import load_cached_lengths, save_cached_lengths

    npz_path, json_path = tmp_path / "cache.npz", tmp_path / "cache.json"
    metadata = {"cache_key": "unchanged-key", "kept_rows": 1}
    np.savez_compressed(npz_path, lengths=[32], keep_indices=[0])
    json_path.write_text(json.dumps(metadata))
    assert load_cached_lengths(npz_path, json_path, metadata).lengths.tolist() == [32]
    save_cached_lengths(npz_path, json_path, lengths=np.array([64]),
                        keep_indices=np.array([0]), metadata=metadata)
    # An obsolete/malformed sidecar cannot corrupt a self-contained entry.
    json_path.write_text("invalid old sidecar")
    assert load_cached_lengths(npz_path, json_path, metadata).lengths.tolist() == [64]
    json_path.unlink()
    assert load_cached_lengths(npz_path, json_path, metadata).lengths.tolist() == [64]
    assert load_cached_lengths(npz_path, json_path, {"cache_key": "other"}) is None


def test_concurrent_cache_publishers_never_mix_arrays_and_metadata(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    import numpy as np

    from omnicoach_sft.train.lengths import load_cached_lengths, save_cached_lengths

    npz_path, json_path = tmp_path / "cache.npz", tmp_path / "cache.json"
    ready = Barrier(4)

    def publish(writer):
        ready.wait(timeout=10)
        for iteration in range(10):
            value = writer * 10 + iteration
            save_cached_lengths(npz_path, json_path, lengths=np.array([value]),
                                keep_indices=np.array([0]),
                                metadata={"cache_key": "key", "kept_rows": 1, "value": value})
            cached = load_cached_lengths(npz_path, json_path, {"cache_key": "key"})
            assert cached is not None
            assert cached.lengths.tolist() == [cached.metadata["value"]]

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(publish, range(4)))
    assert sorted(path.name for path in tmp_path.iterdir()) == ["cache.npz"]


def test_failed_publication_preserves_previous_cache(tmp_path, monkeypatch):
    import numpy as np

    from omnicoach_sft.train import lengths as cache

    npz_path, json_path = tmp_path / "cache.npz", tmp_path / "cache.json"
    metadata = {"cache_key": "key", "kept_rows": 1}
    cache.save_cached_lengths(npz_path, json_path, lengths=np.array([32]),
                              keep_indices=np.array([0]), metadata=metadata)

    def fail_replace(*args):
        raise OSError("injected rename failure")

    monkeypatch.setattr(cache.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected rename failure"):
        cache.save_cached_lengths(npz_path, json_path, lengths=np.array([64]),
                                  keep_indices=np.array([0]), metadata=metadata)
    assert cache.load_cached_lengths(npz_path, json_path, metadata).lengths.tolist() == [32]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["cache.npz"]
