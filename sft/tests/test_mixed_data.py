from __future__ import annotations

import json

import pytest
from datasets import Dataset, Features, Value, concatenate_datasets

from omnicoach_sft.train.dataset import (
    load_agent_parquet_parts,
    normalize_mixed_training_columns,
    prepare_agent_length_dataset,
    validate_agent_training_artifact,
)


def _buffer_addresses(dataset: Dataset, column: str) -> tuple[int, ...]:
    return tuple(
        buffer.address
        for chunk in dataset.data.table.column(column).chunks
        for buffer in chunk.buffers()
        if buffer is not None
    )


def test_agent_parts_load_as_one_ordered_train_split(tmp_path):
    parts = []
    for part_index in range(3):
        part = tmp_path / f"part-{part_index:05d}.parquet"
        Dataset.from_dict(
            {
                "row_id": [part_index * 2, part_index * 2 + 1],
                "messages": [
                    [{"role": "user", "content": f"q{part_index}-0"}],
                    [{"role": "user", "content": f"q{part_index}-1"}],
                ],
            }
        ).to_parquet(part)
        parts.append(str(part))

    loaded = load_agent_parquet_parts(
        parts,
        num_proc=2,
        cache_dir=tmp_path / "cache",
    )

    assert loaded["row_id"] == list(range(6))
    assert len(loaded.cache_files) == 2


def test_base_and_agent_nested_message_features_can_be_concatenated():
    base = Dataset.from_dict(
        {
            "messages": [[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]],
            "task_id": ["base/q"],
            "omni_source_id": ["base"],
            "omni_num_images": [1],
            "omni_audio_num_samples": [[]],
            "omni_last_frame_timestamp_ms": [0],
        }
    )
    agent = Dataset.from_dict(
        {
            "messages": [
                [
                    {"role": "user", "content": "q1", "loss": None, "loss_scale": None},
                    {"role": "assistant", "content": "context", "loss": False, "loss_scale": None},
                    {"role": "user", "content": "q2", "loss": None, "loss_scale": None},
                    {
                        "role": "assistant",
                        "content": "<answer>A</answer>",
                        "loss": None,
                        "loss_scale": 0.5,
                    },
                ]
            ],
            "channel": ["agent"],
            "omni_source_id": ["agent"],
            "omni_num_images": [1],
            "omni_audio_num_samples": [[16000]],
            "omni_last_frame_timestamp_ms": [0],
        }
    )

    normalized_base = normalize_mixed_training_columns(base, channel="base")
    normalized_agent = normalize_mixed_training_columns(agent, channel="agent")
    mixed = concatenate_datasets([normalized_base, normalized_agent])

    assert mixed[0]["channel"] == "base"
    assert mixed[0]["messages"][-1]["loss_scale"] is None
    assert mixed[1]["channel"] == "agent"
    assert mixed[1]["messages"][1]["loss"] is False
    assert mixed[1]["messages"][-1]["loss_scale"] == 0.5
    assert mixed[1]["messages"][-1]["loss"] is None
    assert str(mixed.features["omni_num_images"]) == "Value('int64')"
    assert str(mixed.features["omni_audio_num_samples"]) == "List(Value('int64'))"


def test_canonical_agent_schema_does_not_map_or_copy_media(monkeypatch):
    features = Features(
        {
            "messages": [
                {
                    "role": Value("string"),
                    "content": Value("string"),
                    "loss": Value("bool"),
                    "loss_scale": Value("float64"),
                }
            ],
            "images": [Value("binary")],
            "audios": [Value("binary")],
            "channel": Value("string"),
            "omni_source_id": Value("string"),
            "omni_num_images": Value("int32"),
            "omni_audio_num_samples": [Value("int64")],
            "omni_last_frame_timestamp_ms": Value("int64"),
        }
    )
    agent = Dataset.from_dict(
        {
            "messages": [
                [
                    {"role": "user", "content": "q", "loss": None, "loss_scale": None},
                    {
                        "role": "assistant",
                        "content": "<answer>A</answer>",
                        "loss": None,
                        "loss_scale": 0.5,
                    },
                ]
            ],
            "images": [[b"image-payload"]],
            "audios": [[b"audio-payload"]],
            "channel": ["agent"],
            "omni_source_id": ["agent"],
            "omni_num_images": [1],
            "omni_audio_num_samples": [[16000]],
            "omni_last_frame_timestamp_ms": [0],
        },
        features=features,
    )
    image_buffers = _buffer_addresses(agent, "images")
    audio_buffers = _buffer_addresses(agent, "audios")

    def fail_map(*args, **kwargs):
        raise AssertionError("canonical Agent schema must not call Dataset.map")

    monkeypatch.setattr(Dataset, "map", fail_map)
    normalized = normalize_mixed_training_columns(agent, channel="agent")

    assert str(normalized.features["omni_num_images"]) == "Value('int64')"
    assert normalized[0]["messages"][-1]["loss"] is None
    assert _buffer_addresses(normalized, "images") == image_buffers
    assert _buffer_addresses(normalized, "audios") == audio_buffers


def test_agent_length_prewarm_uses_stable_training_fingerprint():
    agent = Dataset.from_dict(
        {
            "messages": [
                [
                    {"role": "user", "content": "q", "loss": None, "loss_scale": None},
                    {
                        "role": "assistant",
                        "content": "<answer>A</answer>",
                        "loss": None,
                        "loss_scale": 0.5,
                    },
                ]
            ],
            "channel": ["agent"],
        }
    )

    prepared = prepare_agent_length_dataset(
        agent,
        source="/data/canonical-agent",
        channel="agent",
    )
    repeated = prepare_agent_length_dataset(
        prepared,
        source="/data/canonical-agent",
        channel="agent",
    )

    assert prepared["dataset"] == ["/data/canonical-agent"]
    assert repeated._fingerprint == prepared._fingerprint


def test_base_message_normalization_uses_arrow_without_copying_media(monkeypatch):
    base = Dataset.from_dict(
        {
            "messages": [[{"role": "user", "content": "q"}]],
            "images": [[b"image-payload"]],
            "audios": [[b"audio-payload"]],
        }
    )
    image_buffers = _buffer_addresses(base, "images")
    audio_buffers = _buffer_addresses(base, "audios")
    content_buffer = base.data.table.column("messages").chunks[0].values.field("content").buffers()[-1].address

    def fail_map(*args, **kwargs):
        raise AssertionError("schema alignment must not map Python rows")

    monkeypatch.setattr(Dataset, "map", fail_map)
    normalized = normalize_mixed_training_columns(base, channel="base")

    assert normalized[0]["messages"][0]["content"] == "q"
    assert normalized.data.table.column("messages").chunks[0].values.field("content").buffers()[-1].address == content_buffer
    assert normalized[0]["messages"][0]["loss_scale"] is None
    assert normalized[0]["messages"][0]["loss"] is None
    assert normalized["channel"] == ["base"]
    assert _buffer_addresses(normalized, "images") == image_buffers
    assert _buffer_addresses(normalized, "audios") == audio_buffers


def test_published_agent_artifact_validation_is_metadata_only(tmp_path, monkeypatch):
    source = tmp_path / "agent"
    source.mkdir()
    dataset = Dataset.from_dict(
        {
            "messages": [[{"role": "assistant", "content": "a", "loss_scale": 0.5}]],
            "tools": ["[]"],
            "images": [[]],
            "audios": [[]],
            "channel": ["agent"],
            "omni_source_id": ["agent:test"],
            "omni_num_images": [0],
            "omni_audio_num_samples": [[]],
            "omni_last_frame_timestamp_ms": [0],
        }
    )
    (source / "dataset_contract.json").write_text(
        json.dumps(
            {
                "schema_version": "agent_turn_shared_media_v1",
                "table_layout": "agent_turn_shared_media_v1",
                "embedded_media": False,
                "media_columns": "lazy_embedded_media_parquet",
                "columns": sorted(dataset.column_names),
            }
        ),
        encoding="utf-8",
    )
    (source / "validation.json").write_text(
        json.dumps({"ok": True, "rows": 1, "embedded_media": False}),
        encoding="utf-8",
    )

    def fail_map(*args, **kwargs):
        raise AssertionError("artifact validation must not map embedded media")

    monkeypatch.setattr(Dataset, "map", fail_map)
    validate_agent_training_artifact(dataset, source)

    (source / "validation.json").write_text(
        json.dumps({"ok": True, "rows": 2, "embedded_media": False}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="row count"):
        validate_agent_training_artifact(Dataset.from_dict({"messages": [[]]}), source)
