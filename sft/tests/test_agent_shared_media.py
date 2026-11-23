from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from omnicoach_sft.data.export import parse_args
from omnicoach_sft.data.parquet import (
    AUDIO_LIST_TYPE,
    IMAGE_LIST_TYPE,
    write_agent_parquet,
    write_agent_view,
)
from omnicoach_sft.train.dataset import (
    AgentSharedMediaDataset,
    AgentSharedMediaRegistry,
    expand_agent_manifests,
    expand_agent_shared_turn_parts,
    load_agent_parquet_parts,
    normalize_mixed_training_columns,
    validate_agent_training_artifact,
)


def _source_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field(
                "messages",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("role", pa.string()),
                            pa.field("content", pa.string()),
                            pa.field("loss", pa.bool_()),
                            pa.field("loss_scale", pa.float64()),
                        ]
                    )
                ),
            ),
            pa.field("tools", pa.string()),
            pa.field("images", IMAGE_LIST_TYPE),
            pa.field("audios", AUDIO_LIST_TYPE),
            pa.field("channel", pa.string()),
            pa.field("omni_source_id", pa.string()),
            pa.field("omni_num_images", pa.int32()),
            pa.field("omni_image_num_tokens", pa.int64()),
            pa.field("omni_audio_num_samples", pa.list_(pa.int64())),
            pa.field("omni_last_frame_timestamp_ms", pa.int64()),
        ]
    )


def _write_embedded_source(root: Path) -> tuple[bytes, bytes]:
    root.mkdir()
    image = b"already-canonical-webp-bytes"
    audio = b"already-canonical-wav-bytes"
    image_path = sha256(image).hexdigest() + ".webp"
    rows = [
        {
            "messages": [
                {"role": "user", "content": "<0.0 seconds><image><audio>q", "loss": None, "loss_scale": None},
                {"role": "assistant", "content": "a", "loss": None, "loss_scale": 0.5},
            ],
            "tools": "[]",
            "images": [{"bytes": image, "path": image_path}],
            "audios": [audio],
            "channel": "agent",
            "omni_source_id": "agent:source",
            "omni_num_images": 1,
            "omni_image_num_tokens": 16,
            "omni_audio_num_samples": [16000],
            "omni_last_frame_timestamp_ms": 0,
        },
        {
            "messages": [
                {"role": "user", "content": "<0.0 seconds><image><audio>q2", "loss": None, "loss_scale": None},
                {"role": "assistant", "content": "a2", "loss": None, "loss_scale": 0.5},
            ],
            "tools": "[]",
            "images": [{"bytes": image, "path": image_path}],
            "audios": [audio],
            "channel": "agent",
            "omni_source_id": "agent:source",
            "omni_num_images": 1,
            "omni_image_num_tokens": 16,
            "omni_audio_num_samples": [16000],
            "omni_last_frame_timestamp_ms": 0,
        },
    ]
    schema = _source_schema()
    for index, row in enumerate(rows):
        pq.write_table(pa.Table.from_pylist([row], schema=schema), root / f"part-{index:05d}.parquet")
    (root / "dataset_contract.json").write_text(
        json.dumps(
            {
                "schema_version": "agent_turn_embedded_v1",
                "embedded_media": True,
                "source_id": "agent:source",
                "model_family": "qwen3_omni",
                "purpose": "agentic_turn_sft",
                "protocol": "native-tool-v3-seconds",
                "channel": "agent",
                "columns": schema.names,
                "pack_length_stats": {
                    "image_max_pixels": 147456,
                    "image_token_count_basis": "qwen3_omni_v3_patch16_merge2_plus_vision_wrappers",
                    "image_token_count_column": "omni_image_num_tokens",
                    "audio_sample_rate": 16000,
                    "audio_sample_count_basis": "resampled_16khz",
                },
                "media_encoding": {"images": {"format": "webp"}, "audios": {"format": "wav_pcm_s16le"}},
            }
        ),
        encoding="utf-8",
    )
    (root / "validation.json").write_text(
        json.dumps({"ok": True, "rows": 2, "embedded_media": True}), encoding="utf-8"
    )
    return image, audio


def _write_shared(source, output, limit_rows=0):
    rows = [row for path in sorted(source.glob("*.parquet")) for row in pq.read_table(path).to_pylist()]
    return write_agent_parquet(rows[:limit_rows] if limit_rows else rows,
        output=output, source_id="agent:source", schema=_source_schema(), num_shards=2 if not limit_rows else 1,
        contract=json.loads((source / "dataset_contract.json").read_text()))


def test_embedded_agent_conversion_uses_lazy_media_parquet(tmp_path: Path) -> None:
    source = tmp_path / "embedded"
    image, audio = _write_embedded_source(source)
    output = tmp_path / "shared"

    validation = _write_shared(source, output)

    assert validation["rows"] == 2
    assert validation["media_rows"] == 1
    assert validation["media_row_groups"] == 1
    contract = json.loads((output / "dataset_contract.json").read_text(encoding="utf-8"))
    assert contract["schema_version"] == "agent_turn_shared_media_v1"
    assert contract["embedded_media"] is False
    assert contract["tables"]["turns"]
    assert sum(part["rows"] for part in contract["tables"]["media"]) == 1

    turn_rows = []
    for path in sorted((output / "turns").glob("part-*.parquet")):
        turn_rows.extend(pq.read_table(path).to_pylist())
    assert len(turn_rows) == 2
    assert all(row["images"] == [] and row["audios"] == [] for row in turn_rows)
    assert turn_rows[0]["media_id"] == turn_rows[1]["media_id"]

    media_path = next(path for path in sorted((output / "media").glob("part-*.parquet"))
                      if pq.ParquetFile(path).metadata.num_rows)
    media_parquet = pq.ParquetFile(media_path)
    assert media_parquet.metadata.num_rows == media_parquet.metadata.num_row_groups == 1
    media = media_parquet.read_row_group(0).to_pylist()[0]
    assert media["images"][0]["bytes"] == image
    assert media["audios"] == [audio]


def test_agent_shared_media_loader_reads_bytes_only_for_fetched_row(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "embedded"
    image, audio = _write_embedded_source(source)
    output = tmp_path / "shared"
    _write_shared(source, output)

    parts = expand_agent_shared_turn_parts(output)
    dataset = load_agent_parquet_parts(parts, num_proc=1, cache_dir=tmp_path / "hf-cache")
    validate_agent_training_artifact(dataset, output)
    normalized = normalize_mixed_training_columns(dataset, channel="agent")
    registry = AgentSharedMediaRegistry()
    store = registry.store_for_root(output)
    lazy = AgentSharedMediaDataset(normalized, registry)

    calls = 0
    original_read = store.read

    def counted_read(media_id: str):
        nonlocal calls
        calls += 1
        return original_read(media_id)

    monkeypatch.setattr(store, "read", counted_read)
    assert normalized[0]["images"] == []
    assert calls == 0
    resolved = lazy[0]
    assert calls == 1
    assert resolved["images"][0]["bytes"] == image
    assert resolved["audios"] == [audio]
    assert "media_id" not in resolved
    registry.close()


def test_new_agent_export_defaults_to_shared_parquet(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "omnicoach-sft-export",
            "--source-export",
            str(tmp_path / "source"),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    assert not hasattr(parse_args(), "media_layout")


def test_embedded_agent_conversion_can_limit_smoke_rows(tmp_path: Path) -> None:
    source = tmp_path / "embedded"
    _write_embedded_source(source)

    validation = _write_shared(
        source,
        tmp_path / "shared-smoke",
        limit_rows=1,
    )

    assert validation["rows"] == 1


def test_expand_agent_manifests_and_view_partial_cleanup(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "embedded"
    _write_embedded_source(source)
    agent_dir = tmp_path / "shared"
    _write_shared(source, agent_dir)

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"dataset:\n- {agent_dir}\n", encoding="utf-8")
    non_agent_manifest = tmp_path / "other.yaml"
    non_agent_manifest.write_text(f"dataset:\n- {tmp_path / 'missing'}\n", encoding="utf-8")

    assert expand_agent_manifests([str(manifest), str(non_agent_manifest)]) == [
        str(agent_dir),
        str(non_agent_manifest),
    ]

    import omnicoach_sft.data.parquet as parquet_mod

    view = tmp_path / "broken_view"

    def fail_write(*args, **kwargs):
        raise RuntimeError("injected view write failure")

    monkeypatch.setattr(parquet_mod, "write_json", fail_write)
    with pytest.raises(RuntimeError, match="injected view write failure"):
        write_agent_view(agent_dir, view, updates={0: {"messages": []}})
    assert not view.exists()
    assert not (tmp_path / ".broken_view.partial").exists()
