from __future__ import annotations

import json
import subprocess
import sys
import wave
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from omnicoach_sft.data.export import (
    _qwen_image_token_contribution,
    audio_num_samples_16khz,
    cache_base_wav,
    cache_base_webp,
    embedded_image_num_tokens,
    group_rows_by_base_provenance,
    main as export_agent_main,
    probe_image_token_counts,
    validate_agent_row,
)


def test_image_token_probe_reports_all_invalid_paths(tmp_path) -> None:
    first = tmp_path / "first.webp"
    second = tmp_path / "second.webp"
    first.write_bytes(b"broken")
    second.write_bytes(b"also broken")

    with pytest.raises(ValueError, match=r"2 invalid image path\(s\)"):
        probe_image_token_counts([{"images": [str(first), str(second)]}], num_proc=2)


def make_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_clip",
                "description": "Return a clip.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def make_agent_row(*, loss_scale: float = 1 / 3) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "<observation>obs</observation>\n<think>think</think>",
            },
            {
                "role": "tool_call",
                "content": '{"name":"get_clip","arguments":{"start":"00:10","end":"00:20"}}',
            },
            {"role": "tool_response", "content": "Clip [00:10,00:20]."},
            {
                "role": "assistant",
                "content": "<observation>obs</observation>\n<think>think</think>\n<answer>A</answer>",
                "loss_scale": loss_scale,
            },
        ],
        "tools": json.dumps(make_tools()),
        "images": [],
        "audios": [],
        "channel": "agent",
        "omni_source_id": "agent:test",
        "omni_num_images": 0,
        "omni_audio_num_samples": [],
        "omni_last_frame_timestamp_ms": 0,
    }


def test_embedded_agent_features_use_large_binary_media_offsets() -> None:
    features = build_features(True, embedded_image_struct=True, embedded_large_binary=True)
    schema = features.arrow_schema

    assert str(schema.field("images").type).startswith("list<item: struct<bytes: large_binary")
    assert str(schema.field("audios").type) == "list<item: large_binary>"


def test_weighted_input_shards_balance_media_work_without_reordering() -> None:
    rows = [{"id": index} for index in range(8)]
    weights = [1, 1, 1, 1, 10, 10, 10, 10]

    shards = shard_rows(rows, 2, weights=weights)
    flattened = [row for shard in shards for row in shard]
    shard_weights: list[int] = []
    offset = 0
    for shard in shards:
        shard_weights.append(sum(weights[offset : offset + len(shard)]))
        offset += len(shard)

    assert flattened == rows
    assert shard_weights == [24, 20]


def test_grouped_layout_keeps_large_base_atomic_across_row_groups_and_shards() -> None:
    assert plan_grouped_parquet_layout(
        ["base-a"] * 3 + ["base-b"] * 2 + ["base-c"] * 5,
        row_group_size=4,
        num_shards=2,
    ) == [[3, 2], [5]]

    with pytest.raises(ValueError, match="not contiguous"):
        plan_grouped_parquet_layout(
            ["base-a", "base-b", "base-a"],
            row_group_size=4,
            num_shards=1,
        )


def test_base_provenance_grouping_is_stable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    provenance = [
        {"row_idx": 0, "base_sample_id": "base-b"},
        {"row_idx": 1, "base_sample_id": "base-a"},
        {"row_idx": 2, "base_sample_id": "base-b"},
    ]
    (source / "provenance.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in provenance),
        encoding="utf-8",
    )

    rows, group_ids, stats = group_rows_by_base_provenance(
        source,
        [{"id": 0}, {"id": 1}, {"id": 2}],
    )

    assert [row["id"] for row in rows] == [0, 2, 1]
    assert group_ids == ["base-b", "base-b", "base-a"]
    assert stats["base_groups"] == 2
    assert stats["source_order_changed"] is True


def test_agent_validator_accepts_masked_invalid_history_before_feedback() -> None:
    row = make_agent_row(loss_scale=0.5)
    row["messages"][2:2] = [
        {
            "role": "assistant",
            "content": "malformed invalid output",
            "loss": False,
        },
        {
            "role": "user",
            "content": "[ERROR] INVALID_JSON: expected one valid action",
        },
    ]

    stats = validate_agent_row(row, context="fixture", embedded_media=False)

    assert stats["target_loss_scale"] == 0.5
    assert stats["masked_invalid_history_messages"] == 1


def test_agent_validator_rejects_separate_parser_state_control_user() -> None:
    row = make_agent_row(loss_scale=0.5)
    row["messages"].insert(
        2,
        {
            "role": "user",
            "content": (
                "[PARSER_STATE] Write observation and think, then act."
                "\n[NOTICE] Step 0/32. 32 steps remaining."
            ),
        },
    )

    with pytest.raises(ValueError, match=r"invalid role transition.*user -> user"):
        validate_agent_row(row, context="fixture", embedded_media=False)


def test_validate_agent_row_accepts_tool_response_media_alignment() -> None:
    row = make_agent_row(loss_scale=0.5)
    row["messages"][4]["content"] = "Clip [00:10,00:20].\n<10.0 seconds>\n<image>\n<audio>"
    row["images"] = ["frame.webp"]
    row["audios"] = ["clip.wav"]
    row["omni_num_images"] = 1
    row["omni_audio_num_samples"] = [160000]
    row["omni_last_frame_timestamp_ms"] = 10000

    stats = validate_agent_row(row, context="fixture", embedded_media=False)
    assert stats["tool_calls"] == 1
    assert stats["media_counts"] == {"image": 1, "audio": 1, "video": 0}


def test_audio_sample_count_uses_resampled_16khz_basis(tmp_path) -> None:
    audio_path = tmp_path / "clip.wav"
    with wave.open(str(audio_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\0\0" * 8000)

    assert audio_num_samples_16khz(str(audio_path)) == 16000


def test_base_media_cache_writes_valid_direct_embedded_payloads(tmp_path) -> None:
    image_path = tmp_path / "frame.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x0a141e:s=298x168",
            "-frames:v",
            "1",
            "-y",
            str(image_path),
        ],
        check=True,
    )
    audio_path = tmp_path / "clip.wav"
    with wave.open(str(audio_path), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(48000)
        audio.writeframes(b"\0\0\0\0" * 4800)

    cached_image = cache_base_webp(str(image_path), cache_root=tmp_path / "media_cache")
    cached_audio = cache_base_wav(str(audio_path), cache_root=tmp_path / "media_cache")
    image_bytes = Path(cached_image).read_bytes()
    audio_bytes = Path(cached_audio).read_bytes()
    image_name = f"{sha256(image_bytes).hexdigest()}.webp"
    assert cached_image.endswith(image_name)
    assert embedded_image_num_tokens(
        [{"bytes": image_bytes, "path": image_name}],
        context="fixture",
    ) == _qwen_image_token_contribution(304, 160)

    row = make_agent_row(loss_scale=0.5)
    row["messages"][4]["content"] = "Clip [00:10,00:20].\n<10.0 seconds>\n<image>\n<audio>"
    row["images"] = [{"bytes": image_bytes, "path": image_name}]
    row["audios"] = [audio_bytes]
    row["omni_num_images"] = 1
    row["omni_audio_num_samples"] = [1600]
    row["omni_last_frame_timestamp_ms"] = 10000

    stats = validate_agent_row(row, context="embedded fixture", embedded_media=True)
    assert stats["media_counts"] == {"image": 1, "audio": 1, "video": 0}
    with wave.open(cached_audio, "rb") as audio:
        assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 16000)
        assert audio.getnframes() == 1600


def test_base_media_cache_preserves_contract_compatible_webp(tmp_path) -> None:
    image_path = tmp_path / "frame.webp"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x0a141e:s=224x128",
            "-frames:v",
            "1",
            "-c:v",
            "libwebp",
            "-y",
            str(image_path),
        ],
        check=True,
    )

    source_payload = image_path.read_bytes()
    cached_image = cache_base_webp(str(image_path), cache_root=tmp_path / "media_cache")
    assert Path(cached_image).read_bytes() == source_payload
    assert Path(cached_image).name == f"{sha256(source_payload).hexdigest()}.webp"


def test_validate_agent_row_rejects_historical_media_when_latest_response_exists() -> None:
    row = make_agent_row(loss_scale=0.5)
    row["messages"][1]["content"] = "question\n<0.0 seconds>\n<image>"
    row["messages"][4]["content"] = "Clip [00:10,00:20].\n<10.0 seconds>\n<image>"
    row["images"] = ["init.webp", "latest.webp"]

    with pytest.raises(ValueError, match="at most one media-bearing message"):
        validate_agent_row(row, context="fixture", embedded_media=False)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda row: row["messages"].pop(4), "invalid role transition"),
        (lambda row: row["messages"][3].update(content="not-json"), "invalid tool_call JSON"),
        (lambda row: row["messages"][3].update(content='{"name":"unknown","arguments":{}}'), "unknown tool"),
        (lambda row: row["messages"][-1].pop("loss_scale"), "loss_scale"),
        (lambda row: row["images"].append("unused.webp"), "placeholder/media mismatch"),
    ],
)
def test_validate_agent_row_rejects_broken_protocol(mutation, error) -> None:
    row = deepcopy(make_agent_row())
    mutation(row)
    with pytest.raises(ValueError, match=error):
        validate_agent_row(row, context="fixture", embedded_media=False)


def test_validate_agent_row_requires_equal_assistant_and_tool_call_weights() -> None:
    row = make_agent_row(loss_scale=0.5)
    row["messages"] = row["messages"][:4]
    row["messages"][-2]["loss_scale"] = 0.5
    row["messages"][-1]["loss_scale"] = 1 / 3

    with pytest.raises(ValueError, match="identical loss_scale"):
        validate_agent_row(row, context="fixture", embedded_media=False)



from omnicoach_sft.data.parquet import write_agent_parquet, write_agent_view
from omnicoach_sft.io import build_features, plan_grouped_parquet_layout, shard_rows


def _shared_rows():
    return [{"messages": [{"role": "assistant", "content": str(index), "loss_scale": 1 / 3}],
             "images": [{"bytes": b"canonical-webp", "path": "unused"}], "audios": [],
             "tools": "[]"} for index in range(6)]


def _shared_schema():
    return build_features(True, embedded_image_struct=True, embedded_large_binary=True,
                          include_tools=True, include_message_loss_scale=True).arrow_schema


def test_shared_writer_preserves_atomic_groups_and_exact_weights(tmp_path):
    rows = _shared_rows()
    output = tmp_path / "shared"
    result = write_agent_parquet(rows, output=output, source_id="fixture", schema=_shared_schema(),
                                group_ids=["a"] * 4 + ["b"] * 2, row_group_size=2, num_shards=2, workers=2)
    turns = [row for path in sorted((output / "turns").glob("*.parquet")) for row in pq.read_table(path).to_pylist()]
    assert [row["messages"][0]["content"] for row in turns] == [str(index) for index in range(6)]
    assert all(row["messages"][0]["loss_scale"] == 1 / 3 for row in turns)
    assert all(not row["images"] and row["media_id"] for row in turns)
    assert result["media_rows"] == 1
    assert [pq.ParquetFile(path).metadata.row_group(0).num_rows for path in sorted((output / "turns").glob("*.parquet"))] == [4, 2]


def test_shared_writer_single_shard_payload_parallelism_is_bounded(tmp_path, monkeypatch):
    import threading
    import time

    import omnicoach_sft.data.parquet as writer
    lock = threading.Lock()
    active = peak = 0
    original = writer._bundle
    def bundle(row):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.02)
            return original(row)
        finally:
            with lock:
                active -= 1
    monkeypatch.setattr(writer, "_bundle", bundle)
    writer.write_agent_parquet(_shared_rows(), output=tmp_path / "shared", source_id="fixture",
                               schema=_shared_schema(), workers=3, row_group_size=4)
    assert 1 < peak <= 3


def test_shared_writer_resumes_distributed_parts_and_finalize_failure(tmp_path, monkeypatch):
    import omnicoach_sft.data.parquet as writer
    rows, output, partial = _shared_rows(), tmp_path / "shared", tmp_path / ".shared.partial"
    arguments = {
        "output": output,
        "partial": partial,
        "source_id": "fixture",
        "schema": _shared_schema(),
        "num_shards": 2,
        "row_group_size": 2,
    }
    for index in range(2):
        writer.write_agent_parquet(rows, **arguments, worker_index=index, worker_count=2,
                                   build_only=True, resume=index > 0)
    def interrupted(root):
        raise RuntimeError("injected failure after media deduplication")
    with pytest.raises(RuntimeError, match="injected failure"):
        writer.write_agent_parquet(rows, **arguments, resume=True, finalize_artifact=interrupted)
    assert partial.exists() and not output.exists()
    def no_rebuild(row):
        raise AssertionError("completed payload was rebuilt")
    monkeypatch.setattr(writer, "_bundle", no_rebuild)
    result = writer.write_agent_parquet(rows, **arguments, resume=True)
    assert result["rows"] == 6 and result["media_rows"] == 1


def test_shared_writer_rejects_changed_resume_identity(tmp_path):
    rows = _shared_rows()
    arguments = {
        "output": tmp_path / "shared",
        "source_id": "fixture",
        "schema": _shared_schema(),
        "num_shards": 2,
    }
    write_agent_parquet(rows, **arguments, build_only=True)
    rows[0]["messages"][0]["content"] = "different"
    with pytest.raises(ValueError, match="identity"):
        write_agent_parquet(rows, **arguments, resume=True)


def test_shared_view_rewrites_loss_and_hardlinks_unchanged_parts(tmp_path):
    from omnicoach_sft.train.dataset import AgentSharedMediaStore
    parent, view = tmp_path / "parent", tmp_path / "view"
    write_agent_parquet(_shared_rows(), output=parent, source_id="fixture", schema=_shared_schema(), num_shards=2)
    messages = [{"role": "assistant", "content": "0", "loss_scale": 1 / 7}]
    write_agent_view(parent, view, updates={0: {"messages": messages}})
    original = parent / "turns" / "part-00001.parquet"
    unchanged = view / "turns" / original.name
    assert original.stat().st_ino == unchanged.stat().st_ino
    row = pq.read_table(view / "turns" / "part-00000.parquet").to_pylist()[0]
    assert row["messages"][0]["loss_scale"] == 1 / 7
    store = AgentSharedMediaStore(view)
    assert store.read(row["media_id"])["images"][0]["bytes"] == b"canonical-webp"
    store.close()


def test_agent_export_main_writes_shared_contract_and_training_rows(tmp_path, monkeypatch):
    from omnicoach_sft.train.dataset import (
        expand_agent_shared_turn_parts,
        load_agent_parquet_parts,
        validate_agent_training_artifact,
    )
    source, output = tmp_path / "source", tmp_path / "shared"
    source.mkdir()
    (source / "train.jsonl").write_text(json.dumps(make_agent_row()) + "\n")
    (source / "provenance.jsonl").write_text(json.dumps({"row_idx": 0, "base_sample_id": "base"}) + "\n")
    (source / "dataset_contract.json").write_text(json.dumps({
        "protocol": "native-tool-v3-seconds-think-only", "required_chat_template": "qwen3_omni_obs_think_tool_v1"}))
    (source / "validation.json").write_text(json.dumps({"ok": True, "train_rows": 1}))
    monkeypatch.setattr(sys, "argv", ["export", "--source-export", str(source), "--out", str(output),
                                    "--group-by-base", "--num-proc", "2"])
    export_agent_main()
    contract = json.loads((output / "dataset_contract.json").read_text())
    assert contract["table_layout"] == "agent_turn_shared_media_v1"
    assert contract["physical_layout"]["atomic_base_groups"]
    assert contract["pack_length_stats"]["audio_sample_rate"] == 16000
    dataset = load_agent_parquet_parts(expand_agent_shared_turn_parts(output), num_proc=1, cache_dir=tmp_path / "hf")
    validate_agent_training_artifact(dataset, output)
    assert dataset[0]["messages"][-1]["loss_scale"] == 1 / 3


def test_shared_writer_reuses_prior_bundle_when_later_in_window(tmp_path, monkeypatch):
    import omnicoach_sft.data.parquet as writer
    first, second = tmp_path / "a.webp", tmp_path / "b.webp"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    rows = _shared_rows()[:4]
    for row, path in zip(rows, (first, first, second, first), strict=True):
        row["images"] = [str(path)]
    calls = []
    original = writer._bundle
    def bundle(row):
        calls.append(row["images"][0])
        return original(row)
    monkeypatch.setattr(writer, "_bundle", bundle)
    result = writer.write_agent_parquet(rows, output=tmp_path / "shared", source_id="fixture",
                                       schema=_shared_schema(), row_group_size=2, workers=2)
    assert result["media_rows"] == 2
    assert calls.count(str(first)) == calls.count(str(second)) == 1


def test_shared_writer_concurrent_shard_workers_initialize_one_plan(tmp_path):
    code = """
import sys
from omnicoach_sft.data.parquet import write_agent_parquet
from omnicoach_sft.io import build_features
rows = [{'messages': [{'role': 'assistant', 'content': str(i)}], 'images': [], 'audios': []} for i in range(4)]
write_agent_parquet(rows, output=sys.argv[1], source_id='parallel',
    schema=build_features(True, embedded_image_struct=True, embedded_large_binary=True).arrow_schema,
    num_shards=2, worker_index=int(sys.argv[2]), worker_count=2, build_only=True)
"""
    output = tmp_path / "parallel"
    children = [subprocess.Popen([sys.executable, "-c", code, str(output), str(index)]) for index in range(2)]
    assert [child.wait(timeout=15) for child in children] == [0, 0]
    rows = [{"messages": [{"role": "assistant", "content": str(index)}], "images": [], "audios": []} for index in range(4)]
    result = write_agent_parquet(rows, output=output, source_id="parallel", num_shards=2, resume=True,
        schema=build_features(True, embedded_image_struct=True, embedded_large_binary=True).arrow_schema)
    assert result["rows"] == 4


def test_validate_agent_row_supports_get_multi_span() -> None:
    from collections import Counter

    from omnicoach_sft.data.schema import _validate_tool_media_alignment

    tools = [
        *make_tools(),
        {
            "type": "function",
            "function": {
                "name": "get_multi_span",
                "description": "Fetch multiple spans.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
    row = make_agent_row(loss_scale=0.5)
    row["tools"] = json.dumps(tools)
    row["messages"][3]["content"] = '{"name":"get_multi_span","arguments":{"spans":[]}}'
    row["messages"][4]["content"] = "Multi-span.\n<1.0 seconds>\n<image>\n<audio>\n<audio>"
    row["images"] = ["frame.webp"]
    row["audios"] = ["a1.wav", "a2.wav"]
    row["omni_num_images"] = 1
    row["omni_audio_num_samples"] = [16000, 16000]
    row["omni_last_frame_timestamp_ms"] = 1000

    stats = validate_agent_row(row, context="fixture", embedded_media=False)
    assert stats["media_counts"] == {"image": 1, "audio": 2, "video": 0}

    roles = [m["role"] for m in row["messages"]]
    with pytest.raises(ValueError, match="get_multi_span response must contain at least one image or audio payload"):
        _validate_tool_media_alignment(
            row["messages"],
            roles,
            [4],
            Counter({"image": 0, "audio": 0}),
            context="fixture",
        )
