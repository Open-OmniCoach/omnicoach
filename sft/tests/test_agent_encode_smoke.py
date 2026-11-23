from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest

from omnicoach_sft.data.parquet import AUDIO_LIST_TYPE, IMAGE_LIST_TYPE, write_agent_parquet
from omnicoach_sft.train.verify_encode import (
    encode_and_validate,
    load_representative_rows,
    validate_target_weights,
)


class FakeTemplate:
    def __init__(self, *, labels, loss_scale, pieces):
        self._encoded = {
            "input_ids": list(range(len(labels))),
            "labels": labels,
            "loss_scale": loss_scale,
        }
        self._pieces = pieces

    def encode(self, row):
        return self._encoded

    def safe_decode(self, token_ids):
        return "".join(self._pieces[token_id] for token_id in token_ids)


def _answer_row():
    return {
        "channel": "agent",
        "messages": [
            {"role": "user", "content": "question", "loss_scale": None},
            {
                "role": "assistant",
                "content": "<observation>obs</observation><think>think</think><answer>A</answer>",
                "loss_scale": 0.5,
            },
        ],
    }


def test_fractional_tool_call_weight_must_cover_observation_and_call():
    row = {
        "channel": "agent",
        "messages": [
            {"role": "user", "content": "question", "loss_scale": None},
            {
                "role": "assistant",
                "content": "<observation>obs</observation><think>think</think>",
                "loss_scale": 0.5,
            },
            {"role": "tool_call", "content": '{"name":"get_clip","arguments":{}}', "loss_scale": 0.5},
        ],
    }
    assert validate_target_weights(row) == ("tool_call", 0.5)

    row["messages"][-2]["loss_scale"] = None
    with pytest.raises(ValueError, match="observation/think and tool_call"):
        validate_target_weights(row)


def test_fractional_answer_weight_accepts_missing_jsonl_channel():
    row = {
        "channel": "agent",
        "messages": [
            {"role": "user", "content": "question", "loss_scale": None},
            {"role": "assistant", "content": "<answer>A</answer>", "loss_scale": 1 / 3},
        ],
    }
    action, weight = validate_target_weights(row)
    assert action == "answer"
    assert weight == pytest.approx(1 / 3)

    row.pop("channel")
    assert validate_target_weights(row) == ("answer", pytest.approx(1 / 3))

    row["channel"] = "other"
    with pytest.raises(ValueError, match="channel='agent'"):
        validate_target_weights(row)


def test_jsonl_loader_selects_recovery_after_plain_answer(tmp_path):
    tool_row = {
        "messages": [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "<observation>obs</observation><think>think</think>",
                "loss_scale": 0.5,
            },
            {
                "role": "tool_call",
                "content": '{"name":"get_clip","arguments":{}}',
                "loss_scale": 0.5,
            },
        ]
    }
    answer_row = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "<answer>A</answer>", "loss_scale": 0.5},
        ]
    }
    recovery_row = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "invalid", "loss": False},
            {"role": "user", "content": "[ERROR] INVALID_JSON"},
            {"role": "assistant", "content": "<answer>A</answer>", "loss_scale": 0.5},
        ]
    }
    path = tmp_path / "train.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in (tool_row, answer_row, recovery_row)),
        encoding="utf-8",
    )

    selected = load_representative_rows(path)

    assert set(selected) == {"tool_call", "answer", "recovery"}
    assert any(message.get("loss") is False for message in selected["recovery"]["messages"])


def test_encode_requires_exact_weight_only_on_current_target():
    template = FakeTemplate(
        labels=[-100, 1, 2, 3, 4],
        loss_scale=[0.0, 0.5, 0.5, 0.5, 0.5],
        pieces=[
            "historical context",
            "<observation>obs</observation>",
            "<think>think</think>",
            "<answer>A</answer>",
            "<|im_end|>",
        ],
    )
    result = encode_and_validate(template, _answer_row())
    assert result["supervised_tokens"] == 4
    assert result["masked_tokens"] == 1
    assert "historical context" not in result["supervised_text"]


def test_encode_accepts_think_only_protocol_without_observation():
    template = FakeTemplate(
        labels=[-100, 1, 2, 3],
        loss_scale=[0.0, 0.5, 0.5, 0.5],
        pieces=[
            "historical context",
            "<think>think</think>",
            "<answer>A</answer>",
            "<|im_end|>",
        ],
    )

    result = encode_and_validate(
        template,
        _answer_row(),
        require_observation=False,
    )

    assert "<think>" in result["supervised_text"]
    assert "<observation>" not in result["supervised_text"]


@pytest.mark.parametrize(
    ("loss_scale", "error"),
    [
        ([0.0, 0.5, 1.0, 0.5, 0.5], "supervised tokens"),
        ([0.25, 0.5, 0.5, 0.5, 0.5], "masked tokens"),
    ],
)
def test_encode_rejects_supervised_or_masked_weight_leaks(loss_scale, error):
    template = FakeTemplate(
        labels=[-100, 1, 2, 3, 4],
        loss_scale=loss_scale,
        pieces=[
            "context",
            "<observation>obs</observation>",
            "<think>think</think>",
            "<answer>A</answer>",
            "<|im_end|>",
        ],
    )
    with pytest.raises(AssertionError, match=error):
        encode_and_validate(template, _answer_row())


def test_encode_rejects_historical_tool_response_in_supervised_decode():
    template = FakeTemplate(
        labels=[0, 1, 2, 3, 4],
        loss_scale=[0.5] * 5,
        pieces=[
            "<tool_response>history</tool_response>",
            "<observation>obs</observation>",
            "<think>think</think>",
            "<answer>A</answer>",
            "<|im_end|>",
        ],
    )
    with pytest.raises(AssertionError, match="historical tool response leaked"):
        encode_and_validate(template, _answer_row())


def test_parquet_loader_resolves_lazy_agent_shared_media_rows(tmp_path: Path) -> None:
    image = b"canonical-webp-bytes"
    audio = b"canonical-wav-bytes"
    schema = pa.schema(
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
        ]
    )
    rows = [
        {
            "messages": [
                {"role": "user", "content": "<0.0 seconds><image><audio>q1", "loss": None, "loss_scale": None},
                {
                    "role": "assistant",
                    "content": "<observation>obs</observation><think>think</think>",
                    "loss": None,
                    "loss_scale": 0.5,
                },
                {"role": "tool_call", "content": '{"name":"get_clip","arguments":{}}', "loss": None, "loss_scale": 0.5},
            ],
            "tools": "[]",
            "images": [{"bytes": image, "path": "unused"}],
            "audios": [audio],
            "channel": "agent",
        },
        {
            "messages": [
                {"role": "user", "content": "<0.0 seconds><image><audio>q2", "loss": None, "loss_scale": None},
                {"role": "assistant", "content": "bad", "loss": False, "loss_scale": None},
                {"role": "user", "content": "[ERROR] INVALID_JSON", "loss": None, "loss_scale": None},
                {"role": "assistant", "content": "<answer>A</answer>", "loss": None, "loss_scale": 0.5},
            ],
            "tools": "[]",
            "images": [{"bytes": image, "path": "unused"}],
            "audios": [audio],
            "channel": "agent",
        },
    ]
    out = tmp_path / "agent_shared"
    write_agent_parquet(rows, output=out, source_id="agent:smoke", schema=schema)

    selected = load_representative_rows(out)

    assert set(selected) == {"tool_call", "answer", "recovery"}
    assert selected["tool_call"]["images"][0]["bytes"] == image
    assert selected["tool_call"]["audios"] == [audio]
    assert selected["answer"]["audios"] == [audio]
    assert "media_id" not in selected["tool_call"]
