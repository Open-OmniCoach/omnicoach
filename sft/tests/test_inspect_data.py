from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pyarrow as pa

from omnicoach_sft.data.inspect import inspect_source
from omnicoach_sft.data.parquet import AUDIO_LIST_TYPE, IMAGE_LIST_TYPE, write_agent_parquet


def _build_agent_dir(root: Path, name: str = "agent_source") -> Path:
    img = b"canonical-webp"
    aud = b"canonical-wav"
    img_path = sha256(img).hexdigest() + ".webp"
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
            pa.field("omni_source_id", pa.string()),
            pa.field("omni_num_images", pa.int32()),
            pa.field("omni_image_num_tokens", pa.int64()),
            pa.field("omni_audio_num_samples", pa.list_(pa.int64())),
            pa.field("omni_last_frame_timestamp_ms", pa.int64()),
        ]
    )
    rows = [
        {
            "messages": [
                {"role": "user", "content": "<0.0 seconds><image><audio>q", "loss": None, "loss_scale": None},
                {"role": "assistant", "content": "<answer>A</answer>", "loss": None, "loss_scale": 0.5},
            ],
            "tools": "[]",
            "images": [{"bytes": img, "path": img_path}],
            "audios": [aud],
            "channel": "agent",
            "omni_source_id": f"agent:{name}",
            "omni_num_images": 1,
            "omni_image_num_tokens": 16,
            "omni_audio_num_samples": [16000],
            "omni_last_frame_timestamp_ms": 0,
        }
    ]
    out = root / name
    write_agent_parquet(
        rows,
        output=out,
        source_id=f"agent:{name}",
        schema=schema,
        contract={"protocol": "native-tool-v3-seconds"},
    )
    return out


def test_inspect_source_validates_and_summarizes_agent_parquet(tmp_path: Path) -> None:
    agent_dir = _build_agent_dir(tmp_path)

    summary = inspect_source(agent_dir, validate=True, sample_index=0)
    assert summary["validated_turn_rows"] == 1
    assert summary["indexed_media_rows"] == 1
    assert summary["sample"] == {"channel": "agent", "messages": 2, "frames": 1, "audio_tracks": 1}
    assert json.dumps(summary)
