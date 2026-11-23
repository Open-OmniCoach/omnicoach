from __future__ import annotations

import io
import json
import wave
from pathlib import Path

from PIL import Image

from omnicoach_sft.data.evalkit import materialize_evalkit_rows


def test_materialize_evalkit_cache_frame_and_audio_slice(tmp_path: Path) -> None:
    entry = tmp_path / "cache" / "dataset" / "cache-v1" / "media-1"
    entry.mkdir(parents=True)
    frame = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 20, 30)).save(frame, format="JPEG")
    frame_payload = frame.getvalue()
    (entry / "initial.mjpeg").write_bytes(frame_payload)
    with wave.open(str(entry / "audio.wav"), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 32000)
    (entry / "manifest.json").write_text(
        json.dumps(
            {
                "cache_id": "cache-v1",
                "media_id": "media-1",
                "initial": {"file": "initial.mjpeg", "index": [[1.0, 0, len(frame_payload)]]},
                "audio": {"file": "audio.wav"},
            }
        ),
        encoding="utf-8",
    )
    rows = materialize_evalkit_rows(
        [
            {
                "images": ["evalkit-cache://dataset/cache-v1/media-1/initial/frame?timestamp=1.000000"],
                "audios": ["evalkit-cache://dataset/cache-v1/media-1/initial/audio?start=0.5&end=1.5"],
            }
        ],
        resolver={
            "scheme": "evalkit-cache",
            "cache_root": str(tmp_path / "cache"),
            "cache_dataset": "dataset",
        },
        output_root=tmp_path / "materialized",
        workers=2,
    )

    image_path = Path(rows[0]["images"][0])
    audio_path = Path(rows[0]["audios"][0])
    assert image_path.read_bytes() == frame_payload
    with wave.open(str(audio_path), "rb") as audio:
        assert audio.getframerate() == 16000
        assert audio.getnframes() == 16000
