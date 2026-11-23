"""`probe_video`, a cached ffprobe lookup returning `VideoInfo`."""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from .ffmpeg import require_binary, run


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoInfo:
    duration_seconds: float
    width: int
    height: int
    fps: float
    has_audio: bool


def _rate(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    return float(Fraction(value))


def probe_video(path: str, *, ffprobe: str = "ffprobe") -> VideoInfo:
    if "://" in str(path) or str(path).startswith(("pipe:", "fd:")):
        return _probe_video(path, ffprobe=ffprobe)
    try:
        local = Path(path).absolute()
        identity = local.stat()
    except (OSError, ValueError):
        return _probe_video(path, ffprobe=ffprobe)
    if not stat.S_ISREG(identity.st_mode):
        return _probe_video(path, ffprobe=ffprobe)
    return _probe_video_cached(
        str(local), identity.st_mtime_ns, identity.st_size,
        identity.st_dev, identity.st_ino, ffprobe,
    )


@lru_cache(maxsize=512)
def _probe_video_cached(
    path: str, mtime_ns: int, size: int, device: int, inode: int, ffprobe: str,
) -> VideoInfo:
    # Immutable metadata is reusable only while the local file identity matches.
    # lru_cache retains successful values, not exceptions, without locking probes.
    return _probe_video(path, ffprobe=ffprobe)


def _probe_video(path: str, *, ffprobe: str) -> VideoInfo:
    result = run(
        [
            require_binary(ffprobe),
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            path,
        ],
        timeout_seconds=60,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"no video stream: {path}")
    duration_candidates = [video.get("duration"), (payload.get("format") or {}).get("duration")]
    duration = next((float(item) for item in duration_candidates if item not in {None, "N/A"}), 0.0)
    fps = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate"))
    return VideoInfo(
        duration_seconds=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=fps,
        has_audio=any(item.get("codec_type") == "audio" for item in streams),
    )
