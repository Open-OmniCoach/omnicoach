"""`Framepack` and the helpers that build this route's first user message."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

from ..domain.message import MediaRef, Message, media_part, message, text_part
from ..domain.task import TaskInput
from ..media.cache import ensure_operation_dir, operation_lock
from ..media.extract import extract_frames, extract_full_audio, frame_filename
from ..media.probe import probe_video
from ..media.timecode import format_timecode
from ..providers.messages import estimated_data_url_bytes
from .contract import CONTRACT
from .prompt import initial_text


@dataclass(frozen=True, slots=True, kw_only=True)
class Framepack:
    summary: str
    frames: tuple[tuple[float, MediaRef], ...]
    audio: MediaRef | None
    audio_end: float
    has_audio: bool


def select_timestamps(*, duration: float) -> list[float]:
    if duration <= 0:
        candidates = [0.0]
    else:
        last = max(0.0, duration - 0.2)
        step = 1.0 / CONTRACT.framepack_fps
        count = max(1, math.floor(last / step) + 1)
        candidates = [min(last, index * step) for index in range(count)]
    if len(candidates) <= CONTRACT.framepack_max_frames:
        return candidates
    last_index = len(candidates) - 1
    return [
        candidates[round(index * last_index / (CONTRACT.framepack_max_frames - 1))]
        for index in range(CONTRACT.framepack_max_frames)
    ]


def build_framepack(
    task: TaskInput,
    *,
    work_dir: Path,
    ffmpeg: str,
    ffprobe: str = "ffprobe",
) -> Framepack:
    info = (
        probe_video(task.video_path, ffprobe=ffprobe)
        if task.duration_seconds is None or task.has_audio is None
        else None
    )
    duration = float(task.duration_seconds if task.duration_seconds is not None else info.duration_seconds)
    has_audio = bool(task.has_audio if task.has_audio is not None else info.has_audio)
    audio_source = task.audio_path or task.video_path
    timestamps = select_timestamps(duration=duration)
    out = ensure_operation_dir(
        work_dir,
        name="omni_ota_framepack",
        arguments={
            "video": task.video_path,
            "audio_source": audio_source if has_audio else None,
            "duration": duration,
            "has_audio": has_audio,
            "fps": CONTRACT.framepack_fps,
            "max_frames": CONTRACT.framepack_max_frames,
            "max_pixels": CONTRACT.framepack_max_pixels,
            "audio": "full_aac64k_v2",
        },
    )
    with operation_lock(out):
        frame_root = out / "frames"
        cached_paths = [
            frame_root
            / frame_filename(
                index,
                timestamp,
                max_pixels=CONTRACT.framepack_max_pixels,
                resize_after_decode=True,
            )
            for index, timestamp in enumerate(timestamps, start=1)
        ]
        if all(path.is_file() and path.stat().st_size > 0 for path in cached_paths):
            paths = cached_paths
        else:
            paths = extract_frames(
                task.video_path,
                timestamps=timestamps,
                output_dir=frame_root,
                max_pixels=CONTRACT.framepack_max_pixels,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                resize_after_decode=True,
            )
        audio = None
        if has_audio and duration > 0:
            try:
                path = extract_full_audio(
                    audio_source,
                    output_path=out / "full_audio_aac64k_v2.m4a",
                    ffmpeg=ffmpeg,
                )
                audio = MediaRef(kind="audio", local_path=str(path), mime_type="audio/mp4")
            except Exception as exc:
                raise RuntimeError(
                    "INITIAL_AUDIO_REQUIRED: source has audio but framepack audio preparation failed; "
                    "refusing a silent visual-only fallback"
                ) from exc
    frames = tuple(
        (timestamp, MediaRef(kind="image", local_path=str(path), mime_type="image/jpeg"))
        for timestamp, path in zip(timestamps, paths, strict=True)
    )
    summary = (
        f"Initial frame sampling: fps={CONTRACT.framepack_fps}; max_frames={CONTRACT.framepack_max_frames}; "
        f"selected_frames={len(frames)}; max_pixels={CONTRACT.framepack_max_pixels} (~224x224 area); "
        f"sampling=uniform-over-fps-candidates; audio={'full' if audio else 'not available'}."
    )
    size = sum(estimated_data_url_bytes(ref.local_path or "", mime_type=ref.mime_type) for _, ref in frames)
    if audio:
        size += estimated_data_url_bytes(audio.local_path or "", mime_type=audio.mime_type)
    if size > CONTRACT.framepack_request_bytes:
        raise RuntimeError(
            f"INITIAL_REQUEST_TOO_LARGE: estimated data-url media payload {size} bytes exceeds "
            f"local budget {CONTRACT.framepack_request_bytes} bytes"
        )
    return Framepack(summary=summary, frames=frames, audio=audio, audio_end=duration, has_audio=has_audio)


def build_initial_message(
    task: TaskInput,
    *,
    work_dir: Path,
    ffmpeg: str,
    ffprobe: str = "ffprobe",
) -> Message:
    pack = build_framepack(task, work_dir=work_dir, ffmpeg=ffmpeg, ffprobe=ffprobe)
    effective = replace(task, duration_seconds=pack.audio_end, has_audio=pack.has_audio)
    parts = [text_part(initial_text(effective, summary=pack.summary, audio_attached=pack.audio is not None))]
    for index, (timestamp, media) in enumerate(pack.frames, start=1):
        parts.append(text_part(f"Frame {index}: {format_timecode(timestamp)} ({timestamp:g} seconds)"))
        parts.append(media_part(media))
    if pack.audio is not None:
        parts.append(text_part(f"Audio: 00:00.0-{format_timecode(pack.audio_end)} (continuous full audio)"))
        parts.append(media_part(pack.audio))
    else:
        parts.append(text_part("Audio: not available"))
    return message("user", *parts)
