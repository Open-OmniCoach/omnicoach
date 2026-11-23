"""Frame, clip and audio extraction via FFmpeg and PyAV, plus the resize helpers."""

from __future__ import annotations

import logging
import math
import os
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .executor import offload_to_media_pool
from .ffmpeg import require_binary, run
from .probe import probe_video

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class FrameBatch:
    paths: tuple[Path, ...]
    decoder: str
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    source_fps: float
    frame_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class _PyAVPlan:
    source_fps: float
    source_size: tuple[int, int]
    start_time: int
    time_base: float
    indices: tuple[int, ...]
    pending: tuple[int, ...]
    positions: dict[int, list[int]]


def fit_max_side(width: int, height: int, *, max_side: int) -> tuple[int, int]:
    if max_side < 1:
        raise ValueError("max_side must be positive")
    if width <= max_side and height <= max_side:
        return width, height
    scale = min(max_side / float(width), max_side / float(height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def fit_max_pixels(width: int, height: int, *, max_pixels: int) -> tuple[int, int]:
    if max_pixels < 1:
        raise ValueError("max_pixels must be positive")
    pixels = width * height
    if pixels <= max_pixels:
        return width, height
    scale = math.sqrt(max_pixels / float(pixels))
    return max(1, int(width * scale)), max(1, int(height * scale))


def frame_filename(
    index: int,
    timestamp: float,
    *,
    max_side: int | None = None,
    max_pixels: int | None = None,
    quality: int = 90,
    resize_after_decode: bool = False,
) -> str:
    if (max_side is None) == (max_pixels is None):
        raise ValueError("configure exactly one of max_side or max_pixels")
    spec = f"s{max_side}" if max_side is not None else f"p{max_pixels}"
    mode = "post" if resize_after_decode else "dec"
    return f"frame_{index:04d}_{timestamp:.3f}_{spec}_q{quality}_{mode}.jpg"


def resize_image(
    source: str | Path,
    target: str | Path,
    *,
    max_side: int | None = None,
    max_pixels: int | None = None,
    quality: int = 90,
) -> Path:
    if (max_side is None) == (max_pixels is None):
        raise ValueError("configure exactly one of max_side or max_pixels")
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        size = (
            fit_max_side(image.width, image.height, max_side=int(max_side))
            if max_side is not None
            else fit_max_pixels(image.width, image.height, max_pixels=int(max_pixels))
        )
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        image.save(target_path, format="JPEG", quality=quality)
    return target_path


@offload_to_media_pool
def extract_frame(
    video_path: str,
    *,
    timestamp: float,
    output_path: str | Path,
    max_side: int | None = None,
    max_pixels: int | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    timeout_seconds: float = 120,
) -> Path:
    del ffprobe
    target = Path(output_path)
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.stem}.{os.getpid()}.{threading.get_ident()}.source.jpg")
    errors: list[str] = []
    for offset in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        seek = max(0.0, timestamp - offset)
        temp.unlink(missing_ok=True)
        try:
            run(
                [
                    require_binary(ffmpeg),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-y",
                    "-ss",
                    f"{seek:.3f}",
                    "-i",
                    video_path,
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(temp),
                ],
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:  # an end-of-container timestamp may need backoff
            errors.append(str(exc))
        if temp.is_file() and temp.stat().st_size > 0:
            resize_image(temp, target, max_side=max_side, max_pixels=max_pixels)
            temp.unlink(missing_ok=True)
            return target
    raise RuntimeError(f"failed to extract frame near {timestamp:.3f}s: {' | '.join(errors[-2:])}")


def extract_frames(
    video_path: str,
    *,
    timestamps: Iterable[float],
    output_dir: str | Path,
    max_side: int | None = None,
    max_pixels: int | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    resize_after_decode: bool = False,
) -> list[Path]:
    return list(
        extract_frame_batch(
            video_path,
            timestamps=timestamps,
            output_dir=output_dir,
            max_side=max_side,
            max_pixels=max_pixels,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            resize_after_decode=resize_after_decode,
        ).paths
    )


def _plan_pyav_stream(stream: Any, timestamps: list[float]) -> _PyAVPlan:
    rate = next(
        (
            value
            for value in (
                getattr(stream, "average_rate", None),
                getattr(stream, "guessed_rate", None),
                getattr(stream, "base_rate", None),
            )
            if value is not None and float(value) > 0
        ),
        None,
    )
    if rate is None:
        raise RuntimeError("invalid source fps")
    source_fps = float(rate)
    source_width = int(stream.codec_context.width)
    source_height = int(stream.codec_context.height)
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError("invalid source dimensions")
    frame_count = int(stream.frames or 0)
    if frame_count <= 0 and stream.duration is not None:
        frame_count = max(1, round(float(stream.duration * stream.time_base) * source_fps))
    indices = [max(0, round(value * source_fps)) for value in timestamps]
    if frame_count > 0:
        indices = [min(frame_count - 1, index) for index in indices]
    pending = tuple(sorted(set(indices)))
    if not pending:
        raise RuntimeError("no frame indices requested")
    time_base = float(stream.time_base)
    if time_base <= 0:
        raise RuntimeError("invalid video time base")
    positions: dict[int, list[int]] = {}
    for position, index in enumerate(indices):
        positions.setdefault(index, []).append(position)
    return _PyAVPlan(
        source_fps=source_fps,
        source_size=(source_width, source_height),
        start_time=int(stream.start_time or 0),
        time_base=time_base,
        indices=tuple(indices),
        pending=pending,
        positions=positions,
    )


def _save_pyav_frame(
    frame: Any,
    *,
    decode_size: tuple[int, int],
    output_size: tuple[int, int],
    resize_after_decode: bool,
    target_paths: list[Path],
    quality: int,
) -> list[tuple[Path, Path]]:
    uncached = [path for path in target_paths if not (path.is_file() and path.stat().st_size > 0)]
    if not uncached:
        return []
    image = frame.to_image(width=decode_size[0], height=decode_size[1]).convert("RGB")
    if resize_after_decode and image.size != output_size:
        image = image.resize(output_size, Image.Resampling.LANCZOS)
    prepared: list[tuple[Path, Path]] = []
    try:
        for path in uncached:
            temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp.jpg")
            image.save(temp, format="JPEG", quality=quality)
            prepared.append((temp, path))
    except Exception:
        for temp, _ in prepared:
            temp.unlink(missing_ok=True)
        temp.unlink(missing_ok=True)
        raise
    return prepared


def _decode_frames_with_pyav(
    video_path: str,
    *,
    timestamps: list[float],
    output_size: tuple[int, int],
    resize_after_decode: bool,
    decode_threads: int,
    paths: tuple[Path, ...],
    quality: int,
) -> tuple[float, tuple[int, ...]]:
    """Decode source-frame indices with PyAV after Decord fails."""

    import av

    with av.open(video_path, mode="r") as container:
        if not container.streams.video:
            raise RuntimeError("video has no video stream")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = decode_threads
        plan = _plan_pyav_stream(stream, timestamps)
        seek_index = max(0, plan.pending[0] - 1)
        seek_pts = plan.start_time + round((seek_index / plan.source_fps) / plan.time_base)
        container.seek(seek_pts, stream=stream, backward=True, any_frame=False)

        prepared: list[tuple[Path, Path]] = []
        previous: tuple[int, Any] | None = None
        cursor = 0
        decode_size = plan.source_size if resize_after_decode else output_size
        try:
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                frame_index = round((int(frame.pts) - plan.start_time) * plan.time_base * plan.source_fps)
                while cursor < len(plan.pending) and plan.pending[cursor] <= frame_index:
                    target = plan.pending[cursor]
                    chosen = (
                        previous[1]
                        if previous is not None and abs(previous[0] - target) <= abs(frame_index - target)
                        else frame
                    )
                    prepared.extend(
                        _save_pyav_frame(
                            chosen,
                            decode_size=decode_size,
                            output_size=output_size,
                            resize_after_decode=resize_after_decode,
                            target_paths=[paths[pos] for pos in plan.positions[target]],
                            quality=quality,
                        )
                    )
                    cursor += 1
                previous = (frame_index, frame)
                if cursor >= len(plan.pending):
                    break
            if cursor < len(plan.pending):
                raise RuntimeError(f"could not decode requested source frame indices {plan.pending[cursor:][:8]}")
            for temp, path in prepared:
                os.replace(temp, path)
        except Exception:
            for temp, _ in prepared:
                temp.unlink(missing_ok=True)
            raise
        return plan.source_fps, plan.indices


def _decode_frames_with_decord(
    video_path: str,
    *,
    values: list[float],
    source_fps_hint: float,
    output_size: tuple[int, int],
    resize_after_decode: bool,
    decode_threads: int,
    batch_size: int,
    paths: tuple[Path, ...],
    quality: int,
) -> tuple[float, tuple[int, ...]]:
    from decord import VideoReader, cpu

    reader_kwargs: dict[str, object] = {"ctx": cpu(0), "num_threads": decode_threads}
    if not resize_after_decode:
        reader_kwargs.update(width=output_size[0], height=output_size[1])
    reader = VideoReader(video_path, **reader_kwargs)
    frame_count = len(reader)
    source_fps = float(reader.get_avg_fps()) or source_fps_hint
    if frame_count < 1 or source_fps <= 0:
        raise RuntimeError("decoder reported no frames or invalid FPS")
    frame_indices = tuple(min(frame_count - 1, max(0, round(value * source_fps))) for value in values)
    missing = [index for index, path in enumerate(paths) if not path.is_file() or path.stat().st_size <= 0]
    for offset in range(0, len(missing), batch_size):
        positions = missing[offset : offset + batch_size]
        arrays = reader.get_batch([frame_indices[index] for index in positions]).asnumpy()
        for position, array in zip(positions, arrays, strict=True):
            target = paths[position]
            temp = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp.jpg")
            temp.unlink(missing_ok=True)
            image = Image.fromarray(array).convert("RGB")
            if resize_after_decode and image.size != output_size:
                image = image.resize(output_size, Image.Resampling.LANCZOS)
            image.save(temp, format="JPEG", quality=quality)
            os.replace(temp, target)
    return source_fps, frame_indices


@offload_to_media_pool
def extract_frame_batch(
    video_path: str,
    *,
    timestamps: Iterable[float],
    output_dir: str | Path,
    max_side: int | None = None,
    max_pixels: int | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    decode_threads: int = 2,
    batch_size: int = 16,
    quality: int = 90,
    resize_after_decode: bool = False,
) -> FrameBatch:
    """Decode requested frames in batches, with PyAV and per-frame FFmpeg fallbacks.

    Decord avoids one process launch per sampled frame. All choices that affect
    output remain explicit; importing Decord lazily keeps the fallbacks usable in
    constrained environments.
    """

    if (max_side is None) == (max_pixels is None):
        raise ValueError("configure exactly one of max_side or max_pixels")
    if decode_threads < 1 or batch_size < 1:
        raise ValueError("decode_threads and batch_size must be positive")

    root = Path(output_dir)
    values = list(timestamps)
    if not values:
        raise ValueError("timestamps must not be empty")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("timestamps must be finite non-negative seconds")
    root.mkdir(parents=True, exist_ok=True)
    paths = tuple(
        root
        / frame_filename(
            index,
            value,
            max_side=max_side,
            max_pixels=max_pixels,
            quality=quality,
            resize_after_decode=resize_after_decode,
        )
        for index, value in enumerate(values, start=1)
    )
    source = probe_video(video_path, ffprobe=ffprobe)
    source_size = (source.width, source.height)
    output_size = (
        fit_max_side(source.width, source.height, max_side=int(max_side))
        if max_side is not None
        else fit_max_pixels(source.width, source.height, max_pixels=int(max_pixels))
    )

    try:
        source_fps, frame_indices = _decode_frames_with_decord(
            video_path,
            values=values,
            source_fps_hint=source.fps,
            output_size=output_size,
            resize_after_decode=resize_after_decode,
            decode_threads=decode_threads,
            batch_size=batch_size,
            paths=paths,
            quality=quality,
        )
        decoder = "decord"
    except Exception as decord_exc:
        logger.warning("decord frame decode failed for %s (%s); falling back to PyAV", video_path, decord_exc)
        try:
            source_fps, frame_indices = _decode_frames_with_pyav(
                video_path,
                timestamps=values,
                output_size=output_size,
                resize_after_decode=resize_after_decode,
                decode_threads=decode_threads,
                paths=paths,
                quality=quality,
            )
            decoder = "pyav_fallback"
        except Exception as pyav_exc:
            logger.warning(
                "PyAV frame decode failed for %s (%s); falling back to per-frame FFmpeg",
                video_path,
                pyav_exc,
            )
            source_fps = source.fps
            frame_indices = tuple(max(0, round(value * source_fps)) for value in values)
            for value, path in zip(values, paths, strict=True):
                extract_frame(
                    video_path,
                    timestamp=value,
                    output_path=path,
                    max_side=max_side,
                    max_pixels=max_pixels,
                    ffmpeg=ffmpeg,
                    ffprobe=ffprobe,
                )
            decoder = "ffmpeg_fallback"
    return FrameBatch(
        paths=paths,
        decoder=decoder,
        source_size=source_size,
        output_size=output_size,
        source_fps=source_fps,
        frame_indices=frame_indices,
    )


def _extract_audio_ffmpeg(
    video_path: str,
    *,
    output_path: str | Path,
    ffmpeg: str,
    bitrate_kbps: int,
    timeout_seconds: float,
    start: float | None = None,
    end: float | None = None,
    label: str = "audio",
) -> Path:
    target = Path(output_path)
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp.m4a")
    temp.unlink(missing_ok=True)
    command = [
        require_binary(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
    ]
    if start is not None:
        command.extend(["-ss", f"{start:.3f}"])
    command.extend(["-i", video_path])
    if start is not None and end is not None:
        command.extend(["-t", f"{end - start:.3f}"])
    command.extend(
        [
            "-map",
            "0:a:0?",
            "-vn",
            "-strict",
            "-2",
            "-c:a",
            "aac",
            "-b:a",
            f"{bitrate_kbps}k",
            "-movflags",
            "+faststart",
            str(temp),
        ]
    )
    try:
        run(command, timeout_seconds=timeout_seconds)
        if not temp.is_file() or temp.stat().st_size <= 0:
            raise RuntimeError(f"{label} extraction produced no output")
        os.replace(temp, target)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return target


@offload_to_media_pool
def extract_audio(
    video_path: str,
    *,
    start: float,
    end: float,
    output_path: str | Path,
    ffmpeg: str = "ffmpeg",
    bitrate_kbps: int = 64,
    timeout_seconds: float = 300,
) -> Path:
    if end <= start:
        raise ValueError("audio end must exceed start")
    return _extract_audio_ffmpeg(
        video_path,
        output_path=output_path,
        ffmpeg=ffmpeg,
        bitrate_kbps=bitrate_kbps,
        timeout_seconds=timeout_seconds,
        start=start,
        end=end,
        label="audio",
    )


@offload_to_media_pool
def extract_full_audio(
    video_path: str,
    *,
    output_path: str | Path,
    ffmpeg: str = "ffmpeg",
    bitrate_kbps: int = 64,
    timeout_seconds: float = 300,
) -> Path:
    """Extract the complete source audio stream for the initial framepack."""

    return _extract_audio_ffmpeg(
        video_path,
        output_path=output_path,
        ffmpeg=ffmpeg,
        bitrate_kbps=bitrate_kbps,
        timeout_seconds=timeout_seconds,
        label="full audio",
    )
