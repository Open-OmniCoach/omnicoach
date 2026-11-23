"""Image/audio probing, caching, and pack-length enrichment for Agent rows."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
import wave
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

from omnicoach_sft.data.schema import (
    FRAME_TIMESTAMP_RE,
    PACK_AUDIO_SAMPLE_RATE,
    PACK_IMAGE_MAX_PIXELS,
    PACK_IMAGE_MIN_PIXELS,
    PACK_IMAGE_QUALITY,
    PACK_IMAGE_RESIZE_FACTOR,
    PACK_IMAGE_WEBP_METHOD,
    validate_agent_row,
)
from omnicoach_sft.io import atomic_write_bytes, read_mapping_json, write_json
from omnicoach_sft.media.image_tokens import (
    qwen_image_token_contribution as _qwen_image_token_contribution,
    smart_resize,
    webp_dimensions as _webp_dimensions,
)

_IMAGE_PROBE_ERRORS = (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError)


def _source_cache_key(path: Path, *, profile: str) -> str:
    stat = path.stat()
    value = json.dumps(
        {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "profile": profile,
        },
        sort_keys=True,
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _smart_resize(width: int, height: int) -> tuple[int, int]:
    return smart_resize(
        width=width,
        height=height,
        factor=PACK_IMAGE_RESIZE_FACTOR,
        min_pixels=PACK_IMAGE_MIN_PIXELS,
        max_pixels=PACK_IMAGE_MAX_PIXELS,
    )


def _probe_image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            width, height = image.size
        if width > 0 and height > 0:
            return int(width), int(height)
    except (OSError, ValueError):
        pass
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(probe.stdout).get("streams") or []
    if not streams:
        raise ValueError(f"ffprobe found no image stream: {path}")
    return int(streams[0]["width"]), int(streams[0]["height"])


def _path_image_token_contribution(path: str) -> int:
    image_path = Path(path)
    if image_path.suffix.lower() == ".webp":
        with image_path.open("rb") as handle:
            width, height = _webp_dimensions(handle.read(30), context=str(image_path))
    else:
        width, height = _probe_image_size(image_path)
    if width * height > PACK_IMAGE_MAX_PIXELS:
        raise ValueError(
            f"image exceeds declared image_max_pixels={PACK_IMAGE_MAX_PIXELS}: {image_path} is {width}x{height}"
        )
    return _qwen_image_token_contribution(width, height)


def probe_image_token_counts(
    rows: list[dict[str, Any]],
    *,
    num_proc: int,
    invalid_report_path: Path | None = None,
) -> dict[str, int]:
    paths = sorted({str(path) for row in rows for path in row.get("images") or []})
    if not paths:
        return {}

    def probe(path: str) -> tuple[int | None, dict[str, str] | None]:
        try:
            return _path_image_token_contribution(path), None
        except _IMAGE_PROBE_ERRORS as exc:
            return None, {
                "path": path,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    with ThreadPoolExecutor(max_workers=min(max(1, num_proc), len(paths))) as executor:
        results = list(executor.map(probe, paths))
    failures = [failure for _, failure in results if failure is not None]
    if invalid_report_path is not None:
        write_json(
            invalid_report_path,
            {
                "ok": not failures,
                "probed_unique_images": len(paths),
                "invalid_images": failures,
                "invalid_image_count": len(failures),
            },
            parents=True,
        )
    if failures:
        examples = "\n".join(
            f"{failure['path']}: {failure['error_type']}: {failure['error']}" for failure in failures[:100]
        )
        suffix = "" if len(failures) <= 100 else f"\n... and {len(failures) - 100} more"
        raise ValueError(f"{len(failures)} invalid image path(s):\n{examples}{suffix}")
    return {path: int(count) for path, (count, _) in zip(paths, results, strict=True) if count is not None}


def embedded_image_num_tokens(images: list[dict[str, Any]], *, context: str) -> int:
    total = 0
    for image_index, image in enumerate(images):
        payload = image.get("bytes") if isinstance(image, dict) else None
        if not isinstance(payload, bytes):
            raise ValueError(f"{context}.images[{image_index}] has no embedded bytes")
        width, height = _webp_dimensions(payload[:30], context=f"{context}.images[{image_index}]")
        total += _qwen_image_token_contribution(width, height)
    return total


def cache_base_webp(source: str, *, cache_root: Path) -> str:
    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    key = _source_cache_key(
        source_path,
        profile=(
            f"webp_q{PACK_IMAGE_QUALITY}_m{PACK_IMAGE_WEBP_METHOD}_f{PACK_IMAGE_RESIZE_FACTOR}_"
            f"min{PACK_IMAGE_MIN_PIXELS}_max{PACK_IMAGE_MAX_PIXELS}_compatible_passthrough_v1"
        ),
    )
    key_path = cache_root / "image_keys" / f"{key}.json"
    if key_path.is_file():
        cached = cache_root / "images" / str(read_mapping_json(key_path)["file"])
        if cached.is_file() and cached.stat().st_size > 0:
            return str(cached)

    # Avoid a second lossy encode when an upstream pipeline has already
    # produced a contract-compatible WebP. The payload is still content-addressed
    # below and strict Parquet validation rechecks it.
    payload: bytes | None = None
    if source_path.suffix.casefold() == ".webp":
        candidate = source_path.read_bytes()
        try:
            source_width, source_height = _webp_dimensions(candidate, context=str(source_path))
        except ValueError:
            pass
        else:
            pixels = source_width * source_height
            if (
                source_width % PACK_IMAGE_RESIZE_FACTOR == 0
                and source_height % PACK_IMAGE_RESIZE_FACTOR == 0
                and PACK_IMAGE_MIN_PIXELS <= pixels <= PACK_IMAGE_MAX_PIXELS
            ):
                payload = candidate

    if payload is not None:
        output = cache_root / "images" / f"{hashlib.sha256(payload).hexdigest()}.webp"
        atomic_write_bytes(output, payload, parents=True, skip_if_size_matches=True)
        atomic_write_bytes(key_path, json.dumps({"file": output.name}).encode("utf-8"), parents=True, skip_if_size_matches=True)
        return str(output)

    source_width, source_height = _probe_image_size(source_path)
    width, height = _smart_resize(source_width, source_height)
    temporary = cache_root / "image_work" / f"{key}.{os.getpid()}.{uuid.uuid4().hex}.webp"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source_path),
        "-vf",
        f"scale={width}:{height}:flags=bicubic",
        "-frames:v",
        "1",
        "-c:v",
        "libwebp",
        "-threads",
        "1",
        "-quality",
        str(PACK_IMAGE_QUALITY),
        "-compression_level",
        str(PACK_IMAGE_WEBP_METHOD),
        str(temporary),
    ]
    try:
        try:
            with Image.open(source_path) as image:
                image.convert("RGB").resize(
                    (width, height),
                    Image.Resampling.BICUBIC,
                ).save(
                    temporary,
                    format="WEBP",
                    quality=PACK_IMAGE_QUALITY,
                    method=PACK_IMAGE_WEBP_METHOD,
                )
        except (OSError, ValueError):
            subprocess.run(command, check=True, capture_output=True)
        payload = temporary.read_bytes()
    finally:
        temporary.unlink(missing_ok=True)
    _webp_dimensions(payload, context=str(source_path))
    output = cache_root / "images" / f"{hashlib.sha256(payload).hexdigest()}.webp"
    atomic_write_bytes(output, payload, parents=True, skip_if_size_matches=True)
    atomic_write_bytes(key_path, json.dumps({"file": output.name}).encode("utf-8"), parents=True, skip_if_size_matches=True)
    return str(output)


def cache_base_wav(source: str, *, cache_root: Path) -> str:
    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    key = _source_cache_key(source_path, profile=f"wav_pcm_s16le_{PACK_AUDIO_SAMPLE_RATE}hz_mono")
    output = cache_root / "audios" / f"{key}.wav"
    if output.is_file() and output.stat().st_size > 0:
        return str(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.wav")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ar",
        str(PACK_AUDIO_SAMPLE_RATE),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "-threads",
        "1",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        if not temporary.is_file() or temporary.stat().st_size <= 44:
            raise ValueError(f"ffmpeg produced an empty WAV: {source_path}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return str(output)


def _bounded_parallel_map(
    paths: list[str],
    worker: Callable[[str], str],
    *,
    num_proc: int,
    label: str,
) -> dict[str, str]:
    mapped: dict[str, str] = {}
    batch_size = max(64, min(2048, max(1, num_proc) * 32))
    with ThreadPoolExecutor(max_workers=max(1, num_proc)) as executor:
        for start in range(0, len(paths), batch_size):
            batch = paths[start : start + batch_size]
            results = executor.map(worker, batch)
            mapped.update(zip(batch, results, strict=True))
            done = min(start + len(batch), len(paths))
            print(f"{label}: {done}/{len(paths)}", flush=True)
    return mapped


def prepare_embedded_media_paths(
    rows: list[dict[str, Any]],
    *,
    cache_root: Path,
    num_proc: int,
    direct_compatible_webp: bool = False,
) -> list[dict[str, Any]]:
    image_paths = sorted({str(path) for row in rows for path in row.get("images") or []})
    audio_paths = sorted({str(path) for row in rows for path in row.get("audios") or []})
    direct_image_paths = (
        [path for path in image_paths if Path(path).suffix.casefold() == ".webp"] if direct_compatible_webp else []
    )
    direct_image_path_set = set(direct_image_paths)
    converted_image_paths = [path for path in image_paths if path not in direct_image_path_set]
    print(
        f"preparing Base-compatible embedded media: unique_images={len(image_paths)} "
        f"direct_webp={len(direct_image_paths)} converted_images={len(converted_image_paths)} "
        f"unique_audios={len(audio_paths)}",
        flush=True,
    )
    image_map = {path: path for path in direct_image_paths}
    image_map.update(
        _bounded_parallel_map(
            converted_image_paths,
            lambda path: cache_base_webp(path, cache_root=cache_root),
            num_proc=num_proc,
            label="webp",
        )
    )
    audio_map = _bounded_parallel_map(
        audio_paths,
        lambda path: cache_base_wav(path, cache_root=cache_root),
        num_proc=num_proc,
        label="wav",
    )
    prepared: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        row["images"] = [image_map[str(path)] for path in row.get("images") or []]
        row["audios"] = [audio_map[str(path)] for path in row.get("audios") or []]
        prepared.append(row)
    return prepared


@lru_cache(maxsize=4096)
def audio_num_samples_16khz(path: str) -> int:
    try:
        with wave.open(path, "rb") as audio:
            frames = audio.getnframes()
            sample_rate = audio.getframerate()
    except (EOFError, wave.Error):
        try:
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=duration",
                    "-of",
                    "json",
                    path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            streams = json.loads(probe.stdout).get("streams") or []
            duration = float(streams[0]["duration"])
        except (
            FileNotFoundError,
            KeyError,
            IndexError,
            ValueError,
            json.JSONDecodeError,
            subprocess.CalledProcessError,
        ) as exc:
            raise ValueError(f"ffprobe could not determine audio duration: {path}") from exc
        samples = round(duration * PACK_AUDIO_SAMPLE_RATE)
        if samples <= 0:
            raise ValueError(f"audio has no positive resampled samples: {path}")
        return samples
    if frames <= 0 or sample_rate <= 0:
        raise ValueError(f"invalid audio metadata: {path}")
    samples = round(frames * PACK_AUDIO_SAMPLE_RATE / sample_rate)
    if samples <= 0:
        raise ValueError(f"audio has no positive resampled samples: {path}")
    return samples


def probe_audio_sample_counts(rows: list[dict[str, Any]], *, num_proc: int) -> dict[str, int]:
    paths = sorted({str(path) for row in rows for path in row.get("audios") or []})
    with ThreadPoolExecutor(max_workers=min(max(1, num_proc), max(1, len(paths)))) as executor:
        counts = list(executor.map(audio_num_samples_16khz, paths))
    return dict(zip(paths, counts, strict=True))


def add_pack_length_stats(
    rows: list[dict[str, Any]],
    *,
    source_id: str,
    audio_sample_counts: dict[str, int] | None = None,
    image_token_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    audio_sample_counts = audio_sample_counts or {}
    image_token_counts = image_token_counts or {}
    enriched: list[dict[str, Any]] = []
    for row_index, raw_row in enumerate(rows):
        row = dict(raw_row)
        messages = row.get("messages") or []
        timestamps = [
            float(timestamp)
            for message in messages
            for timestamp in FRAME_TIMESTAMP_RE.findall(str(message.get("content") or ""))
        ]
        audios = row.get("audios") or []
        row.update(
            {
                "channel": "agent",
                "omni_source_id": source_id,
                "omni_num_images": len(row.get("images") or []),
                "omni_image_num_tokens": sum(
                    image_token_counts.get(str(path)) or _path_image_token_contribution(str(path))
                    for path in row.get("images") or []
                ),
                "omni_audio_num_samples": [
                    audio_sample_counts.get(str(path)) or audio_num_samples_16khz(str(path)) for path in audios
                ],
                "omni_last_frame_timestamp_ms": round(max(timestamps, default=0.0) * 1000),
            }
        )
        validate_agent_row(row, context=f"source row {row_index}", embedded_media=False)
        enriched.append(row)
    return enriched
