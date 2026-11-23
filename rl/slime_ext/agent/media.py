"""Initial media & on-demand video/audio slicing with budget tracking.

Key guarantees:
- Short-video (<1s) safe initial sampling, range indexing, and audio slicing
  enforcing ``0 <= start < end <= duration``.
- Per-episode ``MediaBudget`` tracking (frames, audio seconds, high-res calls,
  normalized cost units).
- Neutral media observation metadata on initial and tool execution bundles
  (``tool_name``, ``media_kind``, ``observed_spans``, ``frame_timestamps``,
  ``duration_sec``, ``params``) persisted cleanly in ``InitialMediaState``
  across partial rollout pauses/resumes.
- Supports both Parquet+Contract production media and lightweight in-memory
  synthetic media sources for deterministic CPU testing.
"""

from __future__ import annotations

import base64
import bisect
import io
import json
import math
import tempfile
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import numpy as np
from PIL import Image

from .intervals import merge_intervals
from .protocol import (
    DEFAULT_MEDIA_SETTINGS,
    VISUAL_HEADERS,
    MediaBundle,
    ProtocolError,
    ToolRequest,
    natural_seconds,
    tool_max_seconds,
)


@dataclass(frozen=True)
class Sampling:
    fps: float
    max_frames: int


@dataclass(frozen=True)
class Resolution:
    max_pixels: int
    max_side: int | None = None
    from_source: bool = False


@dataclass
class MediaBudget:
    """Tracks media consumption across turns of an episode."""

    frames_used: int = 0
    audio_seconds_used: float = 0.0
    high_res_calls: int = 0
    tool_calls: int = 0
    cumulative_cost_units: float = 0.0
    frame_budget_ref: float = 64.0
    audio_budget_ref: float = 64.0
    high_res_surcharge: float = 0.25

    def record(self, request: ToolRequest, bundle: MediaBundle) -> float:
        high_calls = (
            sum(1 for s in request.spans if s.resolution == "high")
            if request.spans
            else int(request.resolution == "high")
        )
        cost = (
            float(bundle.frame_count) / max(1.0, self.frame_budget_ref)
            + float(bundle.audio_duration_s) / max(1.0, self.audio_budget_ref)
            + float(high_calls) * self.high_res_surcharge
        )
        self.frames_used += int(bundle.frame_count)
        self.audio_seconds_used += float(bundle.audio_duration_s)
        self.high_res_calls += high_calls
        self.tool_calls += 1
        self.cumulative_cost_units += cost
        return cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_used": self.frames_used,
            "audio_seconds_used": self.audio_seconds_used,
            "high_res_calls": self.high_res_calls,
            "tool_calls": self.tool_calls,
            "cumulative_cost_units": self.cumulative_cost_units,
            "frame_budget_ref": self.frame_budget_ref,
            "audio_budget_ref": self.audio_budget_ref,
            "high_res_surcharge": self.high_res_surcharge,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MediaBudget":
        return cls(
            frames_used=int(data.get("frames_used", 0)),
            audio_seconds_used=float(data.get("audio_seconds_used", 0.0)),
            high_res_calls=int(data.get("high_res_calls", 0)),
            tool_calls=int(data.get("tool_calls", 0)),
            cumulative_cost_units=float(data.get("cumulative_cost_units", 0.0)),
            frame_budget_ref=float(data.get("frame_budget_ref", 64.0)),
            audio_budget_ref=float(data.get("audio_budget_ref", 64.0)),
            high_res_surcharge=float(data.get("high_res_surcharge", 0.25)),
        )


def _normalize_observation(obs: Mapping[str, Any]) -> dict[str, Any]:
    """Return a clean copy of a neutral media observation with tuple spans."""
    raw_spans = obs.get("observed_spans", ())
    return {
        "tool_name": str(obs["tool_name"]),
        "media_kind": str(obs["media_kind"]),
        "observed_spans": [(float(a), float(b)) for a, b in raw_spans],
        "frame_timestamps": [float(t) for t in obs.get("frame_timestamps", ())],
        "duration_sec": float(obs["duration_sec"]),
        "params": dict(obs.get("params", {})),
    }


@dataclass
class InitialMediaState:
    """Serializable media state persisted across partial rollout resumes."""

    duration_s: float
    has_audio: bool
    initial_audio_attached: bool
    initial_observations: list[dict[str, Any]] = field(default_factory=list)
    covered_spans: dict[str, list[tuple[float, float]]] = field(
        default_factory=lambda: {"video": [], "audio": [], "av": []}
    )
    budget: dict[str, Any] = field(default_factory=lambda: MediaBudget().to_dict())

    def record_covered(self, observations: Sequence[Mapping[str, Any]]) -> None:
        for obs in observations:
            kind = str(obs["media_kind"])
            spans = [(float(a), float(b)) for a, b in obs.get("observed_spans", ())]
            if not spans:
                continue
            target_kinds = ("av", "video", "audio") if kind == "av" else (kind,)
            for target_kind in target_kinds:
                existing = list(self.covered_spans.get(target_kind, ()))
                existing.extend(spans)
                self.covered_spans[target_kind] = merge_intervals(existing)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_s": self.duration_s,
            "has_audio": self.has_audio,
            "initial_audio_attached": self.initial_audio_attached,
            "initial_observations": [_normalize_observation(obs) for obs in self.initial_observations],
            "covered_spans": {
                kind: [(float(a), float(b)) for a, b in spans]
                for kind, spans in self.covered_spans.items()
            },
            "budget": dict(self.budget),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InitialMediaState":
        raw_cov = data.get("covered_spans", {})
        return cls(
            duration_s=float(data["duration_s"]),
            has_audio=bool(data["has_audio"]),
            initial_audio_attached=bool(data["initial_audio_attached"]),
            initial_observations=[
                _normalize_observation(obs) for obs in data.get("initial_observations", ())
            ],
            covered_spans={
                kind: [(float(a), float(b)) for a, b in raw_cov.get(kind, ())]
                for kind in ("video", "audio", "av")
            },
            budget=dict(data.get("budget", MediaBudget().to_dict())),
        )


def _path(value: str) -> Path:
    return Path(unquote(urlsplit(value).path)) if value.startswith("file:") else Path(value)


def _cap(indices: list[int], count: int) -> list[int]:
    if len(indices) <= count:
        return indices
    if count <= 1:
        return indices[:1]
    positions = (round(i * (len(indices) - 1) / (count - 1)) for i in range(count))
    return [indices[position] for position in positions]


def _size(
    width: int,
    height: int,
    resolution: Resolution,
    *,
    factor: int,
    min_pixels: int,
) -> tuple[int, int]:
    max_pixels, max_side = resolution.max_pixels, resolution.max_side
    if max_side is not None and max(width, height) > max_side:
        scale = max_side / max(width, height)
        width, height = max(1, round(width * scale)), max(1, round(height * scale))
    rw = max(factor, round(width / factor) * factor)
    rh = max(factor, round(height / factor) * factor)
    if rw * rh > max_pixels:
        scale = math.sqrt(width * height / max_pixels)
        rw = max(factor, math.floor(math.nextafter(width / scale / factor, math.inf)) * factor)
        rh = max(factor, math.floor(math.nextafter(height / scale / factor, math.inf)) * factor)
    elif rw * rh < min_pixels:
        scale = math.sqrt(min_pixels / (width * height))
        rw = max(factor, math.ceil(math.nextafter(width / scale / factor, -math.inf)) * factor)
        rh = max(factor, math.ceil(math.nextafter(height / scale / factor, -math.inf)) * factor)
    while rw * rh > max_pixels:
        if rw >= rh and rw > factor:
            rw -= factor
        elif rh > factor:
            rh -= factor
        else:
            break
    return rw, rh


def audio_data(samples: np.ndarray, sample_rate: int) -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(np.rint(samples * 32768).clip(-32768, 32767).astype("<i2"))
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_initial_observations(
    *,
    duration_sec: float,
    frame_timestamps: Sequence[float],
    initial_settings: Mapping[str, Any],
    has_audio: bool,
    attach_audio: bool,
) -> list[dict[str, Any]]:
    """Build neutral media observation metadata for the initial overview bundle."""
    if duration_sec <= 0.0:
        return []
    ts_list = [float(t) for t in frame_timestamps]
    if not ts_list and not attach_audio:
        return []
    if ts_list and attach_audio:
        media_kind = "av"
    elif ts_list:
        media_kind = "video"
    else:
        media_kind = "audio"

    return [
        {
            "tool_name": "initial",
            "media_kind": media_kind,
            "observed_spans": [(0.0, float(duration_sec))],
            "frame_timestamps": ts_list,
            "duration_sec": float(duration_sec),
            "params": {
                "fps": float(initial_settings["fps"]),
                "max_frames": int(initial_settings["max_frames"]),
                "max_pixels": int(initial_settings["max_pixels"]),
                "has_audio": bool(has_audio),
                "initial_audio_attached": bool(attach_audio),
                "cost_units": 0.0,
            },
        }
    ]


def build_tool_observations(
    request: ToolRequest,
    bundle: MediaBundle,
    *,
    duration_sec: float,
    has_audio: bool,
    settings: Mapping[str, Any],
    turn_index: int | None = None,
) -> list[dict[str, Any]]:
    """Build neutral media observation records for an executed tool request."""
    spans = request.spans or (request,)
    per_span_cost = float(bundle.cost_units) / max(1, len(spans))
    observations: list[dict[str, Any]] = []
    for index, span in enumerate(spans):
        if span.end <= span.start:
            raise ProtocolError(
                "INVALID_MEDIA_RANGE",
                f"degenerate span [{span.start}, {span.end}] in tool observation",
            )
        if span.name == "get_audio":
            media_kind = "audio"
            frame_timestamps: list[float] = []
            params: dict[str, Any] = {
                "level": None,
                "resolution": None,
                "fps": None,
                "has_audio": True,
                "parent_tool": request.name,
                "span_index": index,
                "cost_units": per_span_cost,
                "turn_index": int(turn_index) if turn_index is not None else None,
            }
        else:
            media_kind = "av" if (span.name == "get_clip" and has_audio) else "video"
            fps = float(settings["fps"][span.level])
            frame_timestamps = [
                float(ts)
                for ts in bundle.frame_timestamps_s
                if span.start - 1e-3 <= ts <= span.end + 1e-3
            ]
            params = {
                "level": span.level,
                "resolution": span.resolution,
                "fps": fps,
                "has_audio": bool(has_audio and span.name == "get_clip"),
                "parent_tool": request.name,
                "span_index": index,
                "cost_units": per_span_cost,
                "turn_index": int(turn_index) if turn_index is not None else None,
            }

        observations.append(
            {
                "tool_name": span.name,
                "media_kind": media_kind,
                "observed_spans": [(float(span.start), float(span.end))],
                "frame_timestamps": frame_timestamps,
                "duration_sec": float(duration_sec),
                "params": params,
            }
        )
    return observations


def _make_wav_bytes(duration_s: float, sample_rate: int = 16000) -> bytes:
    n_samples = max(1, round(duration_s * sample_rate))
    samples = np.zeros(n_samples, dtype="<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples.tobytes())
    return buf.getvalue()


def _make_png_bytes(width: int = 32, height: int = 32) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(16, 32, 48)).save(buf, format="PNG")
    return buf.getvalue()


class Media:
    """One active trajectory media reader with short-video (<1s) guards and budget tracking."""

    def __init__(
        self,
        location: Mapping[str, Any] | None,
        settings: Mapping[str, Any],
        audio_window_samples: int = 400,
        *,
        in_memory_row: Mapping[str, Any] | None = None,
        in_memory_contract: Mapping[str, Any] | None = None,
    ) -> None:
        self.settings = settings
        self.audio_window_samples = int(audio_window_samples)
        if in_memory_row is not None:
            contract = dict(in_memory_contract or {})
            self.frame_contract = dict(
                contract.get(
                    "frames",
                    {
                        "sample_fps": 2.0,
                        "max_pixels": 200704,
                        "min_pixels": 3136,
                        "resize_factor": 28,
                        "format": "png",
                    },
                )
            )
            self.sample_rate = int(contract.get("audio", {}).get("sample_rate", 16000))
            self.row = dict(in_memory_row)
        else:
            if location is None:
                raise ValueError("location or in_memory_row is required")
            import pyarrow.parquet as pq

            contract = json.loads(Path(location["contract"]).read_text())
            self.frame_contract = contract["frames"]
            self.sample_rate = int(contract["audio"]["sample_rate"])
            with pq.ParquetFile(location["path"]) as parquet:
                self.row = parquet.read_row_group(location["row_group"]).to_pylist()[0]

        self.duration_ms = int(self.row["duration_ms"])
        if self.duration_ms <= 0:
            raise ValueError(f"media duration_ms must be > 0, got {self.duration_ms}")
        self.has_audio = self.row.get("audio_bytes") is not None
        self.timestamps = [int(t) for t in self.row["frame_timestamps_ms"]]
        extra_raw = self.row.get("extra_json")
        self.source_info = (
            json.loads(extra_raw)["source_info"]
            if isinstance(extra_raw, str)
            else dict(self.row.get("source_info", {"width": 64, "height": 64}))
        )
        self.budget = MediaBudget()
        self.temporary = None
        self.source_path = None

    @classmethod
    def synthetic(
        cls,
        *,
        duration_s: float,
        has_audio: bool = True,
        sample_fps: float = 2.0,
        settings: Mapping[str, Any] | None = None,
        sample_rate: int = 16000,
    ) -> "Media":
        """Create a self-contained in-memory Media instance (works for <1s short videos)."""
        if duration_s <= 0.0:
            raise ValueError("duration_s must be > 0")
        duration_ms = max(1, round(duration_s * 1000))
        step_ms = max(1, round(1000.0 / sample_fps))
        timestamps = list(range(0, duration_ms, step_ms))
        if not timestamps:
            timestamps = [0]
        png = _make_png_bytes()
        row = {
            "duration_ms": duration_ms,
            "frame_timestamps_ms": timestamps,
            "frame_bytes": [png for _ in timestamps],
            "audio_bytes": _make_wav_bytes(duration_s, sample_rate) if has_audio else None,
            "source_info": {"width": 32, "height": 32},
        }
        contract = {
            "frames": {
                "sample_fps": sample_fps,
                "max_pixels": 200704,
                "min_pixels": 784,
                "resize_factor": 28,
                "format": "png",
            },
            "audio": {"sample_rate": sample_rate},
        }
        return cls(
            None,
            settings or DEFAULT_MEDIA_SETTINGS,
            audio_window_samples=400,
            in_memory_row=row,
            in_memory_contract=contract,
        )

    def close(self) -> None:
        if self.temporary is not None:
            self.temporary.cleanup()
            self.temporary = None

    def _source(self) -> Path:
        if self.source_path is None:
            ref = self.source_info["video_reference"]
            if ref["kind"] == "file":
                self.source_path = _path(ref["path"])
            else:
                self.temporary = tempfile.TemporaryDirectory(prefix="omni-media-")
                self.source_path = Path(self.temporary.name) / "source.mp4"
                with _path(ref["archive_path"]).open("rb") as source, self.source_path.open("wb") as target:
                    source.seek(ref["offset_data"])
                    block_size = 8 * 1024**2
                    for offset in range(0, ref["size_bytes"], block_size):
                        target.write(source.read(min(block_size, ref["size_bytes"] - offset)))
        return self.source_path

    def _initial_indices(self) -> list[int]:
        if not self.timestamps:
            return []
        initial = self.settings["initial"]
        effective_margin = min(int(initial["end_margin_ms"]), max(0, self.duration_ms // 4))
        last_ms = max(0, self.duration_ms - effective_margin)
        interval_ms = max(1, round(1000.0 / float(initial["fps"])))
        indices: list[int] = []
        for target in range(0, last_ms + 1, interval_ms):
            pos = bisect.bisect_left(self.timestamps, target)
            if pos == len(self.timestamps):
                pos -= 1
            elif pos and target - self.timestamps[pos - 1] <= self.timestamps[pos] - target:
                pos -= 1
            indices.append(pos)
        return _cap(list(dict.fromkeys(indices)), int(initial["max_frames"]))

    def _range_indices(self, start_ms: int, end_ms: int, sampling: Sampling) -> list[int]:
        if start_ms < 0 or end_ms <= start_ms:
            raise ProtocolError(
                "INVALID_MEDIA_RANGE",
                f"invalid millisecond window [{start_ms}, {end_ms}]",
            )
        if not self.timestamps:
            raise ProtocolError("MEDIA_RANGE_UNAVAILABLE", "media contains no cached frames")
        cache_fps = float(self.frame_contract["sample_fps"])
        stride = max(1, round(cache_fps / float(sampling.fps)))
        cache_interval_ms = 1000.0 / cache_fps
        indices = [
            i
            for i, timestamp in enumerate(self.timestamps)
            if start_ms - 1 <= timestamp <= end_ms + 1
            and round(timestamp / cache_interval_ms) % stride == 0
        ]
        if not indices:
            in_bounds = [
                i for i, ts in enumerate(self.timestamps) if start_ms - 1 <= ts <= end_ms + 1
            ]
            if in_bounds:
                indices = in_bounds
            else:
                mid_ms = 0.5 * (start_ms + end_ms)
                nearest = min(range(len(self.timestamps)), key=lambda i: abs(self.timestamps[i] - mid_ms))
                if abs(self.timestamps[nearest] - mid_ms) <= max(cache_interval_ms, 500.0):
                    indices = [nearest]
        if not indices:
            raise ProtocolError(
                "MEDIA_RANGE_UNAVAILABLE",
                f"no cached frames overlap {start_ms}ms to {end_ms}ms",
            )
        count = max(1, math.ceil((end_ms - start_ms) * sampling.fps / 1000.0 - 1e-9))
        return _cap(indices, min(sampling.max_frames, count))

    def _frames(self, indices: list[int], resolution: Resolution) -> list[dict[str, Any]]:
        source_frames = None
        if resolution.from_source and "video_reference" in self.source_info:
            from decord import VideoReader, cpu

            max_side = resolution.max_side
            width, height = self.source_info["width"], self.source_info["height"]
            if max_side is not None and max(width, height) > max_side:
                scale = max_side / max(width, height)
                width, height = max(1, round(width * scale)), max(1, round(height * scale))
            reader = VideoReader(str(self._source()), ctx=cpu(0), num_threads=2, width=width, height=height)
            fps = reader.get_avg_fps()
            positions = [min(len(reader) - 1, round(self.timestamps[i] * fps / 1000.0)) for i in indices]
            source_frames = reader.get_batch(positions).asnumpy()

        parts: list[dict[str, Any]] = []
        for offset, index in enumerate(indices):
            if source_frames is None:
                encoded = self.row["frame_bytes"][index]
                with Image.open(io.BytesIO(encoded)) as source:
                    image = source.convert("RGB")
                mime = self.frame_contract["format"]
            else:
                image = Image.fromarray(source_frames[offset]).convert("RGB")
                encoded, mime = None, "png"
            size = _size(
                image.width,
                image.height,
                resolution,
                factor=int(self.frame_contract["resize_factor"]),
                min_pixels=int(self.frame_contract["min_pixels"]),
            )
            if image.size != size:
                image = image.resize(size, Image.Resampling.LANCZOS)
                encoded = None
            if encoded is None:
                buffer = io.BytesIO()
                image.save(buffer, format="PNG", compress_level=int(self.settings.get("png_compress_level", 1)))
                encoded, mime = buffer.getvalue(), "png"
            image.info["data_url"] = f"data:image/{mime};base64," + base64.b64encode(encoded).decode("ascii")
            parts.extend(
                [
                    {"type": "text", "text": f"<{self.timestamps[index] / 1000.0:.1f} seconds>"},
                    {"type": "image", "image": image, "max_pixels": resolution.max_pixels},
                ]
            )
        return parts

    def _audio(self, start_ms: int, end_ms: int) -> np.ndarray:
        if start_ms < 0 or end_ms <= start_ms:
            raise ProtocolError("INVALID_MEDIA_RANGE", "audio window requires 0 <= start_ms < end_ms")
        with wave.open(io.BytesIO(self.row["audio_bytes"]), "rb") as reader:
            rate, total = reader.getframerate(), reader.getnframes()
            start = min(total, max(0, round(start_ms * rate / 1000.0)))
            end = min(total, max(start, round(end_ms * rate / 1000.0)))
            reader.setpos(start)
            samples = np.frombuffer(reader.readframes(end - start), dtype="<i2").astype(np.float32) / 32768.0
        if len(samples) < self.audio_window_samples:
            samples = np.pad(samples, (0, self.audio_window_samples - len(samples)))
        return samples

    def initial(self) -> tuple[MediaBundle, bool, InitialMediaState]:
        settings = self.settings["initial"]
        duration = self.duration_ms / 1000.0
        attach_audio = self.has_audio and duration <= float(settings["audio_max_seconds"])
        indices = self._initial_indices()
        frame_ts_s = tuple(self.timestamps[i] / 1000.0 for i in indices)
        parts = self._frames(indices, Resolution(max_pixels=int(settings["max_pixels"])))
        audio_summary = "not available"
        if attach_audio:
            audio_summary = "full"
            parts.extend(
                [
                    {"type": "text", "text": f"Audio: 0 seconds-{duration:g} seconds (continuous full audio)"},
                    {"type": "audio", "audio": self._audio(0, self.duration_ms)},
                ]
            )
        elif self.has_audio:
            audio_summary = "available-via-get_audio (full audio omitted to preserve context budget)"
        header = (
            f"Initial frame sampling: fps={settings['fps']:g}; "
            f"max_frames={settings['max_frames']}; selected_frames={len(indices)}; "
            f"max_pixels={settings['max_pixels']} "
            f"(~{math.isqrt(int(settings['max_pixels']))}x{math.isqrt(int(settings['max_pixels']))} area); "
            f"sampling=uniform-over-fps-candidates; audio={audio_summary}."
        )
        init_obs = build_initial_observations(
            duration_sec=duration,
            frame_timestamps=frame_ts_s,
            initial_settings=settings,
            has_audio=self.has_audio,
            attach_audio=attach_audio,
        )
        state = InitialMediaState(
            duration_s=duration,
            has_audio=self.has_audio,
            initial_audio_attached=attach_audio,
            initial_observations=init_obs,
            budget=self.budget.to_dict(),
        )
        state.record_covered(init_obs)
        bundle = MediaBundle(
            kind="initial",
            header=header,
            parts=tuple(parts),
            frame_timestamps_s=frame_ts_s,
            audio_duration_s=duration if attach_audio else 0.0,
            frame_count=len(indices),
            cost_units=0.0,
            metadata={"observations": init_obs},
        )
        return bundle, attach_audio, state

    def _finalize_tool_bundle(
        self,
        request: ToolRequest,
        prelim: MediaBundle,
        *,
        record_budget: bool,
    ) -> MediaBundle:
        cost = self.budget.record(request, prelim) if record_budget else 0.0
        bundle = MediaBundle(
            kind=prelim.kind,
            header=prelim.header,
            parts=prelim.parts,
            frame_timestamps_s=prelim.frame_timestamps_s,
            audio_duration_s=prelim.audio_duration_s,
            frame_count=prelim.frame_count,
            cost_units=cost,
        )
        if record_budget:
            bundle.metadata["tool_observations"] = build_tool_observations(
                request,
                bundle,
                duration_sec=self.duration_ms / 1000.0,
                has_audio=self.has_audio,
                settings=self.settings,
            )
        return bundle

    def execute(self, request: ToolRequest, *, _record_budget: bool = True) -> MediaBundle:
        if request.name == "get_multi_span":
            return self._multi_span(request)
        duration_s = self.duration_ms / 1000.0
        if request.start < 0.0 or request.end <= request.start or request.start >= duration_s:
            raise ProtocolError(
                "INVALID_MEDIA_RANGE",
                f"invalid tool window [{request.start}, {request.end}] for duration {duration_s:.3f}s",
            )
        start_str, end_str = natural_seconds(request.start), natural_seconds(request.end)
        start_ms = max(0, min(self.duration_ms - 1, round(request.start * 1000.0)))
        end_ms = max(start_ms + 1, min(self.duration_ms, round(request.end * 1000.0)))
        span_dur = request.end - request.start

        if request.name == "get_audio":
            header = f"Audio {start_str} to {end_str} (continuous)."
            prelim = MediaBundle(
                kind="audio",
                header=header,
                parts=(
                    {"type": "text", "text": header},
                    {"type": "audio", "audio": self._audio(start_ms, end_ms)},
                ),
                audio_duration_s=span_dur,
                frame_count=0,
            )
            return self._finalize_tool_bundle(request, prelim, record_budget=_record_budget)

        fps = float(self.settings["fps"][request.level])
        max_seconds = tool_max_seconds(self.settings, request.name, request.level)
        sampling = Sampling(fps=fps, max_frames=max(1, math.ceil(max_seconds * fps)))
        view = self.settings["resolutions"][request.resolution]
        max_pixels = int(view.get("max_pixels", self.frame_contract["max_pixels"]))
        resolution = Resolution(
            max_pixels=max_pixels,
            max_side=math.isqrt(max_pixels),
            from_source=view["source"] == "original",
        )
        is_clip = request.name == "get_clip"
        indices = self._range_indices(start_ms, end_ms, sampling)
        frame_ts_s = tuple(self.timestamps[i] / 1000.0 for i in indices)
        frames = self._frames(indices, resolution)
        include_audio = is_clip and self.has_audio
        header = VISUAL_HEADERS[request.name, request.resolution].format(
            start=start_str,
            end=end_str,
            level=request.level,
            fps=sampling.fps,
            count=len(indices),
            max_side=resolution.max_side,
            audio="continuous same-span audio included" if include_audio else "audio unavailable",
        )
        parts: list[dict[str, Any]] = [{"type": "text", "text": header}, *frames]
        if include_audio:
            parts.extend(
                [
                    {"type": "text", "text": f"Audio span: {start_str} to {end_str}, continuous."},
                    {"type": "audio", "audio": self._audio(start_ms, end_ms)},
                ]
            )
        prelim = MediaBundle(
            kind="clip" if is_clip else "frames",
            header=header,
            parts=tuple(parts),
            frame_timestamps_s=frame_ts_s,
            audio_duration_s=span_dur if include_audio else 0.0,
            frame_count=len(indices),
        )
        return self._finalize_tool_bundle(request, prelim, record_budget=_record_budget)

    def _multi_span(self, request: ToolRequest) -> MediaBundle:
        summaries: list[str] = []
        parts: list[dict[str, Any]] = []
        audios: list[np.ndarray] = []
        all_ts: list[float] = []
        total_frames = 0
        total_audio_s = 0.0
        for index, span in enumerate(request.spans, 1):
            bundle = self.execute(span, _record_budget=False)
            modality = {"get_frames": "frames", "get_audio": "audio", "get_clip": "clip"}[span.name]
            summary = f"span_{index} | {modality} | {natural_seconds(span.start)}–{natural_seconds(span.end)}"
            if span.level is not None:
                summary += f" | {span.level} | {span.resolution}"
            summaries.append(summary)
            all_ts.extend(bundle.frame_timestamps_s)
            total_frames += bundle.frame_count
            total_audio_s += bundle.audio_duration_s
            for part in bundle.parts[1:]:
                if part["type"] == "audio":
                    audios.append(part["audio"])
                else:
                    parts.append(part)
        header = "\n".join(summaries)
        if audios:
            parts.extend(
                [
                    {
                        "type": "text",
                        "text": "Ordered audio for the audio-bearing spans above, concatenated in listed span order without gaps.",
                    },
                    {"type": "audio", "audio": np.concatenate(audios)},
                ]
            )
        prelim = MediaBundle(
            kind="multi_span",
            header=header,
            parts=tuple([{"type": "text", "text": header}, *parts]),
            frame_timestamps_s=tuple(all_ts),
            audio_duration_s=total_audio_s,
            frame_count=total_frames,
        )
        return self._finalize_tool_bundle(request, prelim, record_budget=True)


__all__ = [
    "InitialMediaState",
    "Media",
    "MediaBudget",
    "Resolution",
    "Sampling",
    "audio_data",
    "build_initial_observations",
    "build_tool_observations",
]
