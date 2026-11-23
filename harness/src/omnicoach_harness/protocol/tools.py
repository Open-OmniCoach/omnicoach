"""`tool_specs` and `OmniOTATools`, the media tools this route exposes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from ..domain.episode import ToolCall, ToolFailure, ToolResult
from ..domain.message import MediaRef
from ..media.extract import extract_audio, extract_frame_batch
from ..media.probe import probe_video
from ..media.timecode import format_timecode
from ..runtime.route import RouteToolError, ToolContext
from .contract import (
    CONTRACT,
    SPAN_MODALITIES,
    VALID_RESOLUTIONS,
    audio_span_schema,
    timecode_schema,
    visual_properties,
    visual_span_schema,
)

LEVEL_FPS = {"coarse": 1.0, "detail": 2.0}


@dataclass(frozen=True, slots=True, kw_only=True)
class _ValidatedSpan:
    span_id: str
    modality: str
    start: float
    end: float
    level: str | None = None
    resolution: str | None = None
    num_frames: int = 0
    audio_seconds: float = 0.0

    def as_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {"id": self.span_id, "modality": self.modality, "start": self.start, "end": self.end}
        if self.level is not None:
            info.update({"level": self.level, "resolution": self.resolution, "num_frames": self.num_frames})
        return info


def _visual_description(tool_name: str) -> str:
    if tool_name == "get_frames":
        base = (
            "Return sampled visual frames, resized within 384x384 while preserving aspect ratio. "
            "level=coarse samples 1 FPS and level=detail samples 2 FPS."
        )
    elif tool_name == "get_clip":
        base = (
            "Return sampled visual frames resized within 384x384, plus continuous audio from the same span. "
            "level=coarse samples visual frames at 1 FPS and level=detail at 2 FPS."
        )
    else:
        raise ValueError(tool_name)
    return (
        f"{base} start/end must be MM:SS.m strings with exactly one decimal. resolution defaults to default. "
        "resolution=high increases spatial resolution up to 1280 pixels per side without changing the time "
        "window or sampling FPS. Use high when the answer depends on fine spatial evidence such as small or "
        "distant objects, small text or symbols, subtle hand-object interactions, or the motion or state change "
        "of a small object. A resolution=high window must be at most 5 seconds."
    )


def _visual_parameters() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": visual_properties(include_default=True),
        "required": ["start", "end", "level"],
    }


def tool_specs() -> list[dict[str, Any]]:
    return [
        {"name": "get_frames", "description": _visual_description("get_frames"), "parameters": _visual_parameters()},
        {
            "name": "get_audio",
            "description": "Return continuous audio. start/end must be MM:SS.m strings with exactly one decimal.",
            "parameters": {
                "type": "object",
                "properties": {"start": timecode_schema(), "end": timecode_schema()},
                "required": ["start", "end"],
            },
        },
        {"name": "get_clip", "description": _visual_description("get_clip"), "parameters": _visual_parameters()},
        {
            "name": "get_multi_span",
            "description": (
                "Return 2–8 already-localized windows together for comparison or verification. Provide spans "
                "in comparison order without names; runtime labels them span_1, span_2, etc. This tool is not "
                "for broad temporal search. clip returns frames plus same-window audio. Each visual span "
                "independently selects level and resolution. resolution=high is for fine spatial evidence and "
                "does not change the time window or sampling FPS. Across one call, use at most 256 sampled "
                "frames, 128 seconds of audio, and at most 2 high-resolution visual spans; each high-resolution "
                "span must be at most 5 seconds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "spans": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": CONTRACT.multi_max_spans,
                        "items": {
                            "oneOf": [
                                audio_span_schema(),
                                visual_span_schema(["frames"], include_default=True, modality_last=True),
                                visual_span_schema(["clip"], include_default=True, modality_last=True),
                            ]
                        },
                    }
                },
                "required": ["spans"],
                "additionalProperties": False,
            },
        },
    ]


def _failure(code: str, message: str, *, retryable: bool = True) -> RouteToolError:
    return RouteToolError(ToolFailure(code=code, message=message, retryable=retryable))


def _number(value: Any, *, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise _failure("TOOL_ARG_TYPE", f"{field} must be internal numeric seconds")
    return float(value)


def _duration(context: ToolContext) -> float:
    if context.task.duration_seconds is not None:
        return float(context.task.duration_seconds)
    return float(probe_video(context.task.video_path, ffprobe=context.ffprobe).duration_seconds)


def _has_audio(context: ToolContext) -> bool:
    if context.task.audio_path:
        return True
    if context.task.has_audio is not None:
        return bool(context.task.has_audio)
    return bool(probe_video(context.task.video_path, ffprobe=context.ffprobe).has_audio)


def _audio_source(context: ToolContext) -> str:
    return context.task.audio_path or context.task.video_path


def _media_tag(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _range(
    *,
    name: str,
    arguments: Mapping[str, Any],
    context: ToolContext,
    maximum: float,
    allow_point: bool,
) -> tuple[float, float]:
    start = _number(arguments.get("start"), field=f"{name}.start")
    end = _number(arguments.get("end"), field=f"{name}.end")
    duration = _duration(context)
    max_end = max(duration, round(duration * 10) / 10.0) + 1e-6
    if start < 0 or end > max_end:
        raise _failure(
            "MEDIA_TS_OOB",
            f"{_timecode_span(start, end)} is outside video [00:00.0,{format_timecode(duration)}]",
        )
    if end < start or (not allow_point and end <= start):
        raise _failure("CLIP_NEG_LEN", f"invalid range {_timecode_span(start, end)}")
    if end - start > maximum + 1e-6:
        raise _failure(
            "MEDIA_SPAN_TOO_LONG",
            f"{_timecode_span(start, end)} is {end - start:g}s; maximum is {maximum:g}s",
        )
    return start, end


def _level(arguments: Mapping[str, Any], *, field: str) -> str:
    value = arguments.get("level")
    if value not in LEVEL_FPS:
        raise _failure("TOOL_ARG_TYPE", f"{field} must be coarse or detail")
    return str(value)


def _resolution(arguments: Mapping[str, Any], *, field: str) -> str:
    value = arguments.get("resolution", "default")
    if value not in VALID_RESOLUTIONS:
        raise _failure("TOOL_ARG_TYPE", f"{field} must be default or high")
    return str(value)


def _frame_count(start: float, end: float, level: str) -> int:
    return max(1, math.ceil(max(0.0, end - start) * LEVEL_FPS[level] - 1e-9))


def _require_minimum_audio(start: float, end: float, *, field: str) -> None:
    if end - start < CONTRACT.min_audio_seconds - 1e-6:
        raise _failure(
            "AUDIO_TOO_SHORT",
            f"{field} duration {end - start:g}s is shorter than minimum {CONTRACT.min_audio_seconds:g}s",
        )


def _fixed_timestamps(start: float, count: int, level: str) -> list[float]:
    step = 1.0 / LEVEL_FPS[level]
    return [start + index * step for index in range(count)]


def _seconds_value(value: float) -> str:
    return f"{round(value, 6):g}"


def _seconds(value: float) -> str:
    return f"{_seconds_value(value)} seconds"


def _seconds_span(start: float, end: float) -> str:
    return f"{_seconds_value(start)}–{_seconds(end)}"


def _timecode_span(start: float, end: float) -> str:
    return f"[{format_timecode(start)},{format_timecode(end)}]"


def _validate_multi_span_item(
    index: int,
    raw: Any,
    *,
    context: ToolContext,
    has_audio: bool,
) -> _ValidatedSpan:
    prefix = f"get_multi_span.spans[{index}]"
    if not isinstance(raw, Mapping):
        raise _failure("TOOL_ARG_TYPE", f"{prefix} must be an object")
    modality = raw.get("modality")
    if modality not in SPAN_MODALITIES:
        raise _failure("TOOL_ARG_TYPE", f"{prefix}.modality must be audio, frames, or clip")
    start = _number(raw.get("start"), field=f"{prefix}.start")
    end = _number(raw.get("end"), field=f"{prefix}.end")
    span_id = f"span_{index + 1}"
    if modality == "audio":
        _range(
            name=prefix,
            arguments={"start": start, "end": end},
            context=context,
            maximum=CONTRACT.max_audio_seconds,
            allow_point=False,
        )
        _require_minimum_audio(start, end, field=prefix)
        return _ValidatedSpan(span_id=span_id, modality="audio", start=start, end=end, audio_seconds=end - start)
    level = _level(raw, field=f"{prefix}.level")
    resolution = _resolution(raw, field=f"{prefix}.resolution")
    maximum = CONTRACT.max_frames / LEVEL_FPS[level]
    if modality == "clip":
        maximum = min(maximum, CONTRACT.max_clip_seconds)
    if resolution == "high":
        maximum = min(maximum, CONTRACT.high_max_span_seconds)
    _range(
        name=prefix,
        arguments={"start": start, "end": end},
        context=context,
        maximum=maximum,
        allow_point=modality == "frames",
    )
    if modality == "clip" and has_audio:
        _require_minimum_audio(start, end, field=prefix)
    return _ValidatedSpan(
        span_id=span_id,
        modality=str(modality),
        start=start,
        end=end,
        level=level,
        resolution=resolution,
        num_frames=_frame_count(start, end, level),
        audio_seconds=(end - start) if (modality == "clip" and has_audio) else 0.0,
    )


def _validate_multi_spans(
    arguments: Mapping[str, Any],
    context: ToolContext,
) -> tuple[list[_ValidatedSpan], int, float, int]:
    spans = arguments.get("spans")
    if not isinstance(spans, list) or len(spans) < 2:
        raise _failure("TOOL_ARG_TYPE", "get_multi_span.spans must contain at least two spans")
    if len(spans) > CONTRACT.multi_max_spans:
        raise _failure("TOO_MANY_SPANS", f"get_multi_span supports at most {CONTRACT.multi_max_spans} spans")
    has_audio = _has_audio(context)
    # Validate all individual constraints before extracting the first span.
    validated = [
        _validate_multi_span_item(index, raw, context=context, has_audio=has_audio)
        for index, raw in enumerate(spans)
    ]
    high_count = sum(1 for item in validated if item.resolution == "high")
    total_frames = sum(item.num_frames for item in validated)
    total_audio = sum(item.audio_seconds for item in validated)
    if high_count > CONTRACT.multi_max_high_resolution_spans:
        raise _failure(
            "TOO_MANY_HIGH_RESOLUTION_SPANS",
            f"get_multi_span needs {high_count} high-resolution spans > "
            f"maximum {CONTRACT.multi_max_high_resolution_spans}",
        )
    if total_frames > CONTRACT.multi_max_total_frames:
        raise _failure(
            "TOO_MANY_FRAMES",
            f"get_multi_span needs {total_frames} frames > total maximum {CONTRACT.multi_max_total_frames}",
        )
    if total_audio > CONTRACT.multi_max_total_audio_seconds + 1e-6:
        raise _failure(
            "MEDIA_SPAN_TOO_LONG",
            f"get_multi_span audio totals {total_audio:g}s > maximum {CONTRACT.multi_max_total_audio_seconds:g}s",
        )
    if any(item.modality == "audio" for item in validated) and not has_audio:
        raise _failure("NO_AUDIO", "get_multi_span requested audio but the source has no audio")
    return validated, total_frames, total_audio, high_count


class OmniOTATools:
    def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        if call.name == "get_frames":
            return self._frames(call.arguments, context, stem="get_frames")
        if call.name == "get_audio":
            return self._audio(call.arguments, context, stem="audio")
        if call.name == "get_clip":
            return self._clip(call.arguments, context, stem="get_clip")
        if call.name == "get_multi_span":
            return self._multi(call.arguments, context)
        raise _failure("UNKNOWN_TOOL", f"unknown tool: {call.name}")

    def _frames(self, arguments: Mapping[str, Any], context: ToolContext, *, stem: str) -> ToolResult:
        level = _level(arguments, field="get_frames.level")
        resolution = _resolution(arguments, field="get_frames.resolution")
        maximum = CONTRACT.max_frames / LEVEL_FPS[level]
        if resolution == "high":
            maximum = min(maximum, CONTRACT.high_max_span_seconds)
        start, end = _range(name="get_frames", arguments=arguments, context=context, maximum=maximum, allow_point=True)
        count = _frame_count(start, end, level)
        if count > CONTRACT.max_frames:
            raise _failure("TOO_MANY_FRAMES", f"{count} frames exceeds configured maximum")
        timestamps = _fixed_timestamps(start, count, level)
        max_side = CONTRACT.high_max_side if resolution == "high" else CONTRACT.default_max_side
        tag = _media_tag(
            {
                "video": context.task.video_path,
                "start": start,
                "end": end,
                "level": level,
                "resolution": resolution,
                "max_side": max_side,
            }
        )
        output_dir = (
            context.work_dir / f"step_{context.step:02d}_{stem}_{start:.3f}_{end:.3f}_{level}_{resolution}_{tag}"
        )
        batch = extract_frame_batch(
            context.task.video_path,
            timestamps=timestamps,
            output_dir=output_dir,
            max_side=max_side,
            ffmpeg=context.ffmpeg,
            ffprobe=context.ffprobe,
        )
        media = tuple(
            MediaRef(kind="image", local_path=str(path), mime_type="image/jpeg", label=f"<{_seconds(timestamp)}>")
            for timestamp, path in zip(timestamps, batch.paths, strict=True)
        )
        return ToolResult(
            tool_name="get_frames",
            text=f"frames | {_seconds_span(start, end)} | {level} | {resolution}",
            media=media,
            info={
                "start": start,
                "end": end,
                "num": count,
                "level": level,
                "level_fps": LEVEL_FPS[level],
                "resolution": resolution,
                "decoder": batch.decoder,
                "max_frame_side": max_side,
                "source_size": list(batch.source_size),
                "output_size": list(batch.output_size),
                "source_fps": batch.source_fps,
                "frame_indices": list(batch.frame_indices),
                "frame_timestamps_seconds": timestamps,
                "frame_timecodes": [format_timecode(item) for item in timestamps],
            },
        )

    def _audio(self, arguments: Mapping[str, Any], context: ToolContext, *, stem: str) -> ToolResult:
        if not _has_audio(context):
            raise _failure("NO_AUDIO", "source has no audio")
        start, end = _range(
            name="get_audio",
            arguments=arguments,
            context=context,
            maximum=CONTRACT.max_audio_seconds,
            allow_point=False,
        )
        _require_minimum_audio(start, end, field="get_audio")
        source = _audio_source(context)
        tag = _media_tag({"source": source, "start": start, "end": end})
        output_path = context.work_dir / f"step_{context.step:02d}_{stem}_{start:.3f}_{end:.3f}_{tag}.m4a"
        path = extract_audio(source, start=start, end=end, output_path=output_path, ffmpeg=context.ffmpeg)
        return ToolResult(
            tool_name="get_audio",
            text=f"audio | {_seconds_span(start, end)}",
            media=(MediaRef(kind="audio", local_path=str(path), mime_type="audio/mp4", label="<audio>"),),
            info={"start": start, "end": end, "duration": end - start},
        )

    def _clip(self, arguments: Mapping[str, Any], context: ToolContext, *, stem: str) -> ToolResult:
        level = _level(arguments, field="get_clip.level")
        resolution = _resolution(arguments, field="get_clip.resolution")
        maximum = min(CONTRACT.max_clip_seconds, CONTRACT.max_frames / LEVEL_FPS[level])
        if resolution == "high":
            maximum = min(maximum, CONTRACT.high_max_span_seconds)
        start, end = _range(name="get_clip", arguments=arguments, context=context, maximum=maximum, allow_point=False)
        has_audio = _has_audio(context)
        if has_audio:
            # Validate audio bounds before spending decode work on visual frames.
            _require_minimum_audio(start, end, field="get_clip")
        normalized: dict[str, Any] = {"start": start, "end": end, "level": level}
        if resolution == "high":
            normalized["resolution"] = "high"
        frames = self._frames(normalized, context, stem=f"{stem}_frames")
        media = list(frames.media)
        audio_info = None
        if has_audio:
            audio = self._audio({"start": start, "end": end}, context, stem=f"{stem}_audio")
            media.extend(audio.media)
            audio_info = dict(audio.info)
        return ToolResult(
            tool_name="get_clip",
            text=(
                f"clip | {_seconds_span(start, end)} | {level} | {resolution}"
                + (" | audio" if audio_info else " | no_audio")
            ),
            media=tuple(media),
            info={
                **dict(frames.info),
                "duration": end - start,
                "audio_included": audio_info is not None,
                "audio": audio_info,
                "mode": "sampled_frames_plus_continuous_audio",
            },
        )

    def _execute_multi_span_item(
        self,
        index: int,
        span: _ValidatedSpan,
        context: ToolContext,
    ) -> tuple[str, list[MediaRef], dict[str, Any]]:
        subargs: dict[str, Any] = {"start": span.start, "end": span.end}
        if span.level is not None:
            subargs["level"] = span.level
            if span.resolution == "high":
                subargs["resolution"] = "high"
        if span.modality == "frames":
            result = self._frames(subargs, context, stem=f"multi_{index:02d}_{span.span_id}_frames")
        elif span.modality == "audio":
            result = self._audio(subargs, context, stem=f"multi_{index:02d}_{span.span_id}_audio")
        else:
            result = self._clip(subargs, context, stem=f"multi_{index:02d}_{span.span_id}_clip")
        header = " | ".join(
            filter(
                None,
                [
                    span.span_id,
                    span.modality,
                    _seconds_span(span.start, span.end),
                    span.level or "",
                    span.resolution or "",
                ],
            )
        )
        labeled: list[MediaRef] = []
        for media_index, media in enumerate(result.media):
            label = media.label or f"<{media.kind}>"
            if media_index == 0:
                label = f"{header}\n{label}"
            labeled.append(replace(media, label=label))
        return header, labeled, {**span.as_info(), "result": dict(result.info)}

    def _multi(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        validated, total_frames, total_audio, high_count = _validate_multi_spans(arguments, context)
        all_media: list[MediaRef] = []
        span_info: list[dict[str, Any]] = []
        lines = ["ordered multi-span evidence"]
        for index, span in enumerate(validated):
            header, labeled, entry = self._execute_multi_span_item(index, span, context)
            lines.append(header)
            all_media.extend(labeled)
            span_info.append(entry)
        return ToolResult(
            tool_name="get_multi_span",
            text="\n".join(lines),
            media=tuple(all_media),
            info={
                "spans": span_info,
                "total_frames": total_frames,
                "total_audio_seconds": total_audio,
                "high_resolution_spans": high_count,
                "mode": "ordered_multi_span_comparison",
                "span_id_mode": "runtime_ordered",
            },
        )
