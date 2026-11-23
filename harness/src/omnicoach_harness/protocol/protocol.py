"""`decode`, `response_schema` and `parse_temporal_intervals` for this route's OTA JSON."""

from __future__ import annotations

import json
import re
from typing import Any

from ..domain.episode import ActorDecision, FinalAnswer, ProtocolFailure, ToolCall
from ..runtime.route import ActorState, RouteProtocolError
from .contract import (
    CONTRACT,
    SPAN_MODALITIES,
    TIMECODE_PATTERN,
    VALID_LEVELS,
    VALID_RESOLUTIONS,
    VISUAL_MODALITIES,
    audio_span_schema,
    timecode_schema,
    visual_properties,
    visual_span_schema,
)

VISUAL_ACTIONS = {"get_frames", "get_clip"}
TIMECODE_RE = re.compile(TIMECODE_PATTERN)
TIMECODE_RANGE_RE = re.compile(r"\[\s*\d+:[0-5]\d\.\d\s*,\s*\d+:[0-5]\d\.\d\s*\]")
TEMPORAL_INTERVAL_RE = re.compile(r"\[\s*(\d+:[0-5]\d\.\d)\s*,\s*(\d+:[0-5]\d\.\d)\s*\]")


def _fail(code: str, message: str) -> RouteProtocolError:
    return RouteProtocolError(ProtocolFailure(code=code, message=message))


def _load(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise _fail("INVALID_JSON", "not a string")
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise _fail("INVALID_JSON", "must start with '{' and end with '}'")
    try:
        value, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as exc:
        raise _fail("INVALID_JSON", f"json decode error: {exc}") from exc
    if text[end:].strip() or not isinstance(value, dict):
        raise _fail("INVALID_JSON", "expected one JSON object without trailing content")
    return value


def _keys(
    action_type: str,
    value: dict[str, Any],
    allowed: set[str],
    *,
    required: set[str] | None = None,
) -> None:
    required = allowed if required is None else required
    extra = set(value) - allowed
    missing = required - set(value)
    if extra or missing:
        raise _fail(
            "INVALID_JSON",
            f"{action_type} action keys invalid; missing={sorted(missing)}, extra={sorted(extra)}",
        )


def _time(value: Any, *, field: str) -> float:
    if not isinstance(value, str) or TIMECODE_RE.fullmatch(value.strip()) is None:
        raise _fail("INVALID_TIMECODE", f"{field} must match MM:SS.m, for example 01:35.0")
    minutes, seconds = value.strip().split(":")
    return int(minutes) * 60 + float(seconds)


def parse_temporal_intervals(content: str) -> tuple[tuple[float, float], ...]:
    """Parse the complete TR answer; prose and point-only answers are invalid."""

    if not isinstance(content, str) or not content.strip():
        raise ValueError("temporal answer must be a non-empty interval string")
    raw_intervals = content.strip().split(";")
    intervals: list[tuple[float, float]] = []
    for raw_interval in raw_intervals:
        match = TEMPORAL_INTERVAL_RE.fullmatch(raw_interval.strip())
        if match is None:
            raise ValueError("expected [MM:SS.m,MM:SS.m] or semicolon-separated intervals with no other text")
        start = _time(match.group(1), field="answer.start")
        end = _time(match.group(2), field="answer.end")
        intervals.append((start, end))
    return tuple(intervals)


def _visual_options(source: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    level = source.get("level")
    if level not in VALID_LEVELS:
        raise _fail("INVALID_JSON", f"{prefix}.level must be coarse or detail")
    resolution = source.get("resolution", "default")
    if resolution not in VALID_RESOLUTIONS:
        raise _fail("INVALID_JSON", f"{prefix}.resolution must be default or high")
    result: dict[str, Any] = {"level": level}
    if resolution == "high":
        result["resolution"] = "high"
    return result


def _single_arguments(action_type: str, action: dict[str, Any]) -> dict[str, Any]:
    if action_type in VISUAL_ACTIONS:
        _keys(
            action_type,
            action,
            {"type", "start", "end", "level", "resolution"},
            required={"type", "start", "end", "level"},
        )
        return {
            "start": _time(action.get("start"), field=f"{action_type}.start"),
            "end": _time(action.get("end"), field=f"{action_type}.end"),
            **_visual_options(action, prefix=action_type),
        }
    _keys(action_type, action, {"type", "start", "end"})
    return {
        "start": _time(action.get("start"), field=f"{action_type}.start"),
        "end": _time(action.get("end"), field=f"{action_type}.end"),
    }


def _multi_arguments(action: dict[str, Any]) -> dict[str, Any]:
    _keys("get_multi_span", action, {"type", "spans"})
    spans = action.get("spans")
    if not isinstance(spans, list) or len(spans) < 2:
        raise _fail("INVALID_JSON", "get_multi_span.spans must contain at least two spans")
    normalized: list[dict[str, Any]] = []
    for index, raw_span in enumerate(spans):
        prefix = f"get_multi_span.spans[{index}]"
        if not isinstance(raw_span, dict):
            raise _fail("INVALID_JSON", f"{prefix} must be an object")
        modality = raw_span.get("modality")
        allowed = {"modality", "start", "end"}
        if modality in VISUAL_MODALITIES:
            allowed |= {"level", "resolution"}
        required = allowed - {"resolution"}
        keys = set(raw_span)
        if not required <= keys <= allowed:
            raise _fail(
                "INVALID_JSON",
                f"{prefix} keys invalid; required={sorted(required)}, "
                f"optional={sorted(allowed - required)}, extra={sorted(keys - allowed)}",
            )
        if modality not in SPAN_MODALITIES:
            raise _fail("INVALID_JSON", f"{prefix}.modality must be audio, frames, or clip")
        item: dict[str, Any] = {
            "modality": modality,
            "start": _time(raw_span.get("start"), field=f"{prefix}.start"),
            "end": _time(raw_span.get("end"), field=f"{prefix}.end"),
        }
        if modality in VISUAL_MODALITIES:
            item.update(_visual_options(raw_span, prefix=prefix))
        normalized.append(item)
    return {"spans": normalized}


def decode(raw: str, state: ActorState) -> ActorDecision:
    value = _load(raw)
    if set(value) != {"observation", "think", "action"}:
        raise _fail("INVALID_JSON", "top-level keys must be exactly observation, think, and action")
    observation, think, action = value.get("observation"), value.get("think"), value.get("action")
    if not isinstance(observation, str):
        raise _fail("INVALID_JSON", "'observation' must be a string")
    if not isinstance(think, str) or not think.strip():
        raise _fail("INVALID_JSON", "'think' must be a non-empty string")
    if not isinstance(action, dict):
        raise _fail("INVALID_JSON", "'action' must be an object")
    if state.need_observation:
        if not observation.strip():
            raise _fail("MISSING_OBSERVATION", "observation after a tool result must be non-empty")
        if TIMECODE_RANGE_RE.search(observation) is None:
            raise _fail(
                "MISSING_OBSERVATION_TIMESTAMP",
                "observation after a tool result must include a [MM:SS.m,MM:SS.m] span",
            )
    action_type = action.get("type")
    if action_type == "answer":
        _keys("answer", action, {"type", "content"})
        content = action.get("content")
        if not isinstance(content, str):
            raise _fail("ANSWER_NOT_STRING", "'content' missing or not string")
        command: ToolCall | FinalAnswer = FinalAnswer(content=content)
    elif action_type == "get_multi_span":
        command = ToolCall(name="get_multi_span", arguments=_multi_arguments(action))
    elif action_type in {"get_frames", "get_audio", "get_clip"}:
        command = ToolCall(name=str(action_type), arguments=_single_arguments(str(action_type), action))
    else:
        raise _fail("UNKNOWN_ACTION_TYPE", f"unknown action.type: {action_type}")
    state.need_observation = False
    return ActorDecision(command=command, observation=observation, think=think, raw=raw)


def response_schema() -> dict[str, Any]:
    def action_schema(action_type: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"type": {"type": "string", "enum": [action_type]}, **properties},
            "required": ["type", *required],
            "additionalProperties": False,
        }

    spans = {
        "type": "array",
        "minItems": 2,
        "maxItems": CONTRACT.multi_max_spans,
        "items": {"anyOf": [audio_span_schema(), visual_span_schema(VISUAL_MODALITIES)]},
    }
    return {
        "type": "object",
        "required": ["observation", "think", "action"],
        "additionalProperties": False,
        "properties": {
            "observation": {"type": "string"},
            "think": {"type": "string", "minLength": 1},
            "action": {
                "anyOf": [
                    action_schema("answer", {"content": {"type": "string"}}, ["content"]),
                    action_schema("get_frames", visual_properties(), ["start", "end", "level"]),
                    action_schema("get_clip", visual_properties(), ["start", "end", "level"]),
                    action_schema(
                        "get_audio",
                        {"start": timecode_schema(), "end": timecode_schema()},
                        ["start", "end"],
                    ),
                    action_schema("get_multi_span", {"spans": spans}, ["spans"]),
                ]
            },
        },
    }
