"""Qwen3-Omni active-perception agent protocol, parsing, and session state.

Protocol invariants:
- Trailing-window clamping validates ``0 <= start < duration`` and
  ``clamped_end = min(end, duration) > start``, guarding short videos (<1s)
  and preventing negative durations in ``get_multi_span``.
- ``ToolRequest.identity`` uses canonical normalized numeric/enum fields so
  equivalent calls (``"10s"`` vs ``"10.0s"``, omitted vs ``"default"``
  resolution) are detected as duplicates.
- Control-state ``[NOTICE]`` stripping only removes trailing tagged notice
  blocks on ``user`` / ``tool`` messages, preserving ``assistant``, ``system``,
  and user questions containing literal ``"[NOTICE]"``.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Template

SYSTEM_PROMPT = """You are a multimodal video-QA agent.

For every assistant turn, produce outputs in this exact order:
1. <observation>...</observation>
2. <think>...</think>
3. Then either invoke exactly one provided tool through native function calling, or output <answer>...</answer>.

Observation rules:
- Describe only media available in the current request.
- Never predict the result of the tool selected in the same turn.
- On the first turn, summarize useful facts from the initial framepack and audio.
- After a tool response, summarize only that latest tool result.
- Cite focused evidence with absolute seconds and preserve uncertainty.

Thinking rules:
- Track verified evidence, unresolved alternatives, and the smallest useful next check.
- Do not repeat an identical successful call. Reversing ordered multi-span evidence is a distinct comparison.
- Use high resolution only for answer-critical fine spatial evidence.
- Stop once the initial evidence or focused evidence is sufficient.

Tool rules:
- A first-turn <answer> is allowed when the initial framepack/audio is sufficient.
- Otherwise invoke exactly one provided function per turn.
- Tool start/end values are absolute offsets with an s suffix, including fractional values such as 178.5s.
- get_multi_span is for ordered comparison of already-localized windows, not broad search.
- Respect all per-tool, high-resolution, and multi-span limits in the tool schemas.

Answer rules:
- Do not call a tool when answering.
- Put the final answer inside <answer>...</answer>."""

DEFAULT_MEDIA_SETTINGS: dict[str, Any] = {
    "png_compress_level": 1,
    "min_span_seconds": 1e-3,
    "initial": {
        "fps": 0.5,
        "max_frames": 256,
        "max_pixels": 50176,
        "end_margin_ms": 200,
        "audio_max_seconds": 3300,
    },
    "fps": {
        "coarse": 1.0,
        "detail": 2.0,
    },
    "resolutions": {
        "default": {"source": "cache"},
        "high": {"source": "original", "max_pixels": 1638400, "max_seconds": 5},
    },
    "tools": {
        "get_frames": {"max_frames": 256},
        "get_audio": {"max_seconds": 256},
        "get_clip": {"max_seconds": 60},
        "get_multi_span": {
            "min_spans": 2,
            "max_spans": 8,
            "max_frames": 256,
            "max_audio_seconds": 128,
            "max_high_spans": 2,
        },
    },
}

_TOOLS_TEMPLATE = Template(Path(__file__).with_name("tools.json.jinja").read_text())
CHAT_TEMPLATE = Path(__file__).with_name("chat_template.jinja").read_text()

VISUAL_HEADERS = {
    ("get_frames", "default"): (
        "Frames {start} to {end} (level={level}, fps={fps:g}, num={count}, "
        "decoder=decord, max_side={max_side})."
    ),
    ("get_clip", "default"): (
        "Clip {start} to {end} (level={level}, decoder=decord, "
        "max_side={max_side}; {audio})."
    ),
    ("get_frames", "high"): (
        "Frames {start} to {end} (level={level}, resolution=high, max_side={max_side})."
    ),
    ("get_clip", "high"): (
        "Clip {start} to {end} (level={level}, resolution=high, "
        "max_side={max_side}; {audio})."
    ),
}

MEDIA_OMITTED = "[MEDIA OMITTED - Refer to your Observation]"

_SECONDS_RE = re.compile(r"^(\d+(?:\.\d+)?)s$")
_TOOL_TEXT_RE = re.compile(
    r"^\s*<observation>[\s\S]*?</observation>\s*\n"
    r"<think>[\s\S]*?</think>\s*$"
)
_ANSWER_TEXT_RE = re.compile(
    r"^\s*<observation>[\s\S]*?</observation>\s*\n"
    r"<think>[\s\S]*?</think>\s*\n"
    r"<answer>[\s\S]*?</answer>\s*$"
)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>")
_ANSWER_RE = re.compile(r"<answer>\s*([\s\S]*?)\s*</answer>", re.IGNORECASE)
_TRAILING_NOTICE_RE = re.compile(
    r"\n?\[NOTICE\] (?:STEP LIMIT REACHED\.|Operation terminated\.|FINAL STEP!|Step \d+/\d+\.)[^\n]*$"
)


class ProtocolError(ValueError):
    """Model-visible protocol violation during tool or answer parsing."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ToolRequest:
    call_id: str
    name: str
    start_code: str
    end_code: str
    start: float
    end: float
    level: str | None = None
    resolution: str = "default"
    spans: tuple["ToolRequest", ...] = ()
    identity: str = ""


@dataclass(frozen=True)
class ParsedAssistantTurn:
    text: str
    tool_calls: tuple[dict[str, Any], ...]

    def as_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.text}
        if self.tool_calls:
            message["tool_calls"] = [dict(call) for call in self.tool_calls]
        return message


@dataclass(frozen=True)
class MediaBundle:
    kind: str
    header: str
    parts: tuple[dict[str, Any], ...]
    frame_timestamps_s: tuple[float, ...] = ()
    audio_duration_s: float = 0.0
    frame_count: int = 0
    cost_units: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def render_tools(settings: Mapping[str, Any], frames: Mapping[str, Any]) -> list[dict[str, Any]]:
    return json.loads(
        _TOOLS_TEMPLATE.render(
            media=settings,
            default_max_side=math.isqrt(int(frames["max_pixels"])),
            high_max_side=math.isqrt(int(settings["resolutions"]["high"]["max_pixels"])),
        )
    )


def tool_max_seconds(settings: Mapping[str, Any], name: str, level: str | None) -> float:
    tool = settings["tools"][name]
    if "max_frames" in tool:
        return float(tool["max_frames"]) / float(settings["fps"][level])
    return float(tool["max_seconds"])


def natural_seconds(seconds: float) -> str:
    value = float(seconds)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"seconds must be a non-negative finite number, got {value!r}")
    if value.is_integer():
        return f"{int(value)} seconds"
    return f"{value:g} seconds"


def parse_seconds(value: Any, field_name: str) -> float:
    if not isinstance(value, str):
        raise ProtocolError(
            "TOOL_ARG_TYPE",
            f"{field_name} must be an absolute seconds string with an s suffix",
        )
    match = _SECONDS_RE.fullmatch(value.strip())
    if match is None:
        raise ProtocolError(
            "INVALID_TIMECODE",
            f"{field_name} must match an absolute seconds value such as 178s; got {value!r}",
        )
    result = float(match.group(1))
    if not math.isfinite(result):
        raise ProtocolError("INVALID_TIMECODE", f"{field_name} must be finite")
    return result


def parse_generated_turn(raw: str) -> ParsedAssistantTurn:
    text = str(raw or "")
    matches = list(_TOOL_CALL_RE.finditer(text))
    calls: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ProtocolError("INVALID_JSON", f"tool call JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ProtocolError("INVALID_TOOL_CALL", "tool call payload must be an object")
        name = payload.get("name")
        arguments = payload.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, Mapping):
            raise ProtocolError(
                "INVALID_TOOL_CALL",
                "tool call requires string name and object arguments",
            )
        calls.append(
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": name, "arguments": dict(arguments)},
            }
        )
    assistant_text = _TOOL_CALL_RE.sub("", text).strip()
    if "<tool_call>" in assistant_text or "</tool_call>" in assistant_text:
        raise ProtocolError("INVALID_JSON", "malformed or unclosed native tool call")
    return ParsedAssistantTurn(text=assistant_text, tool_calls=tuple(calls))


def validate_assistant_text(text: str, *, has_tool_call: bool) -> None:
    pattern = _TOOL_TEXT_RE if has_tool_call else _ANSWER_TEXT_RE
    if pattern.fullmatch(text or "") is None:
        expected = (
            "<observation> then <think> before one native tool call"
            if has_tool_call
            else "<observation>, <think>, then <answer>"
        )
        raise ProtocolError(
            "INVALID_ASSISTANT_FORMAT",
            f"assistant turn must contain {expected}",
        )


def extract_answer(text: str) -> str | None:
    matches = list(_ANSWER_RE.finditer(str(text or "")))
    if not matches:
        return None
    answer = matches[-1].group(1).strip()
    return answer or None


def classify_turn_role(
    raw_response: str,
    *,
    parsed: ParsedAssistantTurn | None = None,
    has_parse_error: bool = False,
) -> str:
    """Classify assistant intent into 'answer', 'tool', or 'invalid'.

    Preserves answer isolation even when an answer turn has minor formatting
    issues (no tool-call markup and contains <answer>).
    """
    if parsed is not None:
        if parsed.tool_calls:
            return "tool"
        if extract_answer(parsed.text) is not None or "<answer>" in parsed.text.lower():
            return "answer"
        return "invalid"
    text = str(raw_response or "")
    if "<tool_call>" in text:
        return "invalid" if has_parse_error else "tool"
    if "<answer>" in text.lower():
        return "answer"
    return "invalid"


def _arguments(call: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    function = call.get("function")
    if not isinstance(function, Mapping):
        raise ProtocolError("INVALID_TOOL_CALL", "tool call has no function object")
    name = str(function.get("name") or "")
    raw = function.get("arguments") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProtocolError("INVALID_JSON", str(exc)) from exc
    if not isinstance(raw, Mapping):
        raise ProtocolError("INVALID_TOOL_ARGUMENTS", "tool arguments must be an object")
    return name, dict(raw)


def _canonical_single_identity(
    name: str,
    start: float,
    end: float,
    level: str | None,
    resolution: str,
) -> str:
    """Canonical identity resistant to formatting aliases like 10s vs 10.0s."""
    payload: dict[str, Any] = {
        "name": name,
        "start": round(float(start), 6),
        "end": round(float(end), 6),
    }
    if name != "get_audio":
        payload["level"] = level
        payload["resolution"] = resolution
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _single_tool_request(
    name: str,
    arguments: Mapping[str, Any],
    *,
    call_id: str,
    duration: float,
    has_audio: bool,
    settings: Mapping[str, Any],
) -> ToolRequest:
    if not math.isfinite(duration) or duration <= 0.0:
        raise ProtocolError("INVALID_MEDIA_RANGE", f"invalid video duration {duration!r}")
    if name not in {"get_frames", "get_audio", "get_clip"}:
        raise ProtocolError("UNKNOWN_TOOL", f"unsupported tool {name!r}")

    allowed = {"start", "end"} if name == "get_audio" else {"start", "end", "level", "resolution"}
    required = {"start", "end"} if name == "get_audio" else {"start", "end", "level"}
    if not required.issubset(arguments) or not set(arguments).issubset(allowed):
        raise ProtocolError("INVALID_TOOL_ARGUMENTS", f"{name} arguments do not match its schema")

    start_code = str(arguments.get("start") or "")
    end_code = str(arguments.get("end") or "")
    start = parse_seconds(arguments.get("start"), "start")
    end = parse_seconds(arguments.get("end"), "end")

    # Strictly verify 0 <= start < duration and clamped_end > start before returning.
    clamped_end = min(end, duration)
    max_allowed_end = max(duration, float(math.ceil(duration)))
    min_span = min(float(settings.get("min_span_seconds", 1e-3)), 0.5 * duration)
    if (
        start < 0.0
        or start >= duration
        or end <= start
        or end > max_allowed_end
        or clamped_end <= start
        or (clamped_end - start) < min_span - 1e-12
    ):
        raise ProtocolError(
            "INVALID_MEDIA_RANGE",
            f"{start_code} to {end_code} is outside or collapses on video duration {duration:.3f}s",
        )

    level = None if name == "get_audio" else str(arguments.get("level") or "")
    resolution = "default" if name == "get_audio" else str(arguments.get("resolution") or "default")
    if level is not None and level not in settings["fps"]:
        raise ProtocolError("INVALID_LEVEL", f"level must be one of {sorted(settings['fps'])}")
    if resolution not in settings["resolutions"]:
        raise ProtocolError(
            "INVALID_RESOLUTION",
            f"resolution must be one of {sorted(settings['resolutions'])}",
        )

    span = clamped_end - start
    maximum = tool_max_seconds(settings, name, level)
    if span > maximum + 1e-9:
        raise ProtocolError("MEDIA_SPAN_TOO_LONG", f"{name} span exceeds {maximum:g} seconds")
    resolution_limit = settings["resolutions"][resolution].get("max_seconds")
    if resolution_limit is not None and span > float(resolution_limit) + 1e-9:
        raise ProtocolError(
            "HIGH_RES_SPAN_TOO_LONG",
            f"high-resolution span exceeds {resolution_limit:g} seconds",
        )
    if name == "get_audio" and not has_audio:
        raise ProtocolError("NO_AUDIO", "video has no audio")

    identity = _canonical_single_identity(name, start, clamped_end, level, resolution)
    return ToolRequest(
        call_id=call_id,
        name=name,
        start_code=start_code,
        end_code=end_code,
        start=start,
        end=clamped_end,
        level=level,
        resolution=resolution,
        identity=identity,
    )


def parse_tool_call(
    call: Mapping[str, Any],
    *,
    duration: float,
    has_audio: bool,
    settings: Mapping[str, Any],
) -> ToolRequest:
    name, arguments = _arguments(call)
    call_id = str(call.get("id") or "")
    if name != "get_multi_span":
        return _single_tool_request(
            name,
            arguments,
            call_id=call_id,
            duration=duration,
            has_audio=has_audio,
            settings=settings,
        )

    if set(arguments) != {"spans"} or not isinstance(arguments["spans"], list):
        raise ProtocolError("INVALID_TOOL_ARGUMENTS", "get_multi_span requires exactly spans")
    raw_spans = arguments["spans"]
    limits = settings["tools"][name]
    if not limits["min_spans"] <= len(raw_spans) <= limits["max_spans"]:
        raise ProtocolError(
            "INVALID_MULTI_SPAN",
            f"get_multi_span requires {limits['min_spans']} to {limits['max_spans']} spans",
        )

    spans: list[ToolRequest] = []
    total_frames = 0
    total_audio = 0.0
    high_count = 0
    for index, raw_span in enumerate(raw_spans):
        if not isinstance(raw_span, Mapping):
            raise ProtocolError("INVALID_MULTI_SPAN", f"spans[{index}] must be an object")
        modality = str(raw_span.get("modality") or "")
        tool_name = {"frames": "get_frames", "audio": "get_audio", "clip": "get_clip"}.get(modality)
        if tool_name is None:
            raise ProtocolError("INVALID_MULTI_SPAN", f"unsupported modality {modality!r}")
        payload = {key: value for key, value in raw_span.items() if key != "modality"}
        span_request = _single_tool_request(
            tool_name,
            payload,
            call_id=call_id,
            duration=duration,
            has_audio=has_audio,
            settings=settings,
        )
        spans.append(span_request)
        span_dur = span_request.end - span_request.start  # strictly > 0
        if modality in {"frames", "clip"}:
            fps = float(settings["fps"][span_request.level])
            total_frames += max(1, math.ceil(span_dur * fps))
        if modality in {"audio", "clip"} and has_audio:
            total_audio += span_dur
        high_count += int(span_request.resolution == "high")

    if (
        total_frames > limits["max_frames"]
        or total_audio > limits["max_audio_seconds"] + 1e-9
        or high_count > limits["max_high_spans"]
    ):
        raise ProtocolError(
            "MULTI_SPAN_BUDGET",
            f"multi-span budget exceeded: frames={total_frames}, audio={total_audio:g}, high={high_count}",
        )
    identity = json.dumps(
        {"name": name, "spans": [item.identity for item in spans]},
        sort_keys=True,
        separators=(",", ":"),
    )
    return ToolRequest(
        call_id=call_id,
        name=name,
        start_code=spans[0].start_code,
        end_code=spans[-1].end_code,
        start=min(item.start for item in spans),
        end=max(item.end for item in spans),
        spans=tuple(spans),
        identity=identity,
    )


def require_single_tool_call(
    calls: Sequence[Mapping[str, Any]],
    *,
    duration: float,
    has_audio: bool,
    settings: Mapping[str, Any],
) -> ToolRequest:
    if len(calls) != 1:
        raise ProtocolError("TOOL_CALL_COUNT", f"expected exactly one tool call, got {len(calls)}")
    return parse_tool_call(calls[0], duration=duration, has_audio=has_audio, settings=settings)


def render_notice(*, step: int, max_steps: int, done: bool) -> str:
    remain = max_steps - step
    if done:
        return (
            "\n[NOTICE] STEP LIMIT REACHED. Operation terminated."
            if remain <= 0
            else f"\n[NOTICE] Operation terminated. Steps used: {step}/{max_steps}."
        )
    if remain == 1:
        return "\n[NOTICE] FINAL STEP! You MUST provide your answer now."
    return f"\n[NOTICE] Step {step}/{max_steps}. {remain} steps remaining."


def render_error(
    code: str,
    message: str,
    *,
    retry_exceeded: bool = False,
    consecutive_invalid_actions: int = 0,
    max_invalid_retries: int = 0,
) -> str:
    detail = f"{code}: {message}" if message and message != code else code
    if retry_exceeded:
        detail += (
            f"\n[SYSTEM] Invalid action retry budget exceeded "
            f"({consecutive_invalid_actions}/{max_invalid_retries}). "
            "This invalid action is recorded, but the episode will continue. "
            "Switch to a simpler valid action now."
        )
    detail += (
        "\n[SYSTEM] The previous action was invalid. Analyze the error diagnostic above, "
        "resolve the conflict, and issue a corrected command to avoid wasting steps."
    )
    return f"[ERROR] {detail}"


def build_initial_user_text(
    *,
    prompt: str,
    duration_seconds: float,
    has_audio: bool,
    initial_audio_attached: bool,
    initial_header: str,
    settings: Mapping[str, Any],
) -> str:
    task_prompt = str(prompt).replace("<video>", "", 1).strip()
    lines = [
        "Video:",
        f"- duration: {duration_seconds:.3f}s",
        f"- has_audio: {has_audio}",
        "- Tool start/end values use absolute seconds with an s suffix.",
        "",
        "Legal tool spans:",
        f"- get_frames: coarse <= {tool_max_seconds(settings, 'get_frames', 'coarse'):g}s, "
        f"detail <= {tool_max_seconds(settings, 'get_frames', 'detail'):g}s.",
        f"- get_audio: <= {tool_max_seconds(settings, 'get_audio', None):g}s.",
        f"- get_clip: coarse <= {tool_max_seconds(settings, 'get_clip', 'coarse'):g}s, "
        f"detail <= {tool_max_seconds(settings, 'get_clip', 'detail'):g}s; same-span audio included.",
        "",
        task_prompt,
        "",
        "Initial framepack media:",
        initial_header,
        "- These timestamped frames are a global scan, not a continuous video.",
    ]
    if initial_audio_attached:
        lines.append("- The attached full audio may be used for broad speech/sound localization.")
    elif has_audio:
        lines.append(
            "- Full audio is not attached to this initial request; audio remains "
            "available through get_audio and get_clip."
        )
    else:
        lines.append(
            "- No audio is available; do not call get_audio. get_clip returns visual frames only."
        )
    lines.append(
        "- A successful focused tool call replaces this initial media because "
        "media.keep_recent=1; preserve useful initial evidence in observation/think."
    )
    return "\n".join(lines)


def _has_media(content: Any) -> tuple[bool, bool]:
    if isinstance(content, str) or content is None:
        return False, False
    kinds = {str(part.get("type") or "").lower() for part in content if isinstance(part, Mapping)}
    return "image" in kinds, bool(kinds.intersection({"audio", "video"}))


def _first_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is not None:
        for part in content:
            if isinstance(part, Mapping) and str(part.get("type") or "").lower() == "text":
                return str(part.get("text") or "")
    return "Media content"


def _strip_control_state_from_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Strip only injected trailing control-state notices from user/tool messages."""
    cleaned = dict(message)
    role = cleaned.get("role")
    if role not in {"user", "tool"}:
        return cleaned
    content = cleaned.get("content")
    if isinstance(content, str):
        cleaned["content"] = _TRAILING_NOTICE_RE.sub("", content).rstrip()
        return cleaned
    if isinstance(content, list):
        kept: list[Any] = []
        for part in content:
            if isinstance(part, Mapping) and part.get("_control_notice"):
                continue
            if isinstance(part, Mapping) and str(part.get("type") or "").lower() == "text":
                text = str(part.get("text") or "")
                stripped = _TRAILING_NOTICE_RE.sub("", text).rstrip()
                if not stripped and text.lstrip().startswith("[NOTICE]"):
                    continue
                if stripped != text:
                    kept.append({**part, "text": stripped})
                    continue
            kept.append(part)
        cleaned["content"] = kept
    return cleaned


def _append_control_state(message: dict[str, Any], content: str) -> None:
    previous = message.get("content")
    if isinstance(previous, str):
        message["content"] = f"{previous.rstrip()}{content}" if previous.strip() else content.lstrip("\n")
    elif isinstance(previous, list):
        message["content"] = [*previous, {"type": "text", "text": content, "_control_notice": True}]
    else:
        message["content"] = content.lstrip("\n")


@dataclass
class AgentSession:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    focused_tool_succeeded: bool = False
    tool_calls_used: int = 0
    invalid_actions: int = 0
    consecutive_invalid_actions: int = 0
    total_no_progress: int = 0
    consecutive_no_progress: int = 0
    successful_requests: list[ToolRequest] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        prompt: str,
        duration_seconds: float,
        has_audio: bool,
        initial_audio_attached: bool,
        initial: MediaBundle,
        settings: Mapping[str, Any],
        frame_contract: Mapping[str, Any],
    ) -> "AgentSession":
        user_text = build_initial_user_text(
            prompt=prompt,
            duration_seconds=duration_seconds,
            has_audio=has_audio,
            initial_audio_attached=initial_audio_attached,
            initial_header=initial.header,
            settings=settings,
        )
        return cls(
            tools=render_tools(settings, frame_contract),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        *[copy.copy(part) for part in initial.parts],
                    ],
                },
            ],
        )

    def append_assistant(self, message: Mapping[str, Any]) -> None:
        self.messages.append(copy.deepcopy(dict(message)))

    def append_tool_result(self, request: ToolRequest, bundle: MediaBundle) -> None:
        self._omit_visible_media()
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": request.call_id,
                "name": request.name,
                "content": [copy.copy(part) for part in bundle.parts],
            }
        )
        self.focused_tool_succeeded = True
        self.tool_calls_used += 1
        self.successful_requests.append(request)
        self.consecutive_invalid_actions = 0
        self.consecutive_no_progress = 0

    def append_protocol_error(
        self,
        code: str,
        message: str,
        *,
        calls: Sequence[Mapping[str, Any]] = (),
        max_invalid_retries: int,
    ) -> None:
        self.invalid_actions += 1
        self.consecutive_invalid_actions += 1
        content = render_error(
            code,
            message,
            retry_exceeded=self.consecutive_invalid_actions > max_invalid_retries,
            consecutive_invalid_actions=self.consecutive_invalid_actions,
            max_invalid_retries=max_invalid_retries,
        )
        if not calls:
            self.messages.append({"role": "user", "content": content})
            return
        for call in calls:
            function = call.get("function")
            name = str(function.get("name") or "") if isinstance(function, Mapping) else ""
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(call.get("id") or ""),
                    "name": name,
                    "content": content,
                }
            )

    def append_control_state(self, *, step: int, max_steps: int) -> None:
        cleaned_messages: list[dict[str, Any]] = []
        for message in self.messages:
            cleaned = _strip_control_state_from_message(message)
            if cleaned.get("role") == "user" and (
                cleaned.get("content") is None or cleaned.get("content") == ""
            ):
                continue
            cleaned_messages.append(cleaned)
        self.messages = cleaned_messages

        notice = render_notice(step=step, max_steps=max_steps, done=False)
        if self.messages and self.messages[-1].get("role") in {"user", "tool"}:
            _append_control_state(self.messages[-1], notice)
        else:
            self.messages.append({"role": "user", "content": notice.lstrip("\n")})

    def is_duplicate(self, request: ToolRequest) -> bool:
        return any(prior.identity == request.identity for prior in self.successful_requests)

    def append_no_progress(self, request: ToolRequest) -> None:
        self.total_no_progress += 1
        self.consecutive_no_progress += 1
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": request.call_id,
                "name": request.name,
                "content": (
                    f"[NO_PROGRESS] {request.name} repeats an identical successful call. "
                    "Use different evidence or answer from what is already verified."
                ),
            }
        )

    def _omit_visible_media(self) -> None:
        for index, message in enumerate(self.messages):
            has_images, has_other = _has_media(message.get("content"))
            if not (has_images or has_other):
                continue
            header = _first_text(message.get("content")).strip() or "Media content"
            suffix = f" Timestamps: [] {MEDIA_OMITTED}" if has_images else f" {MEDIA_OMITTED}"
            replaced = dict(message)
            replaced["content"] = header + suffix
            self.messages[index] = replaced


__all__ = [
    "CHAT_TEMPLATE",
    "DEFAULT_MEDIA_SETTINGS",
    "MEDIA_OMITTED",
    "SYSTEM_PROMPT",
    "VISUAL_HEADERS",
    "AgentSession",
    "MediaBundle",
    "ParsedAssistantTurn",
    "ProtocolError",
    "ToolRequest",
    "build_initial_user_text",
    "classify_turn_role",
    "extract_answer",
    "natural_seconds",
    "parse_generated_turn",
    "parse_seconds",
    "parse_tool_call",
    "render_error",
    "render_notice",
    "render_tools",
    "require_single_tool_call",
    "tool_max_seconds",
    "validate_assistant_text",
]
