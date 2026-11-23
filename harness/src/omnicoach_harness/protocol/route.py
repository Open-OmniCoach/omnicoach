"""`OmniOTARoute` and its `build()` entrypoint."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..artifacts.schema import ActorContractArtifact
from ..domain.episode import ActorDecision, Failure, FinalAnswer, ProtocolFailure, ToolCall, ToolResult
from ..domain.message import Message, media_part, message, text_message, text_part
from ..domain.task import QuestionType, TaskInput, TaskLabel
from ..media.timecode import format_timecode
from ..runtime.route import ActorState, ToolContext
from . import protocol
from .contract import CONTRACT, ROUTE_ID
from .history import retain_latest_media
from .media import build_initial_message
from .prompt import SYSTEM_PROMPT
from .tools import OmniOTATools, tool_specs


class OmniOTARoute:
    route_id = ROUTE_ID
    max_steps = CONTRACT.max_steps
    response_format = CONTRACT.response_format

    def __init__(self) -> None:
        self._tools = OmniOTATools()

    def contract_artifact(self) -> ActorContractArtifact:
        return ActorContractArtifact(
            route_id=self.route_id,
            snapshot=CONTRACT.snapshot(),
            public_protocol={
                "system_prompt": SYSTEM_PROMPT,
                "tool_specs": tool_specs(),
                "response_schema": protocol.response_schema(),
                "implementation": CONTRACT.implementation_snapshot(),
            },
        )

    def start(
        self,
        task: TaskInput,
        *,
        work_dir: Path,
        ffmpeg: str,
        ffprobe: str = "ffprobe",
    ) -> list[Message]:
        return [
            text_message("system", SYSTEM_PROMPT),
            build_initial_message(task, work_dir=work_dir, ffmpeg=ffmpeg, ffprobe=ffprobe),
        ]

    def control_messages(self, state: ActorState) -> tuple[Message, ...]:
        if state.need_observation and state.last_tool_call is not None:
            call = state.last_tool_call
            start, end = call.arguments.get("start"), call.arguments.get("end")
            span = (
                f"[{format_timecode(float(start))},{format_timecode(float(end))}]"
                if isinstance(start, (int, float)) and isinstance(end, (int, float))
                else ""
            )
            level = call.arguments.get("level")
            resolution = call.arguments.get("resolution")
            suffix = f" level={level}" if isinstance(level, str) else ""
            suffix += f" resolution={resolution}" if resolution == "high" else ""
            parser_state = (
                f"[PARSER_STATE] The latest tool result was {call.name} {span}{suffix}. In the next JSON, "
                "observation must summarize that result and cite at least one [MM:SS.m,MM:SS.m] span. "
                "Then provide think and exactly one action."
            )
            if call.name == "get_multi_span":
                parser_state += (
                    " For every returned span id, record the observed fact or uncertainty; then compare the "
                    "ids and state any missing answer-critical evidence."
                )
        else:
            parser_state = (
                "[PARSER_STATE] Return raw JSON with observation, think, and exactly one action. All start/end "
                "values must be MM:SS.m strings with exactly one decimal."
            )
        remaining = self.max_steps - state.step
        notice = (
            "[NOTICE] FINAL STEP! You MUST provide your answer now."
            if remaining == 1
            else f"[NOTICE] Step {state.step}/{self.max_steps}. {remaining} steps remaining."
        )
        return text_message("user", parser_state), text_message("user", notice)

    def decode(self, raw: str, state: ActorState) -> ActorDecision:
        return protocol.decode(raw, state)

    def validate_answer(
        self,
        decision: ActorDecision,
        state: ActorState,
        task: TaskInput,
    ) -> ProtocolFailure | None:
        del state
        # Direct-first is a named V4.5 behavior, including after invalid turns.
        if task.question_type != QuestionType.TEMPORAL_REASONING:
            return None
        if not isinstance(decision.command, FinalAnswer):
            return None
        required_format = (
            "Use exactly [MM:SS.m,MM:SS.m], or semicolon-separated intervals, with start < end and "
            "no other text. Do not pad a point; use a media tool if onset or offset is uncertain."
        )
        try:
            intervals = protocol.parse_temporal_intervals(decision.command.content)
        except ValueError as exc:
            return ProtocolFailure(
                code="INVALID_TEMPORAL_INTERVAL_ANSWER",
                message=f"{exc}. {required_format}",
                retryable=True,
            )
        for start, end in intervals:
            if start >= end:
                return ProtocolFailure(
                    code="INVALID_TEMPORAL_INTERVAL_ANSWER",
                    message=f"Interval [{start:.1f},{end:.1f}] must have positive width. {required_format}",
                    retryable=True,
                )
            if task.duration_seconds is not None:
                max_end = max(task.duration_seconds, round(task.duration_seconds * 10) / 10.0) + 1e-6
                if end > max_end:
                    return ProtocolFailure(
                        code="INVALID_TEMPORAL_INTERVAL_ANSWER",
                        message=(
                            f"Interval end {end:.1f}s exceeds video duration {task.duration_seconds:.1f}s. "
                            f"{required_format}"
                        ),
                        retryable=True,
                    )
        return None

    def execute_tool(self, call: ToolCall, context: ToolContext) -> ToolResult:
        return self._tools.execute(call, context)

    def tool_message(self, result: ToolResult) -> Message:
        parts = [text_part(result.text)]
        for media in result.media:
            if media.label:
                parts.append(text_part(media.label))
            parts.append(media_part(replace(media, label=None)))
        return message("tool", *parts)

    def tool_failure_message(
        self,
        call: ToolCall,
        failure: Failure,
        state: ActorState,
        *,
        terminal: bool,
    ) -> Message:
        del call
        return self.render_failure(failure, state, terminal=terminal)

    def after_tool(self, messages: list[Message]) -> None:
        retain_latest_media(messages, keep_recent=CONTRACT.keep_recent_media)

    def render_failure(self, failure: Failure, state: ActorState, *, terminal: bool) -> Message:
        del state
        detail = f"{failure.code}: {failure.message}" if failure.message != failure.code else failure.code
        if not terminal:
            detail += (
                "\n[SYSTEM] The previous action was invalid. Analyze the error diagnostic above, "
                "resolve the conflict, and issue a corrected command to avoid wasting steps."
            )
        return text_message("user", f"[ERROR] {detail}")

    def defers_scoring(self, task: TaskInput) -> bool:
        return task.question_type in {QuestionType.FREE_FORM, QuestionType.TEMPORAL_REASONING}

    def score(self, answer: str, label: TaskLabel) -> bool:
        expected = str(label.answer or "").strip()
        prediction = str(answer or "").strip()
        if not expected:
            return False
        if len(expected) == 1 and expected.isalpha():
            return prediction.upper() == expected.upper()
        return prediction.casefold() == expected.casefold()


def build() -> OmniOTARoute:
    return OmniOTARoute()
