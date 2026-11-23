"""The `ActorRoute` protocol, `ActorState`, `ToolContext`, and state serialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..artifacts.schema import ActorContractArtifact
from ..domain.episode import ActorDecision, Failure, ProtocolFailure, ToolCall, ToolResult
from ..domain.message import Message
from ..domain.task import TaskInput, TaskLabel


@dataclass(slots=True)
class ActorState:
    step: int = 0
    successful_tool_calls: int = 0
    invalid_actions: int = 0
    consecutive_invalid_actions: int = 0
    need_observation: bool = False
    last_tool_call: ToolCall | None = None
    last_tool_result: ToolResult | None = None
    route_data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolContext:
    task: TaskInput
    work_dir: Path
    step: int
    ffmpeg: str
    ffprobe: str = "ffprobe"
    history: tuple[Message, ...] = ()
    state: ActorState | None = None


class RouteProtocolError(Exception):
    def __init__(self, failure: ProtocolFailure) -> None:
        super().__init__(f"{failure.code}: {failure.message}")
        self.failure = failure


class RouteToolError(Exception):
    def __init__(self, failure: Failure) -> None:
        super().__init__(f"{failure.code}: {failure.message}")
        self.failure = failure


class ActorRoute(Protocol):
    route_id: str
    max_steps: int
    response_format: Mapping[str, Any] | None

    def contract_artifact(self) -> ActorContractArtifact: ...

    def start(
        self,
        task: TaskInput,
        *,
        work_dir: Path,
        ffmpeg: str,
        ffprobe: str = "ffprobe",
    ) -> list[Message]: ...

    def control_messages(self, state: ActorState) -> tuple[Message, ...]: ...

    def decode(self, raw: str, state: ActorState) -> ActorDecision: ...

    def validate_answer(
        self,
        decision: ActorDecision,
        state: ActorState,
        task: TaskInput,
    ) -> ProtocolFailure | None: ...

    def execute_tool(self, call: ToolCall, context: ToolContext) -> ToolResult: ...

    def tool_message(self, result: ToolResult) -> Message: ...

    def tool_failure_message(
        self,
        call: ToolCall,
        failure: Failure,
        state: ActorState,
        *,
        terminal: bool,
    ) -> Message: ...

    def after_tool(self, messages: list[Message]) -> None: ...

    def render_failure(self, failure: Failure, state: ActorState, *, terminal: bool) -> Message: ...

    def defers_scoring(self, task: TaskInput) -> bool: ...

    def score(self, answer: str, label: TaskLabel) -> bool: ...


def serialize_actor_state(state: ActorState) -> dict[str, Any]:
    return {
        "step": state.step,
        "successful_tool_calls": state.successful_tool_calls,
        "invalid_actions": state.invalid_actions,
        "consecutive_invalid_actions": state.consecutive_invalid_actions,
        "need_observation": state.need_observation,
        "last_tool_call": _serialize_call(state.last_tool_call),
        "last_tool_result": _serialize_result(state.last_tool_result),
        "route_data": dict(state.route_data),
    }


def _serialize_call(value: ToolCall | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {"name": value.name, "arguments": dict(value.arguments), "call_id": value.call_id}


def _serialize_result(value: ToolResult | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "tool_name": value.tool_name,
        "text": value.text,
        "media": [
            {
                "kind": item.kind,
                "local_path": item.local_path,
                "uri": item.uri,
                "mime_type": item.mime_type,
                "label": item.label,
                "metadata": dict(item.metadata),
            }
            for item in value.media
        ],
        "info": dict(value.info),
        "tool_call_id": value.tool_call_id,
    }
