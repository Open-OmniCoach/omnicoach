"""Actor decisions, tool results, failures and the `EpisodeResult` a run produces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .message import MediaRef, Message, serialize_message


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    call_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class FinalAnswer:
    content: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ActorDecision:
    command: ToolCall | FinalAnswer
    observation: str
    think: str
    raw: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolResult:
    tool_name: str
    text: str
    media: tuple[MediaRef, ...] = ()
    info: Mapping[str, Any] = field(default_factory=dict)
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ProtocolFailure:
    code: str
    message: str
    retryable: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolFailure:
    code: str
    message: str
    retryable: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class BudgetExhausted:
    code: Literal["STEP_LIMIT_REACHED"] = "STEP_LIMIT_REACHED"
    message: str = "step limit reached"


Failure = ProtocolFailure | ToolFailure | BudgetExhausted


@dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeEvent:
    index: int
    kind: Literal[
        "model_request",
        "model_response",
        "action",
        "tool_result",
        "invalid_action",
        "model_failure",
        "runtime_failure",
        "final_answer",
    ]
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeResult:
    task_id: str
    trajectory_id: str
    route_id: str
    rollout_index: int
    status: Literal["completed", "error"]
    done: bool
    win: bool
    final_answer: str | None
    steps: int
    events: tuple[EpisodeEvent, ...]
    final_history: tuple[Message, ...]
    error_code: str | None = None
    error_message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "trajectory_id": self.trajectory_id,
            "route_id": self.route_id,
            "rollout_index": self.rollout_index,
            "status": self.status,
            "done": self.done,
            "win": self.win,
            "final_answer": self.final_answer,
            "steps": self.steps,
            "events": [
                {"index": event.index, "kind": event.kind, "payload": dict(event.payload)} for event in self.events
            ],
            "final_history": [serialize_message(item) for item in self.final_history],
            "error_code": self.error_code,
            "error_message": self.error_message,
            "metadata": dict(self.metadata),
        }
