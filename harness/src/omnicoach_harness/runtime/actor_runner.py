"""`ActorRunner`, which drives one episode: request, decode, tool call, repeat."""

from __future__ import annotations

import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any

from ..domain.episode import (
    ActorDecision,
    BudgetExhausted,
    EpisodeEvent,
    EpisodeResult,
    Failure,
    FinalAnswer,
    ToolCall,
    ToolFailure,
)
from ..domain.message import MediaRef, Message, serialize_message, text_message
from ..domain.task import TaskInput, TaskLabel
from ..providers.protocol import ModelClient, ModelRequest
from ..providers.telemetry import ProviderBudgetExhausted
from .route import (
    ActorRoute,
    ActorState,
    RouteProtocolError,
    RouteToolError,
    ToolContext,
    serialize_actor_state,
)


@dataclass(slots=True)
class _Run:
    """Mutable episode accumulator shared across exit paths."""

    task: TaskInput
    label: TaskLabel
    trajectory_id: str
    rollout_index: int
    root: Path
    state: ActorState = field(default_factory=ActorState)
    events: list[EpisodeEvent] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    canonical_steps: list[str] = field(default_factory=list)
    final_answer: str | None = None

    def record(self, kind: str, payload: dict[str, Any]) -> None:
        self.events.append(EpisodeEvent(index=len(self.events), kind=kind, payload=payload))


def _serialize_tool_media(item: MediaRef) -> dict[str, Any]:
    return {
        "kind": item.kind,
        "path": item.local_path,
        "uri": item.uri if item.uri is not None and not item.uri.startswith("data:") else None,
        "mime_type": item.mime_type,
        "label": item.label,
        "metadata": dict(item.metadata),
    }


@dataclass(slots=True, kw_only=True)
class ActorRunner:
    route: ActorRoute
    model: ModelClient
    model_name: str
    max_tokens: int
    sampling_seed: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    start_semaphore: BoundedSemaphore | None = None
    defer_scoring: bool = False

    def run(
        self,
        task: TaskInput,
        label: TaskLabel,
        *,
        rollout_index: int,
        work_dir: str | Path,
        media_cache_dir: str | Path | None = None,
    ) -> EpisodeResult:
        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)
        start_root = Path(media_cache_dir) if media_cache_dir is not None else root
        start_root.mkdir(parents=True, exist_ok=True)
        run = _Run(
            task=task,
            label=label,
            trajectory_id=str(uuid.uuid4()),
            rollout_index=rollout_index,
            root=root,
        )

        try:
            with self.start_semaphore if self.start_semaphore is not None else nullcontext():
                run.messages = self.route.start(task, work_dir=start_root, ffmpeg=self.ffmpeg, ffprobe=self.ffprobe)
        except Exception as exc:
            run.record("runtime_failure", {"code": "ACTOR_START_FAILED", "message": str(exc)})
            run.messages = []
            return self._result(run, status="error", error_code="ACTOR_START_FAILED", error_message=str(exc))

        while run.state.step < self.route.max_steps:
            run.messages.extend(self.route.control_messages(run.state))
            run.state.step += 1

            request = self._request(run)
            run.record("model_request", self._request_payload(run, request))
            try:
                response = self.model.complete(request)
            except ProviderBudgetExhausted:
                raise
            except Exception as exc:  # provider boundary; preserve as operational error
                run.record("model_failure", {"code": "MODEL_REQUEST_FAILED", "message": str(exc)})
                return self._result(run, status="error", error_code="MODEL_REQUEST_FAILED", error_message=str(exc))

            assistant = response.assistant_message or text_message("assistant", response.text)
            run.record(
                "model_response",
                {
                    "step": run.state.step,
                    "text": response.text,
                    "reasoning_content": assistant.reasoning_content,
                    "model": response.model,
                    "provider": response.provider,
                    "usage": dict(response.usage),
                    "metadata": dict(response.metadata),
                },
            )
            run.messages.append(assistant)

            try:
                decision = self.route.decode(response.text, run.state)
            except RouteProtocolError as exc:
                terminal = run.state.step >= self.route.max_steps or not exc.failure.retryable
                self._record_invalid(run, exc.failure, terminal=terminal)
                if terminal:
                    return self._failed(run, exc.failure)
                continue

            run.canonical_steps.append(response.text)
            if isinstance(decision.command, FinalAnswer):
                answered = self._answer(run, decision, decision.command)
                if answered is not None:
                    return answered
                continue

            call = decision.command
            run.record(
                "action",
                {
                    "step": run.state.step,
                    "observation": decision.observation,
                    "think": decision.think,
                    "command": {"type": call.name, **dict(call.arguments)},
                },
            )
            if run.state.step >= self.route.max_steps:
                failure = BudgetExhausted(message="must answer on final step")
                run.messages.append(self.route.tool_failure_message(call, failure, run.state, terminal=True))
                run.record("invalid_action", {"step": run.state.step, "code": failure.code, "message": failure.message})
                self.route.after_tool(run.messages)
                return self._failed(run, failure)

            terminal_failure = self._execute_tool(run, call)
            self.route.after_tool(run.messages)
            if terminal_failure is not None:
                return self._failed(run, terminal_failure)

        failure = BudgetExhausted()
        run.messages.append(self.route.render_failure(failure, run.state, terminal=True))
        return self._failed(run, failure)

    def _request(self, run: _Run) -> ModelRequest:
        return ModelRequest(
            messages=tuple(run.messages),
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.sampling_seed,
            response_format=self.route.response_format,
            metadata={
                "stage": "actor",
                "route_id": self.route.route_id,
                "task_id": run.task.task_id,
                "rollout_index": run.rollout_index,
                "sampling_seed": self.sampling_seed,
                "step": run.state.step,
            },
        )

    @staticmethod
    def _request_payload(run: _Run, request: ModelRequest) -> dict[str, Any]:
        return {
            "step": run.state.step,
            "messages": [serialize_message(item) for item in request.messages],
            "model": request.model,
            "max_tokens": request.max_tokens,
            "sampling": {"seed": request.seed, "temperature": request.temperature, "top_p": request.top_p},
            "response_format": dict(request.response_format) if request.response_format else None,
            "state_snapshot": serialize_actor_state(run.state),
            "canonical_steps": list(run.canonical_steps),
            "metadata": dict(request.metadata),
        }

    def _answer(self, run: _Run, decision: ActorDecision, answer: FinalAnswer) -> EpisodeResult | None:
        run.record(
            "action",
            {
                "step": run.state.step,
                "observation": decision.observation,
                "think": decision.think,
                "command": {"type": "answer", "content": answer.content},
            },
        )
        violation = self.route.validate_answer(decision, run.state, run.task)
        if violation is not None:
            terminal = run.state.step >= self.route.max_steps or not violation.retryable
            self._record_invalid(run, violation, terminal=terminal)
            return self._failed(run, violation) if terminal else None

        run.final_answer = answer.content
        if self.defer_scoring or self.route.defers_scoring(run.task):
            win, scoring = False, {"unjudged": True, "scoring_stage": "deferred"}
            run.messages.append(text_message("user", "ANSWER RECORDED; SCORING DEFERRED"))
        else:
            win = self.route.score(run.final_answer, run.label)
            scoring = {"unjudged": False, "scoring_stage": "online"}
            run.messages.append(text_message("user", "CORRECT ANSWER" if win else "INCORRECT ANSWER"))
        run.record("final_answer", {"step": run.state.step, "content": run.final_answer, "win": win, **scoring})
        return self._result(run, status="completed", done=True, win=win, final_answer=run.final_answer, scoring=scoring)

    def _execute_tool(self, run: _Run, call: ToolCall) -> Failure | None:
        try:
            ctx = ToolContext(
                task=run.task,
                work_dir=run.root,
                step=run.state.step,
                ffmpeg=self.ffmpeg,
                ffprobe=self.ffprobe,
                history=tuple(run.messages),
                state=run.state,
            )
            result = self.route.execute_tool(call, ctx)
        except RouteToolError as exc:
            terminal = not exc.failure.retryable
            run.state.invalid_actions += 1
            run.state.consecutive_invalid_actions += 1
            run.messages.append(self.route.tool_failure_message(call, exc.failure, run.state, terminal=terminal))
            run.record(
                "invalid_action",
                {"step": run.state.step, "code": exc.failure.code, "message": exc.failure.message},
            )
            return exc.failure if terminal else None
        except Exception as exc:
            failure = ToolFailure(code="TOOL_EXECUTION_FAILED", message=str(exc), retryable=True)
            run.messages.append(self.route.tool_failure_message(call, failure, run.state, terminal=False))
            run.record(
                "runtime_failure",
                {"step": run.state.step, "tool": call.name, "code": failure.code, "message": failure.message},
            )
            return None

        if call.call_id is not None and result.tool_call_id is None:
            result = replace(result, tool_call_id=call.call_id)
        run.messages.append(self.route.tool_message(result))
        run.state.successful_tool_calls += 1
        run.state.last_tool_call = call
        run.state.last_tool_result = result
        run.state.need_observation = True
        run.state.consecutive_invalid_actions = 0
        run.record(
            "tool_result",
            {
                "step": run.state.step,
                "tool": result.tool_name,
                "text": result.text,
                "media": [_serialize_tool_media(item) for item in result.media],
                "info": dict(result.info),
            },
        )
        return None

    def _record_invalid(self, run: _Run, failure: Failure, *, terminal: bool) -> None:
        run.state.invalid_actions += 1
        run.state.consecutive_invalid_actions += 1
        run.messages.append(self.route.render_failure(failure, run.state, terminal=terminal))
        run.record("invalid_action", {"step": run.state.step, "code": failure.code, "message": failure.message})

    def _failed(self, run: _Run, failure: Failure) -> EpisodeResult:
        return self._result(
            run,
            status="completed",
            done=True,
            final_answer=run.final_answer,
            error_code=failure.code,
            error_message=failure.message,
        )

    def _result(
        self,
        run: _Run,
        *,
        status: str,
        done: bool = False,
        win: bool = False,
        final_answer: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        scoring: dict[str, Any] | None = None,
    ) -> EpisodeResult:
        return EpisodeResult(
            task_id=run.task.task_id,
            trajectory_id=run.trajectory_id,
            route_id=self.route.route_id,
            rollout_index=run.rollout_index,
            status=status,  # type: ignore[arg-type]
            done=done,
            win=win,
            final_answer=final_answer,
            steps=run.state.step,
            events=tuple(run.events),
            final_history=tuple(run.messages),
            error_code=error_code,
            error_message=error_message,
            metadata={
                "successful_tool_calls": run.state.successful_tool_calls,
                "invalid_actions": run.state.invalid_actions,
                "sampling": {
                    "rollout_seed": self.sampling_seed,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                },
                **({"scoring": dict(scoring)} if scoring is not None else {}),
            },
        )
