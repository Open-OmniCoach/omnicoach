from __future__ import annotations

import json

import pytest

from conftest import QueuedModelClient
from omnicoach_harness.domain.episode import ActorDecision, FinalAnswer
from omnicoach_harness.domain.message import text_message
from omnicoach_harness.domain.task import QuestionType, TaskInput, TaskLabel
from omnicoach_harness.protocol import protocol
from omnicoach_harness.protocol.prompt import question_context
from omnicoach_harness.protocol.route import build
from omnicoach_harness.runtime.actor_runner import ActorRunner
from omnicoach_harness.runtime.route import ActorState, ToolContext


def _task(question_type: QuestionType, *, duration: float = 240.0) -> TaskInput:
    return TaskInput(
        task_id=f"test-{question_type.value}",
        video_path="unused.mp4",
        question="When does the event happen?",
        question_type=question_type,
        options=("A. first", "B. second") if question_type == QuestionType.MCQ else (),
        duration_seconds=duration,
        has_audio=False,
    )


def _answer(content: str) -> ActorDecision:
    return ActorDecision(command=FinalAnswer(content=content), observation="", think="done", raw="{}")


def test_tr_prompt_requires_supported_intervals_not_padded_points() -> None:
    prompt = question_context(_task(QuestionType.TEMPORAL_REASONING))

    assert "[01:23.4,01:27.8]" in prompt
    assert "[01:23.4,01:27.8]; [03:02.0,03:06.5]" in prompt
    assert "isolated point" in prompt
    assert "Do not invent fixed-width padding" in prompt
    assert "use the media tools" in prompt


def test_temporal_parser_accepts_only_complete_interval_lists() -> None:
    assert protocol.parse_temporal_intervals("[01:23.4,01:27.8]") == ((83.4, 87.8),)
    assert protocol.parse_temporal_intervals("[61:02.0,61:06.5]; [63:02.0,63:06.5]") == (
        (3662.0, 3666.5),
        (3782.0, 3786.5),
    )
    for invalid in ("01:23.4", "at [01:23.4,01:27.8]", "[01:23.4,01:27.8];"):
        with pytest.raises(ValueError):
            protocol.parse_temporal_intervals(invalid)


@pytest.mark.parametrize(
    "content",
    (
        "01:23.4",
        "at [01:23.4,01:27.8]",
        "[01:27.8,01:23.4]",
        "[01:23.4,01:23.4]",
        "[03:59.9,04:00.1]",
    ),
)
def test_tr_validator_rejects_point_prose_nonpositive_and_out_of_bounds(content: str) -> None:
    failure = build().validate_answer(
        _answer(content),
        ActorState(),
        _task(QuestionType.TEMPORAL_REASONING),
    )

    assert failure is not None
    assert failure.code == "INVALID_TEMPORAL_INTERVAL_ANSWER"
    assert failure.retryable is True


def test_validator_preserves_mcq_and_accepts_valid_tr() -> None:
    route = build()
    assert route.validate_answer(_answer("A"), ActorState(), _task(QuestionType.MCQ)) is None
    assert (
        route.validate_answer(
            _answer("[01:23.4,01:27.8]; [03:02.0,03:06.5]"),
            ActorState(),
            _task(QuestionType.TEMPORAL_REASONING),
        )
        is None
    )


def test_actor_runner_recovers_from_point_answer_before_accepting_interval(tmp_path) -> None:
    route = build()
    route.start = lambda task, **kwargs: [
        text_message("system", "system"),
        text_message("user", task.question),
    ]
    model = QueuedModelClient(
        [
            json.dumps(
                {
                    "observation": "The event is localized.",
                    "think": "Answer at the observed moment.",
                    "action": {"type": "answer", "content": "01:23.4"},
                }
            ),
            json.dumps(
                {
                    "observation": "The event has a supported onset and offset.",
                    "think": "Return the verified interval.",
                    "action": {"type": "answer", "content": "[01:22.8,01:24.1]"},
                }
            ),
        ]
    )

    result = ActorRunner(route=route, model=model, model_name="queued", max_tokens=128).run(
        _task(QuestionType.TEMPORAL_REASONING),
        TaskLabel(answer="[01:22.8,01:24.1]"),
        rollout_index=0,
        work_dir=tmp_path,
    )

    assert result.status == "completed"
    assert result.final_answer == "[01:22.8,01:24.1]"
    invalid = [event for event in result.events if event.kind == "invalid_action"]
    assert [event.payload["code"] for event in invalid] == ["INVALID_TEMPORAL_INTERVAL_ANSWER"]
    assert len(model.requests) == 2
    retry_context = "\n".join(
        part.text or ""
        for item in model.requests[1].messages
        for part in item.parts
        if part.type == "text"
    )
    assert "Do not pad a point" in retry_context


def test_duration_rounding_tolerance_accepts_displayed_timecode(tmp_path) -> None:
    route = build()
    assert route.route_id == "omni_ota_v1"
    assert route.response_format["json_schema"]["name"] == "observe_think_action"

    # format_timecode(8.75) displays 00:08.8; both answer validation and tool range must accept it,
    # while still rejecting 00:08.9.
    task = _task(QuestionType.TEMPORAL_REASONING, duration=8.75)
    assert route.validate_answer(_answer("[00:00.0,00:08.8]"), ActorState(), task) is None
    bad = route.validate_answer(_answer("[00:00.0,00:08.9]"), ActorState(), task)
    assert bad is not None and bad.code == "INVALID_TEMPORAL_INTERVAL_ANSWER"

    from omnicoach_harness.protocol.tools import _range
    from omnicoach_harness.runtime.route import RouteToolError

    ctx = ToolContext(task=task, work_dir=tmp_path, step=1, ffmpeg="ffmpeg")
    result = _range(
        name="get_frames",
        arguments={"start": 0.0, "end": 8.8},
        context=ctx,
        maximum=256.0,
        allow_point=True,
    )
    assert result == (0.0, 8.8)
    with pytest.raises(RouteToolError) as exc_info:
        _range(name="get_frames", arguments={"start": 0.0, "end": 8.9}, context=ctx, maximum=256.0, allow_point=True)
    assert exc_info.value.failure.code == "MEDIA_TS_OOB"
