from __future__ import annotations

import json
import subprocess

from conftest import QueuedModelClient
from omnicoach_harness.domain.episode import ToolCall
from omnicoach_harness.domain.task import QuestionType, TaskInput, TaskLabel
from omnicoach_harness.protocol.route import build
from omnicoach_harness.runtime.actor_runner import ActorRunner
from omnicoach_harness.runtime.route import ToolContext


def test_real_ffmpeg_ordered_multi_span_smoke(tmp_path) -> None:
    video = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:size=160x120:rate=12:duration=4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-strict",
            "-2",
            "-shortest",
            str(video),
        ],
        check=True,
    )
    task = TaskInput(
        task_id="media-smoke",
        video_path=str(video),
        question="Compare the two intervals.",
        question_type=QuestionType.FREE_FORM,
        duration_seconds=4.0,
        has_audio=True,
    )
    result = build().execute_tool(
        ToolCall(
            name="get_multi_span",
            arguments={
                "spans": [
                    {"modality": "clip", "start": 0.25, "end": 1.25, "level": "detail"},
                    {
                        "modality": "frames",
                        "start": 2.0,
                        "end": 3.0,
                        "level": "coarse",
                        "resolution": "high",
                    },
                ]
            },
        ),
        ToolContext(task=task, work_dir=tmp_path / "tool", step=1, ffmpeg="ffmpeg"),
    )
    assert result.info["span_id_mode"] == "runtime_ordered"
    assert [item["id"] for item in result.info["spans"]] == ["span_1", "span_2"]
    assert result.info["total_frames"] == 3
    assert result.info["total_audio_seconds"] == 1.0
    assert all(item["result"].get("decoder") in {"decord", "pyav_fallback"} for item in result.info["spans"])
    assert any("span_1 | clip | 0.25–1.25 seconds" in str(item.label) for item in result.media)
    assert any("span_2 | frames | 2–3 seconds" in str(item.label) for item in result.media)
    assert "decoder" not in result.text
    assert "fps=" not in result.text


def test_fractional_tail_is_actor_visible_and_json_mode_is_sent(tmp_path) -> None:
    video = tmp_path / "fractional.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:size=80x60:rate=4:duration=9",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    task = TaskInput(
        task_id="fractional-tail",
        video_path=str(video),
        question="What is the dominant color?",
        question_type=QuestionType.MCQ,
        options=("A. blue", "B. red"),
        duration_seconds=8.75,
        has_audio=False,
    )
    model = QueuedModelClient(
        [
            json.dumps(
                {
                    "observation": "The global scan is blue.",
                    "think": "The initial evidence is sufficient.",
                    "action": {"type": "answer", "content": "A"},
                }
            )
        ]
    )
    result = ActorRunner(
        route=build(),
        model=model,
        model_name="queued",
        max_tokens=1024,
        sampling_seed=12345,
        temperature=1.0,
        top_p=0.95,
    ).run(task, TaskLabel(answer="A"), rollout_index=0, work_dir=tmp_path / "actor")
    assert result.win is True
    request = model.requests[0]
    request_events = [event for event in result.events if event.kind == "model_request"]
    assert len(request_events) == 1
    assert request_events[0].payload["messages"]
    initial_text = "\n".join(part.text or "" for item in request.messages for part in item.parts if part.type == "text")
    assert "duration: 00:08.8" in initial_text
    assert request.response_format["type"] == "json_schema"
    assert request.response_format["json_schema"]["strict"] is True
    assert request.seed == 12345
    assert request.temperature == 1.0
    assert request.top_p == 0.95
    assert request_events[0].payload["sampling"] == {
        "seed": 12345,
        "temperature": 1.0,
        "top_p": 0.95,
    }
    assert result.metadata["sampling"]["rollout_seed"] == 12345

    free_form = TaskInput(
        task_id="free-form",
        video_path=str(video),
        question="Name the dominant color.",
        question_type=QuestionType.FREE_FORM,
        duration_seconds=8.75,
        has_audio=False,
    )
    free_form_model = QueuedModelClient(
        [
            json.dumps(
                {
                    "observation": "The global scan is blue.",
                    "think": "The initial evidence is sufficient.",
                    "action": {"type": "answer", "content": "a blue shade"},
                }
            )
        ]
    )
    free_form_result = ActorRunner(
        route=build(),
        model=free_form_model,
        model_name="queued",
        max_tokens=1024,
    ).run(free_form, TaskLabel(answer="blue"), rollout_index=0, work_dir=tmp_path / "free-form-actor")
    assert free_form_result.win is False
    assert free_form_result.metadata["scoring"] == {"unjudged": True, "scoring_stage": "deferred"}
    final_event = next(event for event in free_form_result.events if event.kind == "final_answer")
    assert final_event.payload["unjudged"] is True
