"""End-to-end actor CLI execution and P0/P1 regression tests (no external ffmpeg required)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import av
import pytest
from PIL import Image

from conftest import QueuedModelClient
from omnicoach_harness import cli
from omnicoach_harness.artifacts.store import iter_jsonl, read_json
from omnicoach_harness.domain.episode import ToolCall
from omnicoach_harness.domain.message import MediaRef, media_part, message, text_message
from omnicoach_harness.domain.task import QuestionType, TaskInput, TaskLabel
from omnicoach_harness.media import executor as media_executor
from omnicoach_harness.media import extract as media_extract
from omnicoach_harness.media.probe import VideoInfo
from omnicoach_harness.protocol import media as actor_media
from omnicoach_harness.protocol import tools as actor_tools
from omnicoach_harness.protocol.route import build
from omnicoach_harness.providers import openai_compat, telemetry
from omnicoach_harness.providers.messages import to_provider_messages
from omnicoach_harness.providers.protocol import ModelRequest, ModelResponse
from omnicoach_harness.runtime.actor_runner import ActorRunner
from omnicoach_harness.runtime.batch_runner import ActorBatchSession
from omnicoach_harness.runtime.config import BatchOptions, MediaOptions, RunConfig
from omnicoach_harness.runtime.route import RouteToolError, ToolContext


def _make_synthetic_video(path: Path, *, width: int = 160, height: int = 120, fps: int = 4, seconds: int = 6) -> Path:
    """Create a real MP4 via PyAV (no external ffmpeg subprocess)."""

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for frame_idx in range(fps * seconds):
            img = Image.new("RGB", (width, height), ((frame_idx * 17) % 256, 60, (255 - frame_idx * 11) % 256))
            frame = av.VideoFrame.from_image(img)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_cli_actor_end_to_end_without_mocking_session_execute(tmp_path, monkeypatch) -> None:
    """P0 #1 regression: cli._actor -> run_actor_batch -> ActorBatchSession.execute -> ActorRunner.run -> JSONL."""

    video = _make_synthetic_video(tmp_path / "clip.mp4", seconds=4)
    fake_info = VideoInfo(duration_seconds=4.0, width=160, height=120, fps=4.0, has_audio=False)
    monkeypatch.setattr(media_extract, "probe_video", lambda _p, *, ffprobe="ffprobe": fake_info)

    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        json.dumps(
            {
                "task_id": "mcq-1",
                "video_path": str(video),
                "question": "Pick A or B.",
                "question_type": "mcq",
                "options": ["A. first", "B. second"],
                "answer": "A",
                "duration_seconds": 4.0,
                "has_audio": False,
            }
        )
        + "\n"
        + json.dumps(
            {
                "task_id": "tr-1",
                "video_path": str(video),
                "question": "When does the color change?",
                "question_type": "TR",
                "answer": "[00:01.0,00:02.0]",
                "duration_seconds": 4.0,
                "has_audio": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    config_path = tmp_path / "actor.yaml"
    config_path.write_text(
        f"""stage: actor
route:
  entrypoint: omnicoach_harness.protocol.route:build
io:
  input: {tasks_path}
  output_dir: {out_dir}
provider:
  backend: openai_compat
  base_url: http://127.0.0.1:8000/v1
  model: fake-model
  max_tokens: 256
execution:
  workers: 1
  rollouts: 1
  attempts: 1
sampling:
  base_seed: 42
  temperature: 0.0
""",
        encoding="utf-8",
    )

    responses = {
        "mcq-1": [
            json.dumps(
                {
                    "observation": "The initial scan shows a short color gradient clip.",
                    "think": "Check frames 00:00.0 to 00:02.0 before answering.",
                    "action": {"type": "get_frames", "start": "00:00.0", "end": "00:02.0", "level": "coarse"},
                }
            ),
            json.dumps(
                {
                    "observation": "In [00:00.0,00:02.0], the frames confirm option A.",
                    "think": "Ready to answer.",
                    "action": {"type": "answer", "content": "A"},
                }
            ),
        ],
        "tr-1": [
            json.dumps(
                {
                    "observation": "The initial scan covers 00:00.0 to 00:04.0.",
                    "think": "Report the supported interval directly.",
                    "action": {"type": "answer", "content": "[00:01.0,00:02.0]"},
                }
            )
        ],
    }

    class TaskRoutingClient:
        def complete(self, request: ModelRequest) -> ModelResponse:
            task_id = str(request.metadata["task_id"])
            return ModelResponse(text=responses[task_id].pop(0), model=request.model, provider="fake")

    monkeypatch.setattr(cli, "_model_factory", lambda _cfg: lambda: TaskRoutingClient())

    exit_code = cli._actor(RunConfig.load(config_path))
    assert exit_code == 0

    rows = {row["run_key"]: row for row in iter_jsonl(out_dir / "actor_runs.jsonl")}
    assert set(rows) == {"mcq-1#r0", "tr-1#r0"}
    assert rows["mcq-1#r0"]["episode"]["status"] == "completed"
    assert rows["mcq-1#r0"]["episode"]["win"] is True
    assert rows["mcq-1#r0"]["episode"]["final_answer"] == "A"
    assert rows["mcq-1#r0"]["episode"]["metadata"]["successful_tool_calls"] == 1
    assert rows["tr-1#r0"]["episode"]["status"] == "completed"
    assert rows["tr-1#r0"]["episode"]["metadata"]["scoring"] == {"unjudged": True, "scoring_stage": "deferred"}
    assert (out_dir / "actor_contract.json").is_file()
    assert (out_dir / "actor_sampling_contract.json").is_file()
    assert not (out_dir / "subagent_runs.jsonl").exists()
    assert "feedback_ref" not in rows["mcq-1#r0"]
    assert read_json(out_dir / "actor_contract.json")["route_id"] == "omni_ota_v1"


def test_retry_isolates_attempt_work_dir_and_never_returns_stale_media(tmp_path, monkeypatch) -> None:
    """P0 #2a regression: retrying the same run_key isolates attempt dirs and never returns stale audio bytes."""

    route = build()
    route.start = lambda task, **kwargs: [text_message("system", "sys"), text_message("user", task.question)]

    recorded_audio_paths: list[Path] = []

    def fake_extract_audio(
        video_path: str, *, start: float, end: float, output_path: str | Path, ffmpeg: str = "ffmpeg"
    ) -> Path:
        del video_path, ffmpeg
        target = Path(output_path)
        if target.is_file() and target.stat().st_size > 0:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"audio:{start:.1f}-{end:.1f}".encode())
        recorded_audio_paths.append(target)
        return target

    monkeypatch.setattr(actor_tools, "extract_audio", fake_extract_audio)

    attempt_clients = [
        # Attempt 0: step 1 calls get_audio(00:00.0, 00:02.0), step 2 fails with provider error
        _FailOnSecondCallClient(
            json.dumps(
                {
                    "observation": "Check early audio.",
                    "think": "Need early audio span.",
                    "action": {"type": "get_audio", "start": "00:00.0", "end": "00:02.0"},
                }
            )
        ),
        # Attempt 1: step 1 calls get_audio(00:04.0, 00:06.0), step 2 answers A
        QueuedModelClient(
            [
                json.dumps(
                    {
                        "observation": "Check later audio.",
                        "think": "Need later audio span.",
                        "action": {"type": "get_audio", "start": "00:04.0", "end": "00:06.0"},
                    }
                ),
                json.dumps(
                    {
                        "observation": "In [00:04.0,00:06.0], speech confirms A.",
                        "think": "Answer A.",
                        "action": {"type": "answer", "content": "A"},
                    }
                ),
            ]
        ),
    ]

    session = ActorBatchSession(
        route=route,
        model_factory=lambda: attempt_clients.pop(0),
        model_name="fake",
        max_tokens=128,
        options=BatchOptions(output_dir=tmp_path / "out", attempts=2, retry_errors=True),
    )
    task = TaskInput(
        task_id="retry-task",
        video_path="unused.mp4",
        question="Which option?",
        question_type=QuestionType.MCQ,
        options=("A. yes", "B. no"),
        duration_seconds=10.0,
        has_audio=True,
    )
    result = session.execute((task, TaskLabel(answer="A"), 0), rollouts=1, attempts=2, retry_errors=True)

    assert result["episode"]["status"] == "completed"
    assert result["episode"]["win"] is True
    assert len(recorded_audio_paths) == 2
    assert recorded_audio_paths[0].parent != recorded_audio_paths[1].parent
    assert recorded_audio_paths[0].parent.name.startswith("attempt_00_")
    assert recorded_audio_paths[1].parent.name.startswith("attempt_01_")

    tool_event = next(ev for ev in result["episode"]["events"] if ev["kind"] == "tool_result")
    final_audio = Path(tool_event["payload"]["media"][0]["path"])
    assert final_audio == recorded_audio_paths[1]
    assert final_audio.read_bytes() == b"audio:4.0-6.0"


class _FailOnSecondCallClient:
    def __init__(self, first_output: str) -> None:
        self._first = first_output
        self._calls = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        self._calls += 1
        if self._calls == 1:
            return ModelResponse(text=self._first, model=request.model, provider="fake")
        raise RuntimeError("transient provider disconnect")


def test_same_step_different_tool_args_never_collide_or_return_stale_media(tmp_path, monkeypatch) -> None:
    """P0 #2b: same work_dir + step with changed audio/frame/resolution args never returns stale media."""

    video = _make_synthetic_video(tmp_path / "sample.mp4", width=160, height=120, fps=4, seconds=6)
    fake_info = VideoInfo(duration_seconds=6.0, width=160, height=120, fps=4.0, has_audio=True)
    monkeypatch.setattr(media_extract, "probe_video", lambda _p, *, ffprobe="ffprobe": fake_info)

    def fake_extract_audio(
        video_path: str, *, start: float, end: float, output_path: str | Path, ffmpeg: str = "ffmpeg"
    ) -> Path:
        del video_path, ffmpeg
        target = Path(output_path)
        if target.is_file() and target.stat().st_size > 0:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"audio:{start:.1f}-{end:.1f}".encode())
        return target

    monkeypatch.setattr(actor_tools, "extract_audio", fake_extract_audio)

    route = build()
    task = TaskInput(
        task_id="same-step",
        video_path=str(video),
        question="Inspect.",
        question_type=QuestionType.FREE_FORM,
        duration_seconds=6.0,
        has_audio=True,
    )
    shared_dir = tmp_path / "shared_work"
    ctx = ToolContext(task=task, work_dir=shared_dir, step=1, ffmpeg="ffmpeg")

    a1 = route.execute_tool(ToolCall(name="get_audio", arguments={"start": 0.0, "end": 2.0}), ctx)
    a2 = route.execute_tool(ToolCall(name="get_audio", arguments={"start": 3.0, "end": 5.0}), ctx)
    p1, p2 = Path(a1.media[0].local_path or ""), Path(a2.media[0].local_path or "")
    assert p1 != p2
    assert p1.read_bytes() == b"audio:0.0-2.0"
    assert p2.read_bytes() == b"audio:3.0-5.0"

    f_default = route.execute_tool(
        ToolCall(name="get_frames", arguments={"start": 0.0, "end": 2.0, "level": "coarse", "resolution": "default"}),
        ctx,
    )
    f_high = route.execute_tool(
        ToolCall(name="get_frames", arguments={"start": 0.0, "end": 2.0, "level": "coarse", "resolution": "high"}),
        ctx,
    )
    assert f_default.media[0].local_path != f_high.media[0].local_path

    # Direct extract_frame_batch calls on the same output_dir with different max_side also stay distinct.
    batch_dir = tmp_path / "shared_batch"
    b_small = media_extract.extract_frame_batch(str(video), timestamps=[0.5], output_dir=batch_dir, max_side=64)
    b_large = media_extract.extract_frame_batch(str(video), timestamps=[0.5], output_dir=batch_dir, max_side=128)
    assert b_small.paths[0] != b_large.paths[0]
    with Image.open(b_small.paths[0]) as small_img, Image.open(b_large.paths[0]) as large_img:
        assert max(small_img.size) == 64
        assert max(large_img.size) == 128


def test_custom_ffprobe_propagates_through_session_runner_and_tools(tmp_path, monkeypatch) -> None:
    """Config ffprobe must reach probe_video during both route.start and route.execute_tool."""

    video = _make_synthetic_video(tmp_path / "probe.mp4", seconds=3)
    seen_ffprobe: list[str] = []
    fake_info = VideoInfo(duration_seconds=3.0, width=160, height=120, fps=4.0, has_audio=False)

    def spy_probe(_path: str, *, ffprobe: str = "ffprobe") -> VideoInfo:
        seen_ffprobe.append(ffprobe)
        return fake_info

    monkeypatch.setattr(actor_media, "probe_video", spy_probe)
    monkeypatch.setattr(actor_tools, "probe_video", spy_probe)
    monkeypatch.setattr(media_extract, "probe_video", spy_probe)

    model = QueuedModelClient(
        [
            json.dumps(
                {
                    "observation": "Initial scan done.",
                    "think": "Inspect frames.",
                    "action": {"type": "get_frames", "start": "00:00.0", "end": "00:01.0", "level": "coarse"},
                }
            ),
            json.dumps(
                {
                    "observation": "Checked [00:00.0,00:01.0].",
                    "think": "Done.",
                    "action": {"type": "answer", "content": "A"},
                }
            ),
        ]
    )
    session = ActorBatchSession(
        route=build(),
        model_factory=lambda: model,
        model_name="fake",
        max_tokens=128,
        options=BatchOptions(output_dir=tmp_path / "out"),
        media=MediaOptions(ffprobe="/custom/bin/ffprobe"),
    )
    task = TaskInput(
        task_id="probe-test",
        video_path=str(video),
        question="Choose.",
        question_type=QuestionType.MCQ,
        options=("A. one", "B. two"),
        duration_seconds=None,
        has_audio=None,
    )
    result = session.execute((task, TaskLabel(answer="A"), 0), rollouts=1, attempts=1, retry_errors=False)
    assert result["episode"]["status"] == "completed"
    assert len(seen_ffprobe) >= 3
    assert set(seen_ffprobe) == {"/custom/bin/ffprobe"}


def test_clip_validates_audio_duration_before_extracting_frames(tmp_path, monkeypatch) -> None:
    """_clip() must fail fast on AUDIO_TOO_SHORT before decoding frames."""

    extract_calls = 0

    def fail_if_called(*args, **kwargs):
        nonlocal extract_calls
        extract_calls += 1
        raise AssertionError("extract_frame_batch should not be called before audio validation")

    monkeypatch.setattr(actor_tools, "extract_frame_batch", fail_if_called)

    task = TaskInput(
        task_id="clip-short",
        video_path="unused.mp4",
        question="Check.",
        question_type=QuestionType.FREE_FORM,
        duration_seconds=10.0,
        has_audio=True,
    )
    ctx = ToolContext(task=task, work_dir=tmp_path, step=1, ffmpeg="ffmpeg")
    with pytest.raises(RouteToolError) as exc_info:
        build().execute_tool(
            ToolCall(name="get_clip", arguments={"start": 1.0, "end": 1.05, "level": "detail"}),
            ctx,
        )
    assert exc_info.value.failure.code == "AUDIO_TOO_SHORT"
    assert extract_calls == 0


def test_openai_compat_encodes_once_bounds_429_and_enforces_budget(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(openai_compat.time, "sleep", lambda _s: None)
    telemetry.reset()

    image_path = tmp_path / "img.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image_path, format="JPEG")
    req = ModelRequest(
        messages=(
            message(
                "user",
                media_part(MediaRef(kind="image", local_path=str(image_path), mime_type="image/jpeg")),
            ),
        ),
        model="test-model",
        max_tokens=32,
    )

    encode_calls = 0
    real_encode = openai_compat.to_provider_messages

    def counting_encode(messages, **kwargs):
        nonlocal encode_calls
        encode_calls += 1
        return real_encode(messages, **kwargs)

    monkeypatch.setattr(openai_compat, "to_provider_messages", counting_encode)

    captured_kwargs: list[dict] = []

    class FlakySDK:
        def __init__(self) -> None:
            self.calls = 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            captured_kwargs.append(kwargs)
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("temporary 503")
            return SimpleNamespace(
                id="resp-1",
                model="test-model",
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[], reasoning_content=None))],
                usage=None,
            )

    client = openai_compat.OpenAICompatModelClient(
        base_urls="http://127.0.0.1:8000/v1",
        retries=3,
        clients=[FlakySDK()],
    )
    assert client.extra_body == {}
    resp = client.complete(req)
    assert resp.text == "ok"
    assert encode_calls == 1
    assert "extra_body" not in captured_kwargs[0]
    counts = telemetry.snapshot()
    assert counts["test-model:attempt"] == 3
    assert counts["test-model:error"] == 2
    assert counts["test-model:success"] == 1

    # 429 retries are bounded by max_429_retries, and non-standard negative codes are not treated as 429.
    assert not openai_compat._is_429(RuntimeError("gateway error -9999"))

    class RateLimitedSDK:
        def __init__(self) -> None:
            self.calls = 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            del kwargs
            self.calls += 1
            err = RuntimeError("error code: 429")
            err.status_code = 429
            raise err

    rl_sdk = RateLimitedSDK()
    rl_client = openai_compat.OpenAICompatModelClient(
        base_urls="http://127.0.0.1:8000/v1",
        retries=2,
        retry_429_forever=True,
        max_429_retries=3,
        clients=[rl_sdk],
    )
    with pytest.raises(RuntimeError, match="429"):
        rl_client.complete(req)
    assert rl_sdk.calls == 3

    # Provider attempt budget stops immediately without retrying.
    telemetry.reset()
    telemetry.configure_budget(attempts=1)
    budget_sdk = FlakySDK()
    budget_client = openai_compat.OpenAICompatModelClient(
        base_urls="http://127.0.0.1:8000/v1",
        retries=5,
        clients=[budget_sdk],
    )
    with pytest.raises(telemetry.ProviderBudgetExhausted):
        budget_client.complete(req)
    assert budget_sdk.calls == 1
    telemetry.reset()


def test_media_executor_close_resets_singleton_and_single_frame_weight(monkeypatch) -> None:
    fake_info = VideoInfo(duration_seconds=5.0, width=4000, height=3000, fps=30.0, has_audio=False)
    monkeypatch.setattr("omnicoach_harness.media.probe.probe_video", lambda _p, *, ffprobe="ffprobe": fake_info)

    # 4000*3000*9 = 108 MB per frame: fits in 200 MB for 1 frame, would fail if multiplied by 16 (1.7 GB).
    ex = media_executor.configure_media_executor(workers=1, queue_size=0, decode_bytes=200 * 1024**2)
    try:
        single_weight = ex._estimate_weight(
            media_extract.extract_frame.__wrapped__,
            ("video.mp4",),
            {"timestamp": 1.0, "output_path": "out.jpg"},
        )
        assert single_weight == 4000 * 3000 * 9
    finally:
        ex.close()

    # Re-configuring after close() succeeds because close() resets _executor = None.
    ex2 = media_executor.configure_media_executor(workers=1, queue_size=0, decode_bytes=200 * 1024**2)
    ex2.close()
    assert media_executor._executor is None


def test_to_provider_messages_audio_part_types(tmp_path) -> None:
    audio_file = tmp_path / "clip.m4a"
    audio_file.write_bytes(b"fake-m4a-bytes")
    msg = (message("user", media_part(MediaRef(kind="audio", local_path=str(audio_file), mime_type="audio/mp4"))),)

    default_rendered = to_provider_messages(msg)
    assert default_rendered[0]["content"][0]["type"] == "audio_url"
    assert default_rendered[0]["content"][0]["audio_url"]["url"].startswith("data:audio/mp4;base64,")

    input_rendered = to_provider_messages(msg, audio_part_type="input_audio")
    assert input_rendered[0]["content"][0]["type"] == "input_audio"
    assert input_rendered[0]["content"][0]["input_audio"]["format"] == "m4a"
    assert input_rendered[0]["content"][0]["input_audio"]["data"]

    legacy_rendered = to_provider_messages(msg, audio_part_type="image_url")
    assert legacy_rendered[0]["content"][0]["type"] == "image_url"
    assert legacy_rendered[0]["content"][0]["image_url"]["url"].startswith("data:audio/mp4;base64,")

    with pytest.raises(ValueError, match="unsupported audio_part_type"):
        to_provider_messages(msg, audio_part_type="invalid_part")

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """stage: actor
route: {entrypoint: omnicoach_harness.protocol.route:build}
io: {input: input.json, output_dir: out}
provider: {model: fake, base_url: http://127.0.0.1:8000/v1, audio_part_type: input_audio}
""",
        encoding="utf-8",
    )
    client = cli._model_factory(RunConfig.load(cfg_path))()
    assert client.audio_part_type == "input_audio"


def test_actor_runner_propagates_provider_budget_exhausted(tmp_path) -> None:
    route = build()
    route.start = lambda task, **kwargs: [text_message("system", "sys"), text_message("user", task.question)]

    class BudgetClient:
        def complete(self, request: ModelRequest) -> ModelResponse:
            del request
            raise telemetry.ProviderBudgetExhausted("budget exhausted")

    task = TaskInput(
        task_id="budget-task",
        video_path="unused.mp4",
        question="Pick one.",
        question_type=QuestionType.MCQ,
        options=("A. one", "B. two"),
        duration_seconds=5.0,
        has_audio=False,
    )
    runner = ActorRunner(route=route, model=BudgetClient(), model_name="fake", max_tokens=64)
    with pytest.raises(telemetry.ProviderBudgetExhausted, match="budget exhausted"):
        runner.run(task, TaskLabel(answer="A"), rollout_index=0, work_dir=tmp_path)


def test_pyav_decode_failure_cleans_up_temp_files(tmp_path, monkeypatch) -> None:
    video = _make_synthetic_video(tmp_path / "pyav_cleanup.mp4", seconds=2)
    out_dir = tmp_path / "frames"
    out_dir.mkdir()
    target_paths = (out_dir / "f1.jpg", out_dir / "f2.jpg")
    original_save = media_extract._save_pyav_frame
    calls = 0

    def fail_on_second_save(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected save failure")
        return original_save(*args, **kwargs)

    monkeypatch.setattr(media_extract, "_save_pyav_frame", fail_on_second_save)
    with pytest.raises(RuntimeError, match="injected save failure"):
        media_extract._decode_frames_with_pyav(
            str(video),
            timestamps=[0.2, 0.8],
            output_size=(80, 60),
            resize_after_decode=False,
            decode_threads=1,
            paths=target_paths,
            quality=90,
        )
    assert list(out_dir.iterdir()) == []
