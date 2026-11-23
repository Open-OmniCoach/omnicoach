"""The system prompt and the per-task text blocks prepended to it."""

from __future__ import annotations

import json
import re

from ..domain.task import QuestionType, TaskInput
from ..media.timecode import format_timecode
from .tools import tool_specs

OPTION_LABEL = re.compile(r"^\s*[A-Z][.)]\s+")


def build_system_prompt() -> str:
    rendered_tools = "\n".join(json.dumps(spec, ensure_ascii=False) for spec in tool_specs())
    return (
        "You are a multimodal video-QA agent using one-step Observe-Think-Action JSON.\n"
        "\n"
        "Protocol:\n"
        "- Return exactly one raw JSON object with exactly three top-level keys: observation, think, and action. "
        "Do not use Markdown or surrounding prose.\n"
        "- observation and think must be strings; think must be non-empty. Use exactly one action per turn.\n"
        '- To finish, use {"type":"answer","content":"..."}. Every action uses only the displayed flat type form; '
        "aliases and nested parameters/arguments are invalid.\n"
        "- Tool start/end values use MM:SS.m with exactly one decimal. "
        "After a tool result, cite at least one [MM:SS.m,MM:SS.m] span in observation.\n"
        "\n"
        "Evidence:\n"
        "- observation records what the latest available media establishes. think may retain relevant earlier "
        "evidence, but must distinguish observed evidence from assumptions and state the unresolved "
        "answer-critical uncertainty.\n"
        "- Do not infer speaker identity or audio-video alignment without supporting evidence.\n"
        "- Use another tool when additional evidence is needed; otherwise answer from the accumulated evidence.\n"
        "\n"
        "Available tools, one JSON object per line:\n"
        f"{rendered_tools}\n"
    )


SYSTEM_PROMPT = build_system_prompt()


def question_context(task: TaskInput) -> str:
    lines = [
        "Video:",
        f"- duration: {format_timecode(float(task.duration_seconds or 0.0))}",
        f"- has_audio: {bool(task.has_audio)}",
        "- Tool times use MM:SS.m with exactly one decimal; minutes may exceed 59.",
        "",
        "Legal tool spans:",
        "- get_frames: coarse <= 04:16, detail <= 02:08.",
        "- get_audio: <= 04:16." if task.has_audio else "- get_audio: unavailable.",
        (
            "- get_clip: coarse <= 01:00, detail <= 01:00; same-span audio included."
            if task.has_audio
            else "- get_clip: coarse <= 01:00, detail <= 01:00; visual frames only."
        ),
        "",
        "Question: " + task.question,
    ]
    if task.question_type == QuestionType.MCQ:
        lines.append("Options:")
        lines.extend(
            option if OPTION_LABEL.match(option) else f"{chr(65 + index)}. {option}"
            for index, option in enumerate(task.options)
        )
        lines.append("Answer with exactly one uppercase option letter in action.content.")
    elif task.question_type == QuestionType.TEMPORAL_REASONING:
        lines.extend(
            [
                "Answer with one or more positive-width time intervals and no other text in action.content.",
                "- One interval: [01:23.4,01:27.8]",
                "- Multiple intervals: [01:23.4,01:27.8]; [03:02.0,03:06.5]",
                "- An isolated point such as 01:23.4 is invalid, including for a 'when' or 'moment' question.",
                "- Localize the event's supported onset and offset. Do not invent fixed-width padding around a point.",
                "- If either boundary is uncertain, use the media tools to resolve it before answering.",
            ]
        )
    elif task.question_type in {QuestionType.NUMBER, QuestionType.SIZE}:
        lines.append("Answer with one number only in action.content.")
    else:
        lines.append("Answer with concise free-form text in action.content.")
    return "\n".join(lines)


def initial_text(
    task: TaskInput,
    *,
    summary: str,
    audio_attached: bool,
) -> str:
    audio = (
        "\n- The attached full audio may be used for broad speech/sound localization."
        if audio_attached
        else "\n- No audio is attached; do not call get_audio. get_clip returns visual frames only."
    )
    return (
        question_context(task)
        + "\n\nInitial framepack media:\n"
        + summary
        + "\n- These timestamped frames are a global scan, not a continuous video."
        + audio
        + "\n- A successful focused tool call replaces this initial media because media.keep_recent=1; "
        "preserve useful initial evidence in observation/think."
    )
