"""`TaskInput` and `TaskLabel`: the model-visible task and the answer held back from it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


def _optional_bool(value: Any, *, field_name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"{field_name} must be a boolean when provided")


class QuestionType(StrEnum):
    MCQ = "MCQ"
    TEMPORAL_REASONING = "TR"
    FREE_FORM = "FF"
    NUMBER = "NUM"
    SIZE = "SIZE"

    @classmethod
    def parse(cls, value: Any) -> QuestionType:
        normalized = str(value or "FF").strip().upper().replace("-", "_")
        aliases = {
            "MC": cls.MCQ,
            "MCQ": cls.MCQ,
            "MULTIPLE_CHOICE": cls.MCQ,
            "TR": cls.TEMPORAL_REASONING,
            "TEMPORAL_REASONING": cls.TEMPORAL_REASONING,
            "FF": cls.FREE_FORM,
            "FREE_FORM": cls.FREE_FORM,
            "OPEN": cls.FREE_FORM,
            "NUM": cls.NUMBER,
            "NUMBER": cls.NUMBER,
            "SIZE": cls.SIZE,
        }
        return aliases.get(normalized, cls.FREE_FORM)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskInput:
    """Everything an actor is allowed to observe before model inference."""

    task_id: str
    video_path: str
    audio_path: str | None = None
    question: str
    question_type: QuestionType
    options: tuple[str, ...] = ()
    duration_seconds: float | None = None
    fps: float | None = None
    has_audio: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if not self.video_path.strip():
            raise ValueError("video_path must be non-empty")
        if not self.question.strip():
            raise ValueError("question must be non-empty")
        if self.duration_seconds is not None and self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> TaskInput:
        metadata = dict(value.get("metadata") or {})
        duration = value.get("duration_seconds", metadata.get("duration_seconds"))
        has_audio = value.get("has_audio", metadata.get("has_audio"))
        task_id = value.get("task_id")
        if task_id is None:
            task_id = value.get("sample_id")
        if task_id is None:
            task_id = value.get("index")
        return cls(
            task_id=str(task_id if task_id is not None else "").strip(),
            video_path=str(value.get("video_path") or "").strip(),
            audio_path=str(value.get("audio_path") or "").strip() or None,
            question=str(value.get("question") or "").strip(),
            question_type=QuestionType.parse(value.get("question_type")),
            options=tuple(str(item) for item in (value.get("options") or ())),
            duration_seconds=float(duration) if duration is not None else None,
            fps=float(value["fps"]) if value.get("fps") is not None else None,
            has_audio=_optional_bool(has_audio, field_name="has_audio"),
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskLabel:
    """Private evaluation information that is never passed to ActorRoute.start."""

    answer: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> TaskLabel:
        return cls(answer=value.get("answer"), metadata=dict(value.get("label_metadata") or {}))


def split_task_record(value: Mapping[str, Any]) -> tuple[TaskInput, TaskLabel]:
    return TaskInput.from_record(value), TaskLabel.from_record(value)
