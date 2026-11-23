"""`ModelRequest`, `ModelResponse` and the `ModelClient` protocol a transport implements."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..domain.message import Message


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelRequest:
    messages: tuple[Message, ...]
    model: str
    max_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    response_format: Mapping[str, Any] | None = None
    tools: tuple[Mapping[str, Any], ...] = ()
    tool_choice: str | Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelResponse:
    text: str
    model: str
    provider: str
    usage: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    assistant_message: Message | None = None


class ModelClient(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse: ...
