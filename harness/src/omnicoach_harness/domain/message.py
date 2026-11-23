"""`Message` and its parts, including media references and native tool calls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]
MediaKind = Literal["image", "audio", "video"]


@dataclass(frozen=True, slots=True, kw_only=True)
class MediaRef:
    kind: MediaKind
    local_path: str | None = None
    uri: str | None = None
    mime_type: str | None = None
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def source(self) -> str:
        value = self.uri or self.local_path
        if not value:
            raise ValueError("media reference has neither uri nor local_path")
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class MessagePart:
    type: Literal["text", "image", "audio", "video"]
    text: str | None = None
    media: MediaRef | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class MessageToolCall:
    """Provider-neutral native function call retained on an assistant message."""

    call_id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    raw_arguments: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Message:
    role: Role
    parts: tuple[MessagePart, ...]
    reasoning_content: str | None = None
    tool_calls: tuple[MessageToolCall, ...] = ()
    tool_call_id: str | None = None


def text_part(text: str) -> MessagePart:
    return MessagePart(type="text", text=text)


def media_part(media: MediaRef) -> MessagePart:
    return MessagePart(type=media.kind, media=media)


def message(
    role: Role,
    *parts: MessagePart,
    reasoning_content: str | None = None,
    tool_calls: tuple[MessageToolCall, ...] = (),
    tool_call_id: str | None = None,
) -> Message:
    return Message(
        role=role,
        parts=tuple(parts),
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
    )


def text_message(role: Role, text: str) -> Message:
    return message(role, text_part(text))


def serialize_message(value: Message) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    for part in value.parts:
        if part.type == "text":
            parts.append({"type": "text", "text": part.text or ""})
            continue
        media = part.media
        parts.append(
            {
                "type": part.type,
                "path": media.local_path if media else None,
                "uri": media.uri if media and not str(media.uri or "").startswith("data:") else None,
                "mime_type": media.mime_type if media else None,
                "label": media.label if media else None,
                "metadata": dict(media.metadata) if media else {},
            }
        )
    result: dict[str, Any] = {"role": value.role, "parts": parts}
    if value.reasoning_content is not None:
        result["reasoning_content"] = value.reasoning_content
    if value.tool_calls:
        result["tool_calls"] = [
            {
                "id": item.call_id,
                "type": "function",
                "function": {
                    "name": item.name,
                    "arguments": item.raw_arguments
                    if item.raw_arguments is not None
                    else dict(item.arguments),
                },
            }
            for item in value.tool_calls
        ]
    if value.tool_call_id is not None:
        result["tool_call_id"] = value.tool_call_id
    return result
