"""`OpenAICompatModelClient`, the chat-completions transport, with round-robin endpoints and retries."""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

from ..domain.message import MessageToolCall, message, text_part
from .messages import VALID_AUDIO_PART_TYPES, to_provider_messages
from .protocol import ModelRequest, ModelResponse
from .telemetry import ProviderBudgetExhausted, record

logger = logging.getLogger(__name__)


class OpenAICompatModelClient:
    def __init__(
        self,
        *,
        base_urls: str | Sequence[str],
        api_key: str = "EMPTY",
        retries: int = 3,
        retry_429_forever: bool = True,
        max_429_retries: int = 100,
        timeout_seconds: float = 180.0,
        temperature: float = 0.0,
        audio_part_type: str = "audio_url",
        extra_body: Mapping[str, Any] | None = None,
        clients: Sequence[Any] | None = None,
    ) -> None:
        if retries < 1:
            raise ValueError("retries must be >= 1")
        if max_429_retries < 1:
            raise ValueError("max_429_retries must be >= 1")
        if audio_part_type not in VALID_AUDIO_PART_TYPES:
            raise ValueError(
                f"unsupported audio_part_type {audio_part_type!r}; expected one of {VALID_AUDIO_PART_TYPES}"
            )
        urls = _base_urls(base_urls)
        if clients is None:
            from openai import OpenAI

            clients = tuple(
                OpenAI(api_key=api_key, base_url=url + "/", timeout=timeout_seconds, max_retries=0)
                for url in urls
            )
        if not clients:
            raise ValueError("at least one OpenAI-compatible client is required")
        self._clients = tuple(clients)
        self._next = 0
        self._lock = threading.Lock()
        self.retries = retries
        self.retry_429_forever = retry_429_forever
        self.max_429_retries = max_429_retries
        self.temperature = temperature
        self.audio_part_type = audio_part_type
        self.extra_body = dict(extra_body) if extra_body is not None else {}

    def _request_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": to_provider_messages(request.messages, audio_part_type=self.audio_part_type),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature if request.temperature is not None else self.temperature,
        }
        if self.extra_body:
            kwargs["extra_body"] = dict(self.extra_body)
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.response_format is not None:
            kwargs["response_format"] = dict(request.response_format)
        if request.tools:
            kwargs["tools"] = [dict(item) for item in request.tools]
            if request.tool_choice is not None:
                kwargs["tool_choice"] = request.tool_choice
        return kwargs

    def complete(self, request: ModelRequest) -> ModelResponse:
        # Encode media and construct request payload once across retries.
        kwargs = self._request_kwargs(request)
        attempts = 0
        while True:
            attempts += 1
            client, endpoint_index = self._next_client()
            entry = f"endpoint_{endpoint_index}"
            record(request.model, "attempt", entry=entry)
            try:
                response = client.chat.completions.create(**kwargs)
                choices = getattr(response, "choices", None) or []
                if not choices:
                    raise RuntimeError("OpenAI-compatible endpoint returned no choices")
                choice_message = choices[0].message
                content = getattr(choice_message, "content", None) or ""
                tool_calls = _tool_calls(getattr(choice_message, "tool_calls", None) or [])
                if not content.strip() and not tool_calls:
                    raise RuntimeError("OpenAI-compatible endpoint returned empty content")
                usage = getattr(response, "usage", None)
                record(request.model, "success", entry=entry)
                return ModelResponse(
                    text=content,
                    assistant_message=message(
                        "assistant",
                        *((text_part(content),) if content else ()),
                        reasoning_content=getattr(choice_message, "reasoning_content", None),
                        tool_calls=tool_calls,
                    ),
                    model=str(getattr(response, "model", None) or request.model),
                    provider="openai_compat",
                    usage=usage.model_dump() if hasattr(usage, "model_dump") else {},
                    metadata={
                        "endpoint_index": endpoint_index,
                        "provider_attempts": attempts,
                        "response_id": getattr(response, "id", None),
                        "request_metadata": dict(request.metadata),
                    },
                )
            except ProviderBudgetExhausted:
                logger.warning("provider budget exhausted model=%s endpoint=%d", request.model, endpoint_index)
                raise
            except Exception as exc:
                rate_limited = _is_429(exc)
                record(request.model, "rate_limited" if rate_limited else "error", entry=entry)
                max_attempts = self.max_429_retries if (rate_limited and self.retry_429_forever) else self.retries
                if attempts >= max_attempts:
                    logger.error(
                        "provider request failed permanently model=%s endpoint=%d "
                        "attempts=%d/%d rate_limited=%s error=%s",
                        request.model,
                        endpoint_index,
                        attempts,
                        max_attempts,
                        rate_limited,
                        exc,
                    )
                    raise
                delay = (
                    min(30.0, float(2 ** min(attempts - 1, 5)))
                    if rate_limited
                    else random.uniform(0.0, min(2 ** (attempts - 1), 4.0))
                )
                logger.warning(
                    "provider request failed; retrying model=%s endpoint=%d "
                    "attempt=%d/%d rate_limited=%s delay=%.2fs error=%s",
                    request.model,
                    endpoint_index,
                    attempts,
                    max_attempts,
                    rate_limited,
                    delay,
                    exc,
                )
                time.sleep(delay)

    def _next_client(self) -> tuple[Any, int]:
        with self._lock:
            index = self._next % len(self._clients)
            self._next += 1
        return self._clients[index], index


def _base_urls(value: str | Sequence[str]) -> tuple[str, ...]:
    values = value.split(",") if isinstance(value, str) else value
    urls = tuple(str(item).strip().rstrip("/") for item in values if str(item).strip())
    if not urls:
        raise ValueError("at least one base URL is required")
    return urls


def _is_429(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(seen) < 8:
        seen.add(id(current))
        text = str(current).casefold()
        if getattr(current, "status_code", None) == 429 or "error code: 429" in text:
            return True
        current = current.__cause__ or current.__context__
    return False


def _tool_calls(raw: Sequence[Any]) -> tuple[MessageToolCall, ...]:
    calls = []
    for index, item in enumerate(raw):
        function = getattr(item, "function", None)
        name = getattr(function, "name", None)
        if not name:
            raise RuntimeError(f"endpoint returned tool call {index} without a function name")
        raw_arguments = getattr(function, "arguments", None) or "{}"
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"endpoint returned invalid arguments for tool {name}") from exc
        if not isinstance(arguments, dict):
            raise RuntimeError(f"endpoint returned non-object arguments for tool {name}")
        calls.append(
            MessageToolCall(
                call_id=str(getattr(item, "id", None) or f"call_{index}"),
                name=str(name),
                arguments=arguments,
                raw_arguments=str(raw_arguments),
            )
        )
    return tuple(calls)
