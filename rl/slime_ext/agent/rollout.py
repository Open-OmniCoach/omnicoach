"""Unified multi-turn & partial-rollout engine for Qwen3-Omni active-perception agents.

Rollout invariants:
- Persists explicit ``terminal_status`` and ``failure_reason`` even when
  ``turns == []`` (prevents turn-0 context-overflow crashes and ABORTED
  re-entry livelocks), paired with ``safe_min_weight_version`` / ``safe_policy_lag``.
- On partial resume under ``mask_offpolicy_in_partial_rollout``, zeros
  ``loss_mask`` across all historical ``episode.turns`` and any already-generated
  prefix of ``episode.active``.
- Cleanly terminates on media decode failure (``MEDIA_TOOL_FAILED``) without
  polluting ``episode.turns``; initializes valid empty top-p offsets ``[0]`` on
  0-token completions when ``rollout_top_p != 1.0``.
- Records ``weight_version`` onto a turn only when ``new_token_count > 0``.
- Stores neutral initial and per-turn media observations in
  ``sample.metadata["initial_observations"]``, ``turn.metadata["tool_observations"]``,
  and ``sample.metadata["agent_turns"]`` alongside ``initial_media_state`` across
  partial pauses/resumes.
- Uses a dedicated bounded ``ThreadPoolExecutor`` passed explicitly to
  ``loop.run_in_executor`` without mutating the global event loop default executor.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import sys
import time
import wave
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from itertools import groupby
from typing import Any

from .media import (
    InitialMediaState,
    Media,
    MediaBudget,
    audio_data,
    build_tool_observations,
)
from .protocol import (
    CHAT_TEMPLATE,
    AgentSession,
    MediaBundle,
    ProtocolError,
    ToolRequest,
    classify_turn_role,
    extract_answer,
    parse_generated_turn,
    require_single_tool_call,
    validate_assistant_text,
)

logger = logging.getLogger(__name__)


class ContextLengthExceeded(Exception):
    """Raised when SGLang rejects a prompt for exceeding max context length."""


class FallbackStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass
class AgentSample:
    """Lightweight duck-typed Sample compatible with ``slime.utils.types.Sample``."""

    Status = FallbackStatus

    index: int | None = None
    group_index: int | None = None
    rollout_id: int | None = None
    session_id: str | None = None
    prompt: Any = ""
    label: Any = None
    tokens: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    loss_mask: list[int] | None = None
    reward: Any = None
    status: Any = FallbackStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)
    multimodal_train_inputs: dict[str, Any] | None = None
    rollout_log_probs: list[float] | None = None
    rollout_top_p_token_ids: Any = None
    rollout_top_p_token_offsets: Any = None
    rollout_routed_experts: Any = None
    weight_versions: list[int] = field(default_factory=list)
    remove_sample: bool = False
    non_generation_time: float = 0.0
    trace: Any = None

    def get_reward_value(self, args: Any = None) -> float:
        reward_key = getattr(args, "reward_key", None) if args is not None else None
        if isinstance(self.reward, Mapping) and reward_key is not None:
            return float(self.reward[reward_key])
        return float(self.reward)


def _status_enum(sample: Any):
    return getattr(type(sample), "Status", FallbackStatus)


def _status_name(status: Any) -> str:
    return status.name if isinstance(status, Enum) else str(status).upper()


def _resolve_status(sample: Any, name: str):
    enum_cls = _status_enum(sample)
    key = str(name).upper()
    if hasattr(enum_cls, key):
        return getattr(enum_cls, key)
    return getattr(FallbackStatus, key)


def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


def _media_execution_errors() -> tuple[type[BaseException], ...]:
    """Return concrete media-decoding exception types without masking code bugs."""
    errors: list[type[BaseException]] = [
        RuntimeError,
        OSError,
        ValueError,
        EOFError,
        IndexError,
        wave.Error,
    ]
    decord_mod = sys.modules.get("decord")
    decord_err = getattr(decord_mod, "DECORDError", None) if decord_mod is not None else None
    if isinstance(decord_err, type) and issubclass(decord_err, BaseException):
        errors.append(decord_err)
    return tuple(errors)


async def _run_blocking(
    executor: ThreadPoolExecutor | None,
    func: Callable[..., Any],
    *args: Any,
) -> Any:
    """Run blocking media/processor work in a dedicated bounded executor when available."""
    if executor is None:
        return await asyncio.to_thread(func, *args)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, func, *args)


@dataclass(frozen=True)
class RolloutHooks:
    """Optional extension hooks for customizing rollout lifecycle events."""

    on_initial_media: Callable[[Any, InitialMediaState, MediaBundle], None] = _noop
    on_tool_executed: Callable[[Any, Any, ToolRequest, MediaBundle, list[dict[str, Any]]], None] = _noop
    on_turn_finished: Callable[[Any, Any, dict[str, Any]], None] = _noop


@dataclass
class InitialPreparation:
    initial: asyncio.Task | None = None
    training: asyncio.Task | None = None
    users: int = 0


@dataclass
class Episode:
    """Mutable multi-turn episode state preserved across partial-rollout pauses."""

    session: AgentSession
    media_state: InitialMediaState
    turns: list[Any] = field(default_factory=list)
    active: Any | None = None
    turn_spans: list[tuple[int, int]] = field(default_factory=list)
    turn_roles: list[str] = field(default_factory=list)
    turn_weight_versions: list[int | None] = field(default_factory=list)
    turn_truncated: list[bool] = field(default_factory=list)
    turn_behavior_logprobs: list[list[float]] = field(default_factory=list)
    turn_records: list[dict[str, Any]] = field(default_factory=list)
    response_cursor: int = 0


def safe_min_weight_version(samples: Sequence[Any]) -> int | None:
    """Return the oldest weight version across samples, or None if empty."""
    versions: list[int] = []
    for sample in samples:
        for v in getattr(sample, "weight_versions", None) or ():
            try:
                versions.append(int(v))
            except (TypeError, ValueError):
                continue
    return min(versions) if versions else None


def safe_max_weight_version(samples: Sequence[Any]) -> int | None:
    """Return the newest weight version across samples, or None if empty."""
    versions: list[int] = []
    for sample in samples:
        for v in getattr(sample, "weight_versions", None) or ():
            try:
                versions.append(int(v))
            except (TypeError, ValueError):
                continue
    return max(versions) if versions else None


def safe_policy_lag(training_version: int | None, samples: Sequence[Any]) -> int | None:
    """Compute policy lag safely when training_version is None or samples have no versions."""
    if training_version is None:
        return None
    oldest = safe_min_weight_version(samples)
    if oldest is None:
        return None
    return max(0, int(training_version) - oldest)


def apply_turn_generation_result(
    args: Any,
    turn: Any,
    *,
    new_tokens: Sequence[int],
    new_log_probs: Sequence[float],
    text: str,
    finish_reason: str,
    weight_version: int | None = None,
    meta_info: Mapping[str, Any] | None = None,
) -> None:
    """Apply one generation step while preserving 0-token top-p and weight-version invariants."""
    toks = [int(t) for t in new_tokens]
    lps = [float(lp) for lp in new_log_probs]
    if len(toks) != len(lps):
        raise ValueError(f"new_tokens ({len(toks)}) and new_log_probs ({len(lps)}) length mismatch")
    new_count = len(toks)

    if hasattr(turn, "append_response_tokens") and meta_info is not None:
        # Delegate to Slime Sample API when using live SGLang meta_info, then
        # preserve previous weight versions on 0-token completions and empty top-p offsets.
        prev_versions = list(getattr(turn, "weight_versions", []) or [])
        turn.append_response_tokens(
            args,
            tokens=toks,
            log_probs=lps,
            meta_info=dict(meta_info),
            text=text,
        )
        if new_count == 0:
            turn.weight_versions = prev_versions
    else:
        turn.tokens = list(turn.tokens) + toks
        turn.response = (turn.response or "") + text
        turn.response_length = int(turn.response_length) + new_count
        if turn.loss_mask is None:
            turn.loss_mask = [1] * turn.response_length
        else:
            turn.loss_mask = list(turn.loss_mask) + [1] * new_count
        if turn.rollout_log_probs is None:
            turn.rollout_log_probs = list(lps)
        else:
            turn.rollout_log_probs = list(turn.rollout_log_probs) + lps
        # Record weight_version only when at least one new token was generated.
        if new_count > 0 and weight_version is not None:
            turn.weight_versions = list(turn.weight_versions) + [int(weight_version)]
        raw_reason = finish_reason.get("type", "stop") if isinstance(finish_reason, Mapping) else finish_reason
        reason = str(raw_reason).lower()
        if reason == "abort":
            turn.status = _resolve_status(turn, "ABORTED")
        elif reason == "length":
            turn.status = _resolve_status(turn, "TRUNCATED")
        else:
            turn.status = _resolve_status(turn, "COMPLETED")

    ensure_valid_top_p_fields(args, turn)


def ensure_valid_top_p_fields(args: Any, turn: Any) -> None:
    """Guarantee valid empty top-p offsets on 0-token turns when rollout_top_p != 1.0."""
    if float(getattr(args, "rollout_top_p", 1.0)) == 1.0:
        return
    if int(getattr(turn, "response_length", 0)) == 0 and getattr(turn, "rollout_top_p_token_ids", None) is None:
        turn.rollout_top_p_token_ids = []
        turn.rollout_top_p_token_offsets = [0]


def _apply_partial_offpolicy_mask(args: Any, episode: Episode) -> None:
    """Mask all historical turns and any active prefix on partial rollout resume."""
    if not (
        getattr(args, "partial_rollout", False)
        and getattr(args, "mask_offpolicy_in_partial_rollout", False)
    ):
        return
    for past_turn in episode.turns:
        past_turn.loss_mask = [0] * int(past_turn.response_length)
    if episode.active is not None:
        episode.active.loss_mask = [0] * int(episode.active.response_length)


def _record_completed_turn(episode: Episode, turn: Any, *, role: str, truncated: bool) -> None:
    resp_len = int(turn.response_length)
    start = episode.response_cursor
    end = start + resp_len
    episode.response_cursor = end
    episode.turn_spans.append((start, end))
    episode.turn_roles.append(role)
    versions = [int(v) for v in (getattr(turn, "weight_versions", None) or ())]
    weight_version = min(versions) if versions else None
    episode.turn_weight_versions.append(weight_version)
    episode.turn_truncated.append(bool(truncated))
    lps = list(turn.rollout_log_probs) if turn.rollout_log_probs is not None else []
    episode.turn_behavior_logprobs.append(lps)
    turn.metadata["turn_role"] = role
    turn.metadata["turn_truncated"] = bool(truncated)
    turn.metadata["turn_span"] = (start, end)
    turn.metadata["turn_weight_version"] = weight_version
    episode.turn_records.append(
        {
            "turn_index": int(turn.metadata.get("turn_index", len(episode.turn_records))),
            "role": role,
            "action": turn.metadata.get("action"),
            "invalid": bool(turn.metadata.get("invalid", False)),
            "error_code": turn.metadata.get("error_code"),
            "no_progress": bool(turn.metadata.get("no_progress", False)),
            "truncated": bool(truncated),
            "token_span": (start, end),
            "response_length": resp_len,
            "weight_version": weight_version,
            "cost_units": float(turn.metadata.get("cost_units", 0.0)),
            "tool_observations": copy.deepcopy(turn.metadata.get("tool_observations", [])),
        }
    )


def _paused(base: Any, episode: Episode) -> Any:
    """Build carrier Sample preserving Episode and InitialMediaState across pauses."""
    carrier = copy.copy(episode.active or (episode.turns[-1] if episode.turns else base))
    carrier.metadata = {
        **base.metadata,
        "episode": episode,
        "initial_media_state": episode.media_state.to_dict(),
        "initial_observations": copy.deepcopy(episode.media_state.initial_observations),
    }
    carrier.status = _resolve_status(base, "ABORTED")
    carrier.reward = None
    return carrier


def _finished(
    base: Any,
    episode: Episode,
    status: Any,
    answer: str | None,
    *,
    failure_reason: str | None = None,
) -> Any:
    """Finalize trajectory carrier with full neutral turn metadata and terminal_status."""
    status_str = _status_name(status)
    for turn in episode.turns:
        if status_str != "COMPLETED":
            turn.status = _resolve_status(turn, status_str)
    carrier = copy.copy(episode.turns[-1] if episode.turns else base)
    carrier.weight_versions = [
        int(v) for turn in episode.turns for v in (getattr(turn, "weight_versions", None) or ())
    ]
    carrier.status = _resolve_status(base, status_str)
    carrier.metadata = {
        **base.metadata,
        "turns": list(episode.turns),
        "agent_turns": copy.deepcopy(episode.turn_records),
        "turn_spans": list(episode.turn_spans),
        "turn_roles": list(episode.turn_roles),
        "turn_weight_versions": list(episode.turn_weight_versions),
        "turn_truncated": list(episode.turn_truncated),
        "turn_behavior_logprobs": [list(lp) for lp in episode.turn_behavior_logprobs],
        "total_response_length": episode.response_cursor,
        "initial_media_state": episode.media_state.to_dict(),
        "initial_observations": copy.deepcopy(episode.media_state.initial_observations),
        "budget_used": float(episode.media_state.budget.get("cumulative_cost_units", 0.0)),
        "answer": answer,
        "invalid_actions": episode.session.invalid_actions,
        "no_progress": episode.session.total_no_progress,
        "terminal_status": status_str,
        "failure_reason": failure_reason,
    }
    return carrier


def _extract_user_prompt(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, Sequence) and prompt and isinstance(prompt[0], Mapping):
        return str(prompt[0].get("content", ""))
    return str(prompt or "")


def _new_turn_sample(base: Any, *, prompt: str, tokens: list[int], turn_index: int) -> Any:
    cls = type(base)
    turn = cls(
        index=base.index,
        group_index=base.group_index,
        rollout_id=base.rollout_id,
        session_id=base.session_id,
        prompt=prompt,
        label=base.label,
        tokens=list(tokens),
        metadata={**base.metadata, "turn_index": turn_index},
    )
    if hasattr(base, "trace"):
        turn.trace = base.trace
    return turn


def _media_inputs(messages: list[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
    images, audios = [], []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            continue
        for part in content:
            if part["type"] == "image":
                images.append(part["image"])
            elif part["type"] == "audio":
                audios.append(part["audio"])
    return images, audios


def _sglang_payload(args: Any, turn: Any, messages: list[dict[str, Any]], processor: Any, sampling: dict) -> dict:
    placeholders = {
        processor.tokenizer.convert_tokens_to_ids(getattr(processor, f"{kind}_token"))
        for kind in ("image", "audio", "video")
    }
    input_ids: list[int] = []
    prompt_length = len(turn.tokens) - turn.response_length
    for token, repeated in groupby(turn.tokens[:prompt_length]):
        input_ids.extend([token] if token in placeholders else repeated)
    input_ids.extend(turn.tokens[prompt_length:])

    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": sampling,
        "return_logprob": True,
    }
    images, audios = _media_inputs(messages)
    if images:
        payload["image_data"] = [image.info["data_url"] for image in images]
    if audios:
        payload["audio_data"] = [
            audio_data(audio, turn.metadata["audio_sample_rate"]) for audio in audios
        ]
    if getattr(args, "use_rollout_routing_replay", False):
        payload["return_routed_experts"] = True
    return payload


async def _post_turn_http(url: str, payload: dict, headers: dict | None) -> dict:
    import httpx
    from slime.utils.http_utils import post

    for attempt in range(60):
        try:
            return await post(url, payload, headers=headers, max_retries=1)
        except httpx.HTTPStatusError as error:
            if (
                error.response.status_code == 400
                and "Requested token count exceeds the model's maximum context length"
                in error.response.text
            ):
                raise ContextLengthExceeded from error
            if error.response.status_code == 400 or attempt == 59:
                raise
        except (httpx.HTTPError, OSError, asyncio.TimeoutError, TimeoutError):
            if attempt == 59:
                raise
        await asyncio.sleep(1)
    raise RuntimeError("unreachable")


def _release_initial(state: Any, group_index: Any, preparation: InitialPreparation) -> None:
    preparation.users -= 1
    if preparation.users <= 0:
        state.initial_preparations.pop(group_index, None)


def _prepare_sglang_turn(args: Any, base: Any, episode: Episode, processor: Any) -> Any:
    started = time.perf_counter()
    prompt = processor.apply_chat_template(
        episode.session.messages,
        chat_template=CHAT_TEMPLATE,
        tools=episode.session.tools,
        tokenize=False,
        add_generation_prompt=True,
        **(getattr(args, "apply_chat_template_kwargs", None) or {}),
    )
    tokens = processor.tokenizer.encode(prompt, add_special_tokens=False)
    turn = _new_turn_sample(base, prompt=prompt, tokens=tokens, turn_index=len(episode.turns))
    turn.non_generation_time = time.perf_counter() - started
    return turn


def _build_training_inputs(turn: Any, messages: list[dict[str, Any]], processor: Any, dtype: Any):
    started = time.perf_counter()
    images, audios = _media_inputs(messages)
    output = processor(
        text=turn.prompt,
        images=images or None,
        audio=audios or None,
        sampling_rate=turn.metadata["audio_sample_rate"],
        return_tensors="pt",
    ).to(dtype=dtype)
    return output, time.perf_counter() - started


async def _generate_turn_sglang(
    args: Any,
    turn: Any,
    episode: Episode,
    state: Any,
    sampling_params: dict,
    initial_preparation: InitialPreparation | None = None,
    executor: ThreadPoolExecutor | None = None,
) -> None:
    from slime.observability.trace_utils import build_sglang_meta_trace_attrs, trace_span
    from slime.rollout.sglang_rollout import get_model_url

    sampling = sampling_params.copy()
    sampling["max_new_tokens"] = max(0, int(sampling["max_new_tokens"]) - int(turn.response_length))
    if sampling["max_new_tokens"] == 0:
        turn.status = _resolve_status(turn, "TRUNCATED")
        ensure_valid_top_p_fields(args, turn)
        return

    started = time.perf_counter()
    payload = await _run_blocking(
        executor, _sglang_payload, args, turn, episode.session.messages, state.processor, sampling
    )
    turn.non_generation_time += time.perf_counter() - started
    if getattr(state, "aborted", False):
        turn.status = _resolve_status(turn, "ABORTED")
        return

    headers = (
        {"X-SMG-Routing-Key": turn.session_id}
        if getattr(args, "router_policy", None) == "consistent_hashing"
        else None
    )

    async def request():
        with trace_span(turn, "llm_request") as span:
            result = await _post_turn_http(get_model_url(args, "actor"), payload, headers)
            if hasattr(span, "update"):
                span.update(build_sglang_meta_trace_attrs(result["meta_info"]))
            return result

    response_coro = request()
    if turn.multimodal_train_inputs is None:
        prep = initial_preparation.training if initial_preparation is not None else None
        if prep is None:
            prep = asyncio.create_task(
                _run_blocking(
                    executor,
                    _build_training_inputs,
                    turn,
                    episode.session.messages,
                    state.processor,
                    args.params_dtype,
                )
            )
            if initial_preparation is not None:
                initial_preparation.training = prep
        output, (training_inputs, prep_s) = await asyncio.gather(response_coro, prep)
        response_tokens = turn.tokens[-turn.response_length :] if turn.response_length else []
        turn.tokens = training_inputs["input_ids"][0].tolist() + response_tokens
        turn.multimodal_train_inputs = {
            k: v for k, v in training_inputs.items() if k not in {"input_ids", "attention_mask"}
        }
        turn.non_generation_time += prep_s
    else:
        output = await response_coro

    meta_info = output["meta_info"]
    logprobs = meta_info.get("output_token_logprobs", [])
    apply_turn_generation_result(
        args,
        turn,
        new_tokens=[item[1] for item in logprobs],
        new_log_probs=[item[0] for item in logprobs],
        text=output.get("text", ""),
        finish_reason=(
            str(meta_info["finish_reason"].get("type", "stop"))
            if isinstance(meta_info.get("finish_reason"), Mapping)
            else str(meta_info.get("finish_reason", "stop"))
        ),
        weight_version=meta_info.get("weight_version"),
        meta_info=meta_info,
    )


def _extract_tool_observations(
    request: ToolRequest,
    bundle: MediaBundle,
    *,
    duration: float,
    turn_index: int,
    has_audio: bool,
    settings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_obs = bundle.metadata.get("tool_observations")
    if raw_obs is None:
        return build_tool_observations(
            request,
            bundle,
            duration_sec=duration,
            has_audio=has_audio,
            settings=settings,
            turn_index=turn_index,
        )
    stamped: list[dict[str, Any]] = []
    for obs in raw_obs:
        copied = copy.deepcopy(obs)
        copied.setdefault("params", {})["turn_index"] = int(turn_index)
        stamped.append(copied)
    return stamped


async def generate_episode(
    args: Any,
    sample: Any,
    sampling_params: dict[str, Any],
    *,
    evaluation: bool = False,
    hooks: RolloutHooks | None = None,
    media_factory: Callable[[Any, Mapping[str, Any]], Media] | None = None,
    turn_generator: Callable[[Any, Any, Episode, dict[str, Any]], Awaitable[None]] | None = None,
    is_aborted_fn: Callable[[], bool] | None = None,
) -> Any:
    """Run or resume a multi-turn active-perception episode.

    Works both in live Slime/SGLang async rollout and in pure CPU mode when
    ``media_factory`` and ``turn_generator`` are injected.
    """
    active_hooks = hooks or RolloutHooks()

    # Restore terminal status even when turns == [] to prevent re-entry livelocks.
    if "terminal_status" in sample.metadata:
        sample.status = _resolve_status(sample, sample.metadata["terminal_status"])
        return sample
    if "turns" in sample.metadata:
        if sample.metadata["turns"]:
            sample.status = sample.metadata["turns"][-1].status
        return sample

    config = args.agent_config
    episode: Episode | None = sample.metadata.pop("episode", None)
    if sample.rollout_id is None:
        sample.rollout_id = sample.index

    live_state = None
    media_executor: ThreadPoolExecutor | None = None
    initial_preparation: InitialPreparation | None = None
    if media_factory is None or turn_generator is None:
        from slime.rollout.sglang_rollout import GenerateState
        from slime.utils.http_utils import get_rollout_num_engines

        live_state = GenerateState(args)
        if not hasattr(live_state, "media_executor"):
            cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
            workers = min(
                cpu_count,
                args.sglang_server_concurrency * get_rollout_num_engines(args),
            )
            live_state.media_executor = ThreadPoolExecutor(
                max_workers=max(1, workers), thread_name_prefix="media"
            )
            live_state.initial_preparations = {}
        media_executor = live_state.media_executor

    if media_factory is not None:
        media = media_factory(sample, config["media"])
    else:
        media = await _run_blocking(
            media_executor,
            Media,
            sample.metadata["media"],
            config["media"],
            live_state.processor.feature_extractor.n_fft,
        )

    def _aborted() -> bool:
        if is_aborted_fn is not None:
            return bool(is_aborted_fn())
        return bool(getattr(live_state, "aborted", False))

    try:
        duration = media.duration_ms / 1000.0
        sample.metadata["media_duration_seconds"] = duration
        sample.metadata["audio_sample_rate"] = media.sample_rate

        if episode is None:
            if live_state is not None and not evaluation and sample.group_index is not None:
                initial_preparation = live_state.initial_preparations.setdefault(
                    sample.group_index, InitialPreparation()
                )
                initial_preparation.users += 1
                if initial_preparation.initial is None:
                    initial_preparation.initial = asyncio.create_task(
                        _run_blocking(media_executor, media.initial)
                    )
                initial_bundle, attached, media_state = await initial_preparation.initial
                media_state = InitialMediaState.from_dict(media_state.to_dict())
            else:
                initial_bundle, attached, media_state = await _run_blocking(media_executor, media.initial)

            sample.metadata["initial_media_state"] = media_state.to_dict()
            sample.metadata["initial_observations"] = copy.deepcopy(media_state.initial_observations)
            active_hooks.on_initial_media(sample, media_state, initial_bundle)

            session = AgentSession.create(
                prompt=_extract_user_prompt(sample.prompt),
                duration_seconds=duration,
                has_audio=media.has_audio,
                initial_audio_attached=attached,
                initial=initial_bundle,
                settings=config["media"],
                frame_contract=media.frame_contract,
            )
            episode = Episode(session=session, media_state=media_state)
        else:
            # Partial resume: restore exact cumulative budget and mask prior off-policy tokens.
            media.budget = MediaBudget.from_dict(episode.media_state.budget)
            _apply_partial_offpolicy_mask(args, episode)

        session = episode.session
        max_turns = int(config["max_turns"])
        max_invalid_retries = int(config.get("max_invalid_retries", 3))

        while len(episode.turns) < max_turns:
            if _aborted():
                return _paused(sample, episode)

            if episode.active is None:
                session.append_control_state(step=len(episode.turns), max_steps=max_turns)
                if turn_generator is not None:
                    episode.active = _new_turn_sample(
                        sample,
                        prompt=json.dumps(session.messages, default=str),
                        tokens=[],
                        turn_index=len(episode.turns),
                    )
                else:
                    episode.active = await _run_blocking(
                        media_executor, _prepare_sglang_turn, args, sample, episode, live_state.processor
                    )

            turn = episode.active
            try:
                if turn_generator is not None:
                    await turn_generator(args, turn, episode, sampling_params)
                else:
                    await _generate_turn_sglang(
                        args,
                        turn,
                        episode,
                        live_state,
                        sampling_params,
                        initial_preparation,
                        executor=media_executor,
                    )
            except ContextLengthExceeded:
                logger.warning(
                    "Context overflow: task=%s sample=%s turn=%s",
                    sample.metadata.get("task_id"),
                    sample.index,
                    len(episode.turns),
                )
                return _finished(
                    sample,
                    episode,
                    _resolve_status(sample, "FAILED"),
                    answer=None,
                    failure_reason="CONTEXT_LENGTH_EXCEEDED",
                )
            finally:
                if initial_preparation is not None:
                    _release_initial(live_state, sample.group_index, initial_preparation)
                    initial_preparation = None

            if _status_name(turn.status) == "ABORTED":
                return _paused(sample, episode)

            turn_truncated = _status_name(turn.status) == "TRUNCATED"
            episode.turns.append(turn)
            episode.active = None

            try:
                parsed = parse_generated_turn(turn.response)
            except ProtocolError as error:
                role = classify_turn_role(turn.response, parsed=None, has_parse_error=True)
                turn.metadata.update(action="invalid", invalid=True, error_code=error.code)
                _record_completed_turn(episode, turn, role=role, truncated=turn_truncated)
                active_hooks.on_turn_finished(sample, turn, turn.metadata)
                session.append_assistant({"role": "assistant", "content": turn.response})
                session.append_protocol_error(
                    error.code, str(error), max_invalid_retries=max_invalid_retries
                )
                continue

            session.append_assistant(parsed.as_message())
            calls = parsed.tool_calls
            intent_role = classify_turn_role(turn.response, parsed=parsed, has_parse_error=False)

            try:
                validate_assistant_text(parsed.text, has_tool_call=bool(calls))
                if not calls:
                    turn.metadata["action"] = "answer"
                    _record_completed_turn(episode, turn, role="answer", truncated=turn_truncated)
                    active_hooks.on_turn_finished(sample, turn, turn.metadata)
                    return _finished(sample, episode, turn.status, extract_answer(parsed.text))

                request = require_single_tool_call(
                    calls, duration=duration, has_audio=media.has_audio, settings=config["media"]
                )
                turn.metadata["action"] = request.name
                if len(episode.turns) == max_turns:
                    raise ProtocolError("STEP_LIMIT_REACHED", "must answer on final step")
                if session.is_duplicate(request):
                    turn.metadata["no_progress"] = True
                    _record_completed_turn(episode, turn, role="tool", truncated=turn_truncated)
                    active_hooks.on_turn_finished(sample, turn, turn.metadata)
                    session.append_no_progress(request)
                    continue

                try:
                    bundle = await _run_blocking(media_executor, media.execute, request)
                except ProtocolError:
                    raise
                except _media_execution_errors() as exc:
                    # Drop the unexecuted turn so media decode errors never penalize
                    # a valid tool call with negative advantage.
                    episode.turns.pop()
                    logger.warning(
                        "Media execution failed (%s): task=%s",
                        type(exc).__name__,
                        sample.metadata.get("task_id"),
                    )
                    return _finished(
                        sample,
                        episode,
                        _resolve_status(sample, "FAILED"),
                        answer=None,
                        failure_reason="MEDIA_TOOL_FAILED",
                    )

                tool_observations = _extract_tool_observations(
                    request,
                    bundle,
                    duration=duration,
                    turn_index=int(turn.metadata["turn_index"]),
                    has_audio=media.has_audio,
                    settings=config["media"],
                )
                episode.media_state.record_covered(tool_observations)
                episode.media_state.budget = media.budget.to_dict()
                turn.metadata["tool_observations"] = tool_observations
                turn.metadata["cost_units"] = bundle.cost_units
                _record_completed_turn(episode, turn, role="tool", truncated=turn_truncated)
                active_hooks.on_tool_executed(sample, turn, request, bundle, tool_observations)
                active_hooks.on_turn_finished(sample, turn, turn.metadata)
                session.append_tool_result(request, bundle)
            except ProtocolError as error:
                turn.metadata.update(invalid=True, error_code=error.code)
                # Preserve answer isolation even if an answer turn had formatting errors.
                _record_completed_turn(episode, turn, role=intent_role, truncated=turn_truncated)
                active_hooks.on_turn_finished(sample, turn, turn.metadata)
                session.append_protocol_error(
                    error.code, str(error), calls=calls, max_invalid_retries=max_invalid_retries
                )
                if error.code == "STEP_LIMIT_REACHED":
                    return _finished(sample, episode, _resolve_status(sample, "FAILED"), answer=None)

        return _finished(sample, episode, _resolve_status(sample, "TRUNCATED"), answer=None)
    finally:
        if initial_preparation is not None and live_state is not None:
            _release_initial(live_state, sample.group_index, initial_preparation)
        await _run_blocking(media_executor, media.close)


__all__ = [
    "AgentSample",
    "ContextLengthExceeded",
    "Episode",
    "FallbackStatus",
    "InitialPreparation",
    "RolloutHooks",
    "apply_turn_generation_result",
    "ensure_valid_top_p_fields",
    "generate_episode",
    "safe_max_weight_version",
    "safe_min_weight_version",
    "safe_policy_lag",
]
