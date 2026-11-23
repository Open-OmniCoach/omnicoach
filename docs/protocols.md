# Tool Protocols: Trace Harness vs. Native Qwen-Omni Agent

[← Back to Open-OmniCoach](../README.md)

Open-OmniCoach intentionally keeps two wire formats cleanly decoupled:

| Dimension | `harness/` (`omnicoach_harness`, route `omni_ota_v1`) | `rl/slime_ext/agent/` & `sft/` & `evalkit/` |
| --- | --- | --- |
| **Primary use** | Offline teacher/student trace collection over heterogeneous OpenAI-compatible endpoints | On-policy multi-turn SFT / RL rollouts inside SGLang + Megatron and Open-OmniAgent evaluation (`evalkit/`) |
| **Turn envelope** | Strict JSON object `{"observation", "think", "action"}` via `response_format` | XML tags `<observation>...</observation>`, `<think>...</think>`, `<answer>...</answer>` + native `<tool_call>` |
| **Time format** | One-decimal `MM:SS.m` strings (e.g., `"01:12.5"`) | String seconds with an `"s"` suffix (e.g., `"72.5s"`) |
| **Chat template** | Standard OpenAI chat completions (no custom template required) | Qwen-Omni Hermes-style function calling (`rl/slime_ext/agent/chat_template.jinja`) |
| **Multi-span batching** | `get_multi_span` with `spans: [{"modality": "frames" \| "audio" \| "clip", "start": "MM:SS.m", "end": "MM:SS.m", ...}]` | `get_multi_span` with `spans: [{"modality": "frames" \| "audio" \| "clip", "start": "72.5s", "end": "80.0s", ...}]` |

## Why Two Wire Formats?

1. **Teacher compatibility (`harness/`)**: Frontier teacher endpoints vary in native function-calling support and multimodal tool-role handling. Enforcing a flat JSON schema (`{"observation", "think", "action"}`) guarantees deterministic parsing across any OpenAI-compatible API without custom chat templates.
2. **Native Qwen-Omni training (`rl/slime_ext/agent/` & `sft/`)**: Trained Open-OmniAgent models operate natively with Qwen's `<tool_call>` / `<tool_response>` tokens and `"Ns"` second-string arguments so token-level loss masks, reflection masking, and Turn-Level GSPO align cleanly with the tokenizer and SGLang engine.

## Shared Tool Semantics

- **`get_frames`**: Extracts video frames over a bounded interval at a discrete `level` (`"coarse"` = 1 FPS, `"detail"` = 2 FPS) and `resolution` (`"default"` or `"high"`).
- **`get_audio`**: Extracts audio over a bounded interval for acoustic, speech, or sound-event verification (64 kbps AAC `.m4a` in `harness/`; 16 kHz mono PCM WAV in `rl/` and `sft/`).
- **`get_clip`**: Extracts both video frames and synchronized same-span audio over an interval.
- **`get_multi_span`**: Dispatches multiple non-contiguous ordered sub-requests (`modality`: `"frames"`, `"audio"`, or `"clip"`) in a single assistant turn so the agent can sweep long videos in parallel without burning turns.
