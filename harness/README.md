[← Back to Open-OmniCoach](../README.md)

# Open-OmniCoach Harness

`omnicoach_harness` runs the frozen Observe-Think-Action JSON protocol
(`omni_ota_v1`) over active-perception media tools and persists each episode as
a structured JSONL artifact for offline trace collection.

Supply your own OpenAI-compatible model endpoint, task file, and media; no
end-to-end run or bundled media is included in this checkout.

## Install and configure a provider

`decord` is installed automatically on non-macOS platforms; on macOS or when
Decord fails to decode a file, media decoding falls back to PyAV (and then
per-frame FFmpeg).

```bash
cd harness
uv sync --locked --extra dev
uv run omnicoach-harness doctor
uv run omnicoach-harness describe --entrypoint omnicoach_harness.protocol.route:build
```

The only backend, `openai_compat`, takes any OpenAI-compatible
chat-completions endpoint you supply, such as vLLM, SGLang or a hosted API.
Copy [`examples/actor_openai_compat.yaml`](examples/actor_openai_compat.yaml),
set `provider.base_url` and `provider.model`, point `io.input` at a task file
shaped like [`examples/tasks.jsonl`](examples/tasks.jsonl), then run
`uv run omnicoach-harness actor --config your-run.yaml`.

Images travel as base64 data URLs on `image_url` parts. Audio parts default to
`provider.audio_part_type: audio_url` (`type: audio_url`), and can also be
configured as `input_audio` (base64 + format) or `image_url` for endpoints or
gateways requiring those part schemas.

| Package directory under `src/omnicoach_harness/` | Responsibility |
| --- | --- |
| `protocol/` | Frozen `omni_ota_v1` Observe-Think-Action actor protocol |
| `domain/`, `artifacts/` | Neutral types and persisted stage artifacts |
| `media/` | Media decoding, preparation and probing |
| `providers/` | OpenAI-compatible transport and budget telemetry |
| `runtime/` | Execution, retries and resumable batch runs |

## Relationship to the RL training protocol

This harness and `rl/slime_ext/agent/` serve different stages of the OmniCoach
pipeline and intentionally use different wire formats for the same underlying
active-perception tools (`get_frames`, `get_audio`, `get_clip`,
`get_multi_span`):

- **`harness/` (data collection / offline rollouts):** uses strict-JSON
  Observe-Think-Action (`{"observation", "think", "action"}`) enforced via
  `response_format` with flat action objects and exact-one-decimal `MM:SS.m`
  timecodes so heterogeneous OpenAI-compatible teacher endpoints follow the
  contract without requiring a custom chat template.
- **`rl/slime_ext/agent/` (on-policy RL training):** uses XML-delimited
  `<observation>...</observation>`, `<think>...</think>`, and
  `<answer>...</answer>` turns paired with native `<tool_call>` function
  calling and `"Ns"` second-string arguments matching the Qwen-Omni chat
  template (`rl/slime_ext/agent/chat_template.jinja`).

## Scope of this checkout

One actor route ships here: `omni_ota_v1` (`omnicoach_harness.protocol.route:build`).
The route emits JSONL run artifacts. For 7B/30B checkpoint inference and
evaluation, use [Video Agent EvalKit (`omnicoach` branch)](https://github.com/Open-OmniCoach/video-agent-evalkit/tree/omnicoach).

## License

Licensed under the [Apache License 2.0](../LICENSE).
