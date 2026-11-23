# Serving Open-OmniAgent Models

[← Back to Open-OmniCoach](../README.md)

[`evalkit/`](https://github.com/Open-OmniCoach/video-agent-evalkit/tree/omnicoach) talks to standard OpenAI-compatible `/v1/chat/completions` endpoints using the native Open-OmniAgent template and tool protocol. You can serve compatible Qwen2.5-Omni-7B, Qwen3-Omni-30B-A3B, or released Open-OmniAgent checkpoints locally with **SGLang** or **vLLM**.

## Recommended Hardware

| Model | Base Architecture | Typical Serving Footprint |
| --- | --- | --- |
| **Open-OmniAgent-7B** | Qwen2.5-Omni-7B | 1× GPU (24–48 GB VRAM) |
| **Open-OmniAgent-30B-A3B** | Qwen3-Omni-30B-A3B-Instruct | 2×–4× 80 GB GPUs (TP=2 or TP=4) |

## Option A: Serve with SGLang

For `Qwen3-Omni-30B-A3B` / `Open-OmniAgent-30B-A3B`:

```bash
python -m sglang.launch_server \
  --model-path /path/to/Open-OmniAgent-30B-A3B \
  --served-model-name Open-OmniAgent-30B \
  --host 0.0.0.0 \
  --port 8000 \
  --tp 4 \
  --trust-remote-code
```

## Option B: Serve with vLLM / vLLM-Omni

For `Qwen2.5-Omni-7B` / `Open-OmniAgent-7B`:

```bash
vllm serve /path/to/Open-OmniAgent-7B \
  --served-model-name Open-OmniAgent-7B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --limit-mm-per-prompt '{"image": 64, "audio": 16}' \
  --trust-remote-code
```

## Verify the Endpoint

```bash
curl http://127.0.0.1:8000/v1/models
```

Then run single-video inference or full benchmark evaluation via [`evalkit/`](https://github.com/Open-OmniCoach/video-agent-evalkit/blob/omnicoach/examples/omni_agent/README.md) (`omnicoach` branch):

```bash
cd evalkit
uv sync --locked
# See evalkit/examples/omni_agent/README.md for Qwen2.5-Omni & Qwen3-Omni configs
```
