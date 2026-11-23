[← Back to Open-OmniCoach](../README.md)

# Open-OmniCoach RL (`slime_ext`)

Reusable Qwen3-Omni multimodal reinforcement learning extensions, a multi-turn active-perception agent runtime, Turn-Level GSPO support, and standalone training examples built on top of [`THUDM/slime`](https://github.com/THUDM/slime) (`79aa41a884ab0bf65222c94418f6357d3c31192b`).

## Overview

`slime_ext` keeps upstream `THUDM/slime` nearly untouched (a small EPD & TG-GSPO patch in `tools/env/patches/slime-omni.patch`) and separates Qwen3-Omni model, backend, agent, and training-recipe code into acyclic layers verified by `tests/examples/test_example_isolation.py`:

- **`slime_ext/`** — Installable extension package for Qwen3-Omni on Megatron-Core (`0.20.0`) + SGLang (`v0.5.21`) + Transformer Engine (`2.18.0`), including frozen-encoder feature caching, CPU mRoPE, packed THD/CP support, rolling async `nvidia-resiliency-ext` checkpoints, Turn-Level GSPO (`--use-turn-level-gspo`), and a shared multi-turn active-perception agent runtime (`slime_ext.agent`). Carries no coupling to `examples/`.
- **`examples/`** — Two isolated training recipes sharing common launcher helpers in `examples/_common/`:
  1. `examples/qwen3_omni_rl/` — Single-turn multimodal QA RL (GRPO / GSPO).
  2. `examples/qwen3_omni_agent_gspo/` — Multi-turn active-perception agent with outcome GSPO.
- **`tools/`** — Portable environment setup/patch utilities (`tools/env/`) and checkpoint conversion/prefetch utilities (`tools/checkpoints/`). See [`tools/README.md`](tools/README.md).

## Repository Architecture

```text
.
├── README.md
├── pyproject.toml                  Package metadata (`slime-ext`), CUDA 13 uv env group, ruff & pytest config
├── MANIFEST.in
├── install.sh                      Editable install + `slime_ext` autoload hook installer
├── third_party/
│   └── slime/                      Pinned THUDM/slime 79aa41a8 + EPD & TG-GSPO patch (ignored by git)
├── slime_ext/                      Reusable Qwen3-Omni + Agent extension package (no recipe coupling)
│   ├── __init__.py                 Idempotent apply() registering Megatron, SGLang, and weight hooks
│   ├── install_hook.py             Install/uninstall `--uninstall` helper for startup autoload `.pth`
│   ├── train.py                    Unified CLI entrypoint (`python -m slime_ext.train`)
│   ├── backends/
│   │   ├── megatron/               Actor, arguments, CPU mRoPE, data, encoder cache, loss (TG-GSPO),
│   │   │                           packed THD/CP, rolling checkpoint, weight sync
│   │   └── sglang/                 FP8, processor (`ExactTokenQwen3OmniProcessor`), top-p transport
│   ├── models/qwen3_omni/          Audio, vision, text, thinker, RoPE, parallel, provider, weights/
│   └── agent/                      Reusable multi-turn active-perception agent runtime and templates
├── examples/                       Decoupled training recipes (see examples/README.md)
│   ├── _common/                    Shared training launcher helpers
│   ├── qwen3_omni_rl/              Example 1: Single-turn multimodal QA RL (GRPO / GSPO)
│   └── qwen3_omni_agent_gspo/      Example 2: Multi-turn agent + outcome GSPO
├── tools/
│   ├── README.md
│   ├── env/                        `versions.env`, `prepare_slime.sh`, `build_local_uv.sh`, `ensure_uv.sh`,
│   │                               `requirements.build-wheel`, `patches/`
│   └── checkpoints/                `convert_torch_dist_to_hf.py`, `convert_hf_to_fp8_qwen3_omni.py`
└── tests/
    ├── core/                       Core upgrade, packed THD, weights, rolling checkpoint, TG-GSPO, apply, parallel
    ├── agent/                      Shared agent runtime and Agent-GSPO contracts
    ├── examples/                   AST isolation and example recipe contracts
    └── gpu/                        GPU / Megatron-Core / SGLang / Ray integration tests
```

## Setup

Run all commands below from the `rl/` directory.

### 1. Populate pinned `third_party/slime`

`third_party/slime/` is git-ignored so upstream history stays separate from extension commits. From `rl/`, check out the exact commit pinned in `tools/env/versions.env` and apply the patch:

```bash
cd rl

source tools/env/versions.env
git clone https://github.com/THUDM/slime.git third_party/slime
git -C third_party/slime checkout "${SLIME_COMMIT}"

bash tools/env/prepare_slime.sh apply
bash tools/env/prepare_slime.sh check
```

### 2. Install `slime_ext`

Inside an environment that already has the GPU dependencies installed (or after building one with `tools/env/build_local_uv.sh`):

```bash
bash install.sh
```

`install.sh` verifies the `third_party/slime` patch, installs this package in editable mode (`-e .`), and registers the `slime_ext` autoload hook (`python -m slime_ext.install_hook --uninstall` removes it).

## Quick Start: Examples

| Example | Entrypoint | Training Objective | Dependencies |
| --- | --- | --- | --- |
| `qwen3_omni_rl` | `examples/qwen3_omni_rl/run.sh` | Single-turn GRPO / GSPO | `slime` + Omni core only |
| `qwen3_omni_agent_gspo` | `examples/qwen3_omni_agent_gspo/run.sh` | Multi-turn outcome GSPO | `slime_ext.agent` |

### Dry-Run Command Rendering (No Cluster Required)

Both launchers build on `examples/_common/`, invoke `python -m slime_ext.train`, support `DRY_RUN=1` rendering without a Ray cluster, and support optional rolling async checkpoints via `LATEST_SAVE_INTERVAL`:

```bash
DRY_RUN=1 bash examples/qwen3_omni_rl/run.sh
DRY_RUN=1 bash examples/qwen3_omni_agent_gspo/run.sh

# Example with rolling async checkpoints every 5 steps, retaining every 100 steps:
DRY_RUN=1 SAVE_INTERVAL=100 LATEST_SAVE_INTERVAL=5 bash examples/qwen3_omni_agent_gspo/run.sh
```

### Launching on a GPU Ray Cluster

```bash
export RAY_ADDRESS="auto"
export INIT_MODEL_DIR="/path/to/Qwen3-Omni-30B-A3B"
export DATA_PATH="/path/to/tasks.jsonl"
export ROLLOUT_DATA_TRANSPORT="object-store"  # or straw

bash examples/qwen3_omni_agent_gspo/run.sh
```

See [`examples/README.md`](examples/README.md) and each example's `README.md` / `config.yaml` for task schemas and tunable environment variables.

## CPU Verification

The core, agent, and example isolation/contract suites run on CPU (only PyTorch and lightweight Python dependencies required):

```bash
bash tools/env/prepare_slime.sh check

CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=".:third_party/slime" \
  pytest -v tests/core tests/agent tests/examples
```

## GPU Environment Notes

Pinned GPU stack versions live in [`tools/env/versions.env`](tools/env/versions.env) and align with upstream `THUDM/slime` (`79aa41a8`):

- **CUDA backend**: `cu130` (Python `3.12`)
- **Megatron-LM**: `ae76c60e6c35d913867d6dd7bad0f30b128b54e7` (`megatron-core 0.20.0`)
- **SGLang**: `v0.5.21` (with EPD patch in `tools/env/patches/sglang-qwen3-omni-epd.patch`)
- **Transformer Engine**: `2.18.0`
- **`nvidia-resiliency-ext`**: `0.7.0` (async persistent checkpoint backend)
- **`torch_memory_saver`**: `bdab8c55f490a00a746ec34e301391617f5ca878`

Use `bash tools/env/build_local_uv.sh` (optionally setting `FLASH_ATTN_CUDA_ARCHS` for your GPU architecture) to build and verify a matching GPU venv, then run `pytest -v tests/gpu`.

## License

This module is released under the [Apache License 2.0](../LICENSE).

## Acknowledgements

Built upon and inspired by:

- [`THUDM/slime`](https://github.com/THUDM/slime)
- [`SGLang`](https://github.com/sgl-project/sglang)
- [`NVIDIA/Megatron-LM`](https://github.com/NVIDIA/Megatron-LM)
- [`Qwen3-Omni`](https://github.com/QwenLM/Qwen3-Omni)
