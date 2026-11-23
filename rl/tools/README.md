# Tools

Reusable environment, patch-management, and checkpoint utilities for Qwen3-Omni RL training on `THUDM/slime`.

## `tools/env/` — Environment & Upstream Patch Tooling

| File | Purpose |
| --- | --- |
| `versions.env` | Pinned commits and package versions (`THUDM/slime`, `SGLang`, `Megatron-LM`, `Apex`, `Transformer Engine`, `nvidia-resiliency-ext`, `torch_memory_saver`, `sglang-router`, `CUDA_BACKEND`, `PYTHON_VERSION`). |
| `prepare_slime.sh` | Verifies (`check`) or applies (`apply`) `tools/env/patches/slime-omni.patch` onto the pinned `third_party/slime` checkout (override target via positional arg or `SLIME_DIR`). |
| `patches/` | Minimal source patches kept outside upstream trees: `slime-omni.patch` (Qwen3-Omni EPD port/encoder isolation and generic Turn-Level GSPO batch fields) and `sglang-qwen3-omni-epd.patch` (SGLang Qwen3-Omni EPD support). |
| `build_local_uv.sh` | Builds the pinned wheelhouse and local `.venv` via `uv` according to `versions.env` and `pyproject.toml`. Set `FLASH_ATTN_CUDA_ARCHS` explicitly if you want to restrict FlashAttention compilation to specific GPU architectures. |
| `ensure_uv.sh` | Bootstraps `uv` into a temporary helper venv when `uv` is not already on `PATH` (sourced by `build_local_uv.sh`). |
| `requirements.build-wheel` | Reference list of source-built wheels (`flash-attn`, `flash_qla`, `torch_memory_saver`, `apex`, `fake_int4_quant_cuda`). |

Common commands:

```bash
# Apply and verify the pinned Slime patch
bash tools/env/prepare_slime.sh apply
bash tools/env/prepare_slime.sh check

# Build a local GPU environment using tools/env/versions.env
bash tools/env/build_local_uv.sh

# Optionally restrict FlashAttention compilation to specific GPU architectures (e.g. H100=90, A100=80)
FLASH_ATTN_CUDA_ARCHS="90" bash tools/env/build_local_uv.sh
```

## `tools/checkpoints/` — Checkpoint Conversion Utilities

| File | Purpose |
| --- | --- |
| `convert_torch_dist_to_hf.py` | Applies `slime_ext` Megatron-to-HF Qwen3-Omni weight mappings and delegates CLI arguments unchanged to upstream `third_party/slime/tools/convert_torch_dist_to_hf.py`. |
| `convert_hf_to_fp8_qwen3_omni.py` | Converts Qwen3-Omni Thinker decoder (`--scope decoder`) or MoE experts (`--scope experts_only`) to blockwise FP8 while keeping multimedia towers, embeddings, normalizations, LM head, and routers in BF16. |

Example usage:

```bash
# Convert a torch_dist training checkpoint to Hugging Face format
python tools/checkpoints/convert_torch_dist_to_hf.py --help

# Convert a Hugging Face Qwen3-Omni checkpoint to blockwise FP8
python tools/checkpoints/convert_hf_to_fp8_qwen3_omni.py \
  --model-dir /path/to/Qwen3-Omni-30B-A3B \
  --save-dir /path/to/Qwen3-Omni-30B-A3B-FP8 \
  --slime-converter third_party/slime/tools/convert_hf_to_fp8.py \
  --scope decoder
```
