[← Back to Open-OmniCoach](../README.md)

# Open-OmniCoach SFT

Agentic multimodal supervised fine-tuning for Qwen3-Omni (and Qwen2.5-Omni trace SFT) with Swift/Megatron, content-addressed Agent shared-media Parquet, fast packing length estimation, and exact turn-suffix loss scaling.

## Package Layout

| Path | Responsibility |
| --- | --- |
| `src/omnicoach_sft/` | Python package (`omnicoach_sft.media`, `omnicoach_sft.data`, `omnicoach_sft.train`) |
| `config/train/` | Flat training configs (for example `qwen3_omni_agent_sft.yaml`) |
| `config/pack_length/` | Token/packing length estimation profiles (`qwen3_omni_contract_v1.yaml`) |
| `scripts/` | General-purpose environment installer (`install.sh`) and training launcher (`train.sh`) |
| `tests/` | CPU verification suite for export, views, pooling, packing lengths, and loss scaling |

## Setup

Use Python 3.12 and the dependency pins in `pyproject.toml` and `uv.lock`. Run commands from `sft/`. Do not install training dependencies into the evaluation environment:

```bash
bash scripts/install.sh h100  # or a100 / b200 / train
```

For CPU-only unit checks:

```bash
uv pip install -e . --group test-cpu
pytest -v tests
```

## CLI Tools & Training

| Entrypoint | Module | Purpose |
| --- | --- | --- |
| `omnicoach-sft-export` | `omnicoach_sft.data.export` | Export Agent JSONL/evalkit trajectories to Agent shared-media Parquet |
| `omnicoach-sft-inspect` | `omnicoach_sft.data.inspect` | Validate and sample-inspect `turns/` + `media/` Agent Parquet artifacts |
| `omnicoach-sft-verify-encode` | `omnicoach_sft.train.verify_encode` | Pre-training processor/template/loss-scale check on sample Agent rows |

Launch training with `scripts/train.sh`, which runs `python -m omnicoach_sft sft` and honours standard `torchrun` environment variables (`MASTER_ADDR`, `MASTER_PORT`, `NODE_RANK`, `NNODES`, and `NPROC_PER_NODE`). `DRY_RUN=1` previews the launcher configuration without requiring GPUs or dataset files.

`config/train/qwen3_omni_agent_sft.yaml` holds the reported Agent-SFT recipe with placeholder `dataset:` roots. Use the [released OmniCoach-240K seed Parquet](../docs/omnicoach-240k.md#download-prebuilt-sft-parquet) or export your own Agent shared-media Parquet, then edit a copy of the config or pass `--dataset` to `scripts/train.sh`.

Licensed under the [Apache License 2.0](../LICENSE).
