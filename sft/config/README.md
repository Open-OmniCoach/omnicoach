# Configuration Layout

- `train/`: self-contained flat training configs (for example `train/qwen3_omni_agent_sft.yaml`).
- `pack_length/`: multimodal packing-length estimation profiles (`pack_length/qwen3_omni_contract_v1.yaml`).

## Running Training

Launch a flat training config from `sft/`:

```bash
bash scripts/train.sh config/train/qwen3_omni_agent_sft.yaml --dataset /path/to/agent_parquet_dir
```

Multiple Agent shared-media Parquet directories can be passed directly in the YAML `dataset:` list, via `--dataset ...` CLI overrides, or grouped into a YAML manifest referencing validated `turns/` + `media/` roots (expanded automatically by `expand_agent_manifests`).
