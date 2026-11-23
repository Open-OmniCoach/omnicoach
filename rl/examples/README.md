# Qwen3-Omni RL & Active-Perception Agent Examples (`THUDM/slime` `79aa41a8`)

Two decoupled examples built on a strict DAG with zero cross-example imports, shared launcher/config helpers in `examples/_common/`, and the unified `python -m slime_ext.train` entrypoint:

| Dimension | Example 1: `qwen3_omni_rl` | Example 2: `qwen3_omni_agent_gspo` |
| --- | --- | --- |
| Paradigm | Single-turn multimodal QA RL | Multi-turn active-perception Agent |
| Objective | GRPO / GSPO (pure outcome) | Outcome GSPO |
| Rollout Engine | `slime.rollout.sglang_rollout` | `slime.rollout.fully_async_rollout` + `slime_ext.agent` |
| Shared Dependencies | `_common/launch_qwen3_omni.sh` (`slime` only) | `slime_ext.agent` + `_common/` |
| Launch Script | `qwen3_omni_rl/run.sh` | `qwen3_omni_agent_gspo/run.sh` |

## Verification (CPU)

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:third_party/slime pytest -v tests/core tests/agent tests/examples
CUDA_VISIBLE_DEVICES="" PYTHONPATH=.:third_party/slime python3 -m unittest discover -s tests/examples -v

# Dry-run launch rendering
DRY_RUN=1 bash examples/qwen3_omni_rl/run.sh
DRY_RUN=1 bash examples/qwen3_omni_agent_gspo/run.sh
```
