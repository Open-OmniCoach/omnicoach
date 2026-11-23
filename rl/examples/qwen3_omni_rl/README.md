# Example 1: `qwen3_omni_rl` (Single-Turn Multimodal QA RL)

Minimal single-turn Qwen3-Omni multimodal RL recipe using `python -m slime_ext.train` (`THUDM/slime` `79aa41a8`) with GRPO or GSPO via `examples/_common/launch_qwen3_omni.sh`.

- **Isolation**: Zero imports from `slime_ext.agent` or other examples.
- **Tasks**: Single-turn multimodal MCQ, audio-visual match/mismatch, and temporal IoU grounding.
- **Dry run**:
  ```bash
  DRY_RUN=1 bash examples/qwen3_omni_rl/run.sh
  ```
