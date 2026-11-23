# Example 2: `qwen3_omni_agent_gspo` (Multi-Turn Omni Agent + Pure Outcome GSPO)

Multi-turn active-perception Qwen3-Omni agent trained with pure outcome GSPO using `slime_ext.agent`, `python -m slime_ext.train` (`THUDM/slime` `79aa41a8`), and `examples/_common/launch_qwen3_omni.sh`.

- **Isolation**: Imports `slime_ext.agent` and `examples._common.agent_recipe`; zero cross-example imports.
- **No evidence annotations required**: Works on any MCQ / `span_set` dataset without `evidence_targets`.
- **Dry run**:
  ```bash
  DRY_RUN=1 bash examples/qwen3_omni_agent_gspo/run.sh
  ```
