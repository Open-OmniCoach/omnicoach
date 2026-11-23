# Contributing to Open-OmniCoach

Thank you for your interest in contributing to **Open-OmniCoach**.

## Module Boundaries

Each top-level module maintains its own Python environment and test suite:

- `harness/` — Offline trace-collection runtime (`omnicoach_harness`, protocol `omni_ota_v1`).
- `sft/` — Agent shared-media Parquet builders, pack-length caching, and Swift/Megatron SFT launchers (`omnicoach_sft`).
- `rl/` — Reusable Qwen3-Omni RL extension (`slime_ext/`), `THUDM/slime` patch (`tools/env/patches/slime-omni.patch`), and decoupled training recipes (`examples/`).
- `evalkit/` — Maintained upstream in [Video Agent EvalKit](https://github.com/Open-OmniCoach/video-agent-evalkit) (`omnicoach` branch). Submit evaluation or single-video inference changes there and update the submodule pointer here afterward.

### Architecture Invariants in `rl/`

- `slime_ext/` must stay recipe-agnostic (zero imports of `examples/`).
- `examples/qwen3_omni_rl/` must not import `slime_ext.agent` or `examples.qwen3_omni_agent_gspo`.
- `examples/qwen3_omni_agent_gspo/` and `examples/_common/` must not import other recipes.

These invariants are enforced automatically by `rl/tests/examples/test_example_isolation.py`.

## Running Checks Locally

```bash
# Harness
cd harness && uv run ruff check src tests && uv run pytest -v tests

# SFT
cd sft && uv run --group test-cpu ruff check src tests && uv run --group test-cpu pytest -v tests

# RL (after populating & patching rl/third_party/slime per rl/README.md)
cd rl && ruff check slime_ext examples tools tests
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=".:third_party/slime" \
  pytest -v tests/core tests/agent tests/examples
```

## Pull Request Checklist

- Keep changes scoped to the relevant module and avoid cross-module dependency leaks.
- Add or update unit tests for bug fixes and new behavior.
- Ensure `ruff`, `compileall`, and the affected module's CPU test suite pass.
- Do not commit model weights, datasets, credentials, or environment-specific absolute paths.
