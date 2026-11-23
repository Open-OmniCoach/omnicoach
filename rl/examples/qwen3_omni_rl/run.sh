#!/usr/bin/env bash
# Example 1: Single-turn Qwen3-Omni multimodal QA RL (GRPO / GSPO).
set -euo pipefail

EXAMPLE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${EXAMPLE_DIR}/../.." && pwd)
# shellcheck source=../_common/launch_qwen3_omni.sh
source "${ROOT_DIR}/examples/_common/launch_qwen3_omni.sh"

EXAMPLE_NAME=qwen3_omni_rl
PACKAGE=examples.qwen3_omni_rl
DEFAULT_DATA_HINT=/path/to/qwen3_omni_qa.jsonl
DATA_PATH_HELP="Set DATA_PATH to a single-turn multimodal JSONL"
CONFIG_BASE=${RL_CONFIG:-${EXAMPLE_DIR}/config.yaml}

ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-8}
PACK_TOKENS=${PACK_TOKENS:-32768}
LOG_PROBS_PACK_TOKENS=${LOG_PROBS_PACK_TOKENS:-16384}
FLUSH_CACHE_INTERVAL=${FLUSH_CACHE_INTERVAL:-1}

qwen3_omni_init_env
RESOLVED_ESTIMATOR=$("${PYTHON_BIN}" -c 'import sys, yaml; print(yaml.safe_load(sys.argv[1])["qwen3_omni_rl"]["advantage_estimator"])' "${CONFIG_YAML}")
RESOLVED_ABLATION=$("${PYTHON_BIN}" -c 'import sys, yaml; print(yaml.safe_load(sys.argv[1])["qwen3_omni_rl"]["ablation_mode"])' "${CONFIG_YAML}")
DATA_PREPARE_ARGS=(--ablation-mode "${RESOLVED_ABLATION}")
RECIPE_ARGS=(
    --rollout-function-path slime.rollout.sglang_rollout.generate_rollout
    --custom-rm-path "${PACKAGE}.reward.score"
    --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
    --apply-chat-template
    --rollout-max-context-len 32768 --rollout-max-prompt-len 28672 --rollout-max-response-len 2048
    --advantage-estimator "${RESOLVED_ESTIMATOR}"
    --sglang-mem-fraction-static 0.70 --sglang-context-length 32768
)
qwen3_omni_run "$@"
