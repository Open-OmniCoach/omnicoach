#!/usr/bin/env bash
# Example 2: Multi-turn active-perception Qwen3-Omni Agent + outcome GSPO.
set -euo pipefail

EXAMPLE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${EXAMPLE_DIR}/../.." && pwd)
# shellcheck source=../_common/launch_qwen3_omni.sh
source "${ROOT_DIR}/examples/_common/launch_qwen3_omni.sh"

EXAMPLE_NAME=qwen3_omni_agent_gspo
PACKAGE=examples.qwen3_omni_agent_gspo
DEFAULT_DATA_HINT=/path/to/agent_tasks.jsonl
DATA_PATH_HELP="Set DATA_PATH to an agent task JSONL"
CONFIG_BASE=${AGENT_CONFIG:-${EXAMPLE_DIR}/config.yaml}

qwen3_omni_configure_agent_defaults
qwen3_omni_init_env
RECIPE_ARGS=()
qwen3_omni_append_agent_args
qwen3_omni_run "$@"
