#!/usr/bin/env bash
# Generic Swift/Megatron SFT launcher. Experiment behavior belongs in YAML.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

ENV_DIR="${OMNI_SFT_ENV_DIR:-$PROJECT_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_DIR/bin/python}"
DRY_RUN="${DRY_RUN:-0}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml> [extra swift args...]" >&2
  exit 2
fi

CONFIG_INPUT="$1"
shift
if [[ "$CONFIG_INPUT" == /* ]]; then
  CONFIG_PATH="$CONFIG_INPUT"
else
  CONFIG_PATH="$PROJECT_ROOT/$CONFIG_INPUT"
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 2
fi

if [[ "$DRY_RUN" == "1" && ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Python venv not found: ${OMNI_SFT_ENV_DIR:-$PROJECT_ROOT/.venv}/bin/python" >&2
  echo "Run bash scripts/install.sh before training." >&2
  exit 3
fi

CONFIG_DEFAULTS="$("$PYTHON_BIN" -m omnicoach_sft.train.config --launcher-defaults "$CONFIG_PATH" "$DRY_RUN" "$@")"
mapfile -t CONFIG_DEFAULTS <<<"$CONFIG_DEFAULTS"
IS_QWEN25_TRACE="${CONFIG_DEFAULTS[0]:-0}"
export MODEL_SEQ_LEN="${MODEL_SEQ_LEN:-${CONFIG_DEFAULTS[1]:-65536}}"
if [[ "$IS_QWEN25_TRACE" == "1" ]]; then
  export ENABLE_AUDIO_OUTPUT="${ENABLE_AUDIO_OUTPUT:-0}"
fi

# Standard torchrun topology; override in multi-node jobs or schedulers.
export NODE_RANK="${NODE_RANK:-0}"
export NNODES="${NNODES:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-23333}"

# Essential Swift / Megatron multimodal runtime defaults (all overridable).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export USE_MCORE_GDN="${USE_MCORE_GDN:-1}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export SWIFT_AUDIO_LOAD_BACKEND="${SWIFT_AUDIO_LOAD_BACKEND:-soundfile_pyav}"
export VIDEO_MIN_TOKEN_NUM="${VIDEO_MIN_TOKEN_NUM:-2}"
export SWIFT_DATASET_MAP_BATCH_SIZE="${SWIFT_DATASET_MAP_BATCH_SIZE:-16}"

if [[ "$DRY_RUN" != "1" ]]; then
  CUDA_PY_SITE_PACKAGES="$("$PYTHON_BIN" -m omnicoach_sft.train.config --cuda-site-packages)"
  if [[ -n "$CUDA_PY_SITE_PACKAGES" ]]; then
    export CUDNN_HOME="${CUDNN_HOME:-$CUDA_PY_SITE_PACKAGES/nvidia/cudnn/lib}"
    export NVRTC_HOME="${NVRTC_HOME:-$CUDA_PY_SITE_PACKAGES/nvidia/cuda_nvrtc/lib}"
    export CURAND_HOME="${CURAND_HOME:-$CUDA_PY_SITE_PACKAGES/nvidia/curand/lib}"
  fi
fi

echo "Config:          $CONFIG_PATH"
echo "Python:          $PYTHON_BIN"
echo "Topology:        rank=$NODE_RANK nnodes=$NNODES nproc_per_node=$NPROC_PER_NODE"
echo "Rendezvous:      $MASTER_ADDR:$MASTER_PORT"
echo "Model seq len:   $MODEL_SEQ_LEN"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1, not launching training."
  exit 0
fi

exec "$PYTHON_BIN" -m omnicoach_sft sft "$CONFIG_PATH" "$@"
