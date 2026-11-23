#!/usr/bin/env bash
# Install the SFT Python environment from pyproject.toml + uv.lock.
# Usage: bash scripts/install.sh [h100|a100|b200|train]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-h100}"
case "$PROFILE" in
  h100|a100|b200|train) ;;
  *)
    echo "Usage: bash $0 [h100|a100|b200|train]" >&2
    exit 2
    ;;
esac

ENV_DIR="${OMNI_SFT_ENV_DIR:-$PROJECT_ROOT/.venv}"
export PATH="$HOME/.local/bin:$HOME/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required (https://docs.astral.sh/uv/installation/)" >&2
  exit 2
fi

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_PROJECT_ENVIRONMENT="$ENV_DIR"

cd "$PROJECT_ROOT"
uv venv "$ENV_DIR" --python 3.12 --allow-existing

# Build transformer-engine-torch in a second pass after the pinned PyTorch wheel
# is installed so build-isolation disabled builds see the exact torch ABI.
uv sync --frozen --no-default-groups --group "$PROFILE" --no-install-package transformer-engine-torch

if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
      | awk 'NF && !seen[$1]++ {print $1}' | paste -sd ';' - || true)"
  fi
  if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    case "$PROFILE" in
      a100) TORCH_CUDA_ARCH_LIST="8.0" ;;
      h100) TORCH_CUDA_ARCH_LIST="9.0" ;;
      b200) TORCH_CUDA_ARCH_LIST="10.0" ;;
      train) TORCH_CUDA_ARCH_LIST="" ;;
    esac
  fi
fi
if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  export TORCH_CUDA_ARCH_LIST
fi
export MAX_JOBS="${MAX_JOBS:-16}"

uv sync --frozen --no-default-groups --group "$PROFILE" --no-build-isolation-package transformer-engine-torch
echo "Install complete: $ENV_DIR ($PROFILE)"
