#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYPI_INDEX_URL="${PYPI_INDEX_URL_OVERRIDE:-https://pypi.org/simple}"

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export PIP_INDEX_URL="$PYPI_INDEX_URL"
export UV_INDEX_URL="$PYPI_INDEX_URL"
unset PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL

bash "${SCRIPT_DIR}/tools/env/prepare_slime.sh" check

echo "slime_ext: using interpreter ${PYTHON_BIN}"

if command -v uv >/dev/null 2>&1; then
   uv pip install --python "${PYTHON_BIN}" --link-mode "${UV_LINK_MODE}" --index-url "${PYPI_INDEX_URL}" -e "${SCRIPT_DIR}" --no-build-isolation
else
   "${PYTHON_BIN}" -m pip install -e "${SCRIPT_DIR}" --no-build-isolation --index-url "${PYPI_INDEX_URL}"
fi

"${PYTHON_BIN}" -m slime_ext.install_hook

echo "slime_ext: install complete."
