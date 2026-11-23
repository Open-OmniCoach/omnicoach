#!/usr/bin/env bash

ensure_uv() {
   if command -v uv >/dev/null 2>&1; then
      return
   fi

   local root="${BUILD_TMP_ROOT:-/tmp/${USER:-$(id -un)}/slime-ext-build}"
   local venv="${UV_BOOTSTRAP_VENV:-${root}/uv-bootstrap}"
   local python_bin="${UV_BOOTSTRAP_PYTHON:-python3}"
   local index_url="${PYPI_INDEX_URL_OVERRIDE:-${PYPI_INDEX_URL:-https://pypi.org/simple}}"

   mkdir -p "${root}"

   if [ ! -x "${venv}/bin/uv" ]; then
      printf '[slime_ext_env] uv not found; bootstrapping uv into %s\n' "${venv}"
      rm -rf "${venv}"
      if "${python_bin}" -m venv "${venv}"; then
         "${venv}/bin/python" -m pip install --upgrade pip setuptools wheel
         "${venv}/bin/python" -m pip install --index-url "${index_url}" uv
      else
         printf '[slime_ext_env] python venv bootstrap failed; falling back to user uv install\n' >&2
         "${python_bin}" -m pip install --user --index-url "${index_url}" uv
         export PATH="${HOME}/.local/bin:${PATH}"
      fi
   fi

   if [ -x "${venv}/bin/uv" ]; then
      export PATH="${venv}/bin:${PATH}"
   fi

   command -v uv >/dev/null 2>&1 || {
      printf '[slime_ext_env] ERROR: uv bootstrap failed\n' >&2
      return 1
   }
}
