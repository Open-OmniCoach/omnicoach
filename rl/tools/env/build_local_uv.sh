#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/versions.env"
source "${SCRIPT_DIR}/ensure_uv.sh"

BUILD_TMP_ROOT=${BUILD_TMP_ROOT:-/tmp/${USER:-$(id -un)}/slime-omni-build}
DEPS_DIR=${DEPS_DIR:-${REPO_ROOT}/.deps}
WHEELHOUSE=${WHEELHOUSE:-${BUILD_TMP_ROOT}/wheelhouse}
VENV_DIR=${VENV_DIR:-${HOME}/.venv/slime-omni}
SLIME_DIR=${SLIME_DIR:-${REPO_ROOT}/third_party/slime}
SGLANG_DIR=${SGLANG_DIR:-${DEPS_DIR}/sglang}
MEGATRON_DIR=${MEGATRON_DIR:-${DEPS_DIR}/Megatron-LM}
APEX_DIR=${APEX_DIR:-${DEPS_DIR}/apex}
PYPI_INDEX_URL=${PYPI_INDEX_URL_OVERRIDE:-https://pypi.org/simple}
PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/${CUDA_BACKEND}}
SGLANG_WHL_INDEX_URL=${SGLANG_WHL_INDEX_URL:-https://docs.sglang.ai/whl/${CUDA_BACKEND}/}
UV_INDEX_STRATEGY=${UV_INDEX_STRATEGY:-unsafe-best-match}
UV_LINK_MODE=${UV_LINK_MODE:-copy}
CLEAR_VENV=${CLEAR_VENV:-0}
CLEAR_WHEELHOUSE=${CLEAR_WHEELHOUSE:-1}
SKIP_SMOKE=${SKIP_SMOKE:-0}
CUDA_SOURCE=${CUDA_SOURCE:-auto}

BUILD_PARALLEL=${BUILD_PARALLEL:-32}
if [ -n "${FLASH_ATTN_CUDA_ARCHS:-}" ]; then
   export FLASH_ATTN_CUDA_ARCHS
else
   unset FLASH_ATTN_CUDA_ARCHS
fi
FLASH_ATTN_MAX_JOBS=${FLASH_ATTN_MAX_JOBS:-${BUILD_PARALLEL}}
FLASH_ATTN_NVCC_THREADS=${FLASH_ATTN_NVCC_THREADS:-2}
TE_BUILD_MAX_JOBS=${TE_BUILD_MAX_JOBS:-${BUILD_PARALLEL}}
APEX_PARALLEL=${APEX_PARALLEL:-${BUILD_PARALLEL}}
APEX_NVCC_THREADS=${APEX_NVCC_THREADS:-4}

export UV_LINK_MODE PIP_DISABLE_PIP_VERSION_CHECK=1
export CARGO_HOME=${CARGO_HOME:-${BUILD_TMP_ROOT}/cargo}
export RUSTUP_HOME=${RUSTUP_HOME:-${BUILD_TMP_ROOT}/rustup}
export PIP_INDEX_URL="${PYPI_INDEX_URL}"
export UV_INDEX_URL="${PYPI_INDEX_URL}"
export PATH="${VENV_DIR}/bin:${DEPS_DIR}/bin:${CARGO_HOME}/bin:${PATH}"
unset PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL VIRTUAL_ENV
ensure_uv

PYTHON="${VENV_DIR}/bin/python"
RESOLVE_LOCK="${BUILD_TMP_ROOT}/requirements.resolve.lock"

log() { printf '[build_slime_omni_env] %s\n' "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

uv_install() {
   uv pip install --python "${PYTHON}" --link-mode "${UV_LINK_MODE}" \
      --index-strategy "${UV_INDEX_STRATEGY}" --index-url "${PYPI_INDEX_URL}" "$@"
}

pip_wheel() {
   "${PYTHON}" -m pip wheel -w "${WHEELHOUSE}" "$@"
}

ensure_repo() {
   local url="$1" dir="$2" ref="$3" recursive="${4:-0}"
   if [ ! -d "${dir}/.git" ]; then
      mkdir -p "$(dirname "${dir}")"
      git clone "${url}" "${dir}"
   fi
   git -C "${dir}" fetch --all --tags
   git -C "${dir}" checkout "${ref}"
   if [ "${recursive}" = "1" ]; then
      git -C "${dir}" submodule update --init --recursive
   fi
}

apply_repo_patch() {
   local repo="$1" patch_file="$2" label="$3"
   [ -f "${patch_file}" ] || die "missing ${label} patch: ${patch_file}"
   (
      cd "${repo}"
      if git apply --check "${patch_file}" 2>/dev/null; then
         git apply "${patch_file}"
         if git grep -n '^<<<<<<< '; then
            die "${label} patch has conflicts"
         fi
      elif git apply --reverse --check "${patch_file}" 2>/dev/null; then
         log "${label} patch already applied"
      else
         die "${label} patch does not match this checkout"
      fi
   )
}

create_venv() {
   uv python install "${PYTHON_VERSION}"
   [ "${CLEAR_VENV}" = "1" ] && rm -rf "${VENV_DIR}"
   if [ -x "${PYTHON}" ]; then
      log "reuse venv: ${VENV_DIR}"
      return
   fi
   local uv_python_bin
   uv_python_bin="$(env -u VIRTUAL_ENV uv python find --managed-python --no-project --resolve-links "${PYTHON_VERSION}")"
   mkdir -p "$(dirname "${VENV_DIR}")"
   "${uv_python_bin}" -m venv --copies "${VENV_DIR}"
}

detect_pip_cuda_home() {
   "${PYTHON}" - <<'PY'
import pathlib
import site

for root in site.getsitepackages():
    candidate = pathlib.Path(root) / "nvidia" / "cuda_nvcc"
    if (candidate / "bin" / "nvcc").exists():
        print(candidate)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

detect_system_cuda_home() {
   if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
      printf '%s\n' "${CUDA_HOME}"
      return 0
   fi
   local nvcc
   for nvcc in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
      [ -x "${nvcc}" ] && dirname "$(dirname "${nvcc}")" && return 0
   done
   command -v nvcc >/dev/null 2>&1 && dirname "$(dirname "$(command -v nvcc)")"
}

detect_cuda_lib_path() {
   "${PYTHON}" - <<'PY'
import pathlib
import site

paths = []
for root in site.getsitepackages():
    nvidia_root = pathlib.Path(root) / "nvidia"
    if nvidia_root.exists():
        paths.extend(str(path) for path in nvidia_root.glob("*/lib") if path.exists())
print(":".join(paths))
PY
}

set_cuda_env() {
   if [ "${CUDA_SOURCE}" = "system" ]; then
      CUDA_HOME="$(detect_system_cuda_home)" || die "CUDA_SOURCE=system but nvcc was not found"
   elif [ "${CUDA_SOURCE}" = "auto" ] || [ "${CUDA_SOURCE}" = "pip" ]; then
      CUDA_HOME="$(detect_pip_cuda_home)" || {
         [ "${CUDA_SOURCE}" = "auto" ] && CUDA_HOME="$(detect_system_cuda_home)" || die "pip CUDA nvcc was not found"
      }
   else
      die "unsupported CUDA_SOURCE=${CUDA_SOURCE}"
   fi
   export CUDA_HOME
   export PATH="${CUDA_HOME}/bin:${PATH}"
   local cuda_lib_path
   cuda_lib_path="$(detect_cuda_lib_path)"
   export LD_LIBRARY_PATH="${cuda_lib_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
   export LIBRARY_PATH="${cuda_lib_path}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
   "${CUDA_HOME}/bin/nvcc" --version
}

ensure_rust() {
   if command -v rustc >/dev/null 2>&1 && command -v cargo >/dev/null 2>&1; then
      rustc --version
      cargo --version
      return
   fi
   mkdir -p "${CARGO_HOME}" "${RUSTUP_HOME}"
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --profile minimal --default-toolchain stable
   local rust_bin
   rust_bin="$(find "${RUSTUP_HOME}/toolchains" -path '*/bin/rustc' -print -quit | xargs -r dirname)"
   [ -n "${rust_bin}" ] || die "could not install rust toolchain"
   export PATH="${rust_bin}:${PATH}"
   rustc --version
   cargo --version
}

ensure_protoc() {
   [ -n "${PROTOC:-}" ] && [ -x "${PROTOC}" ] && return
   if command -v protoc >/dev/null 2>&1; then
      PROTOC="$(command -v protoc)"
   else
      mkdir -p "${DEPS_DIR}/bin"
      cat >"${DEPS_DIR}/bin/protoc" <<EOF
#!/usr/bin/env bash
exec "${PYTHON}" -m grpc_tools.protoc "\$@"
EOF
      chmod +x "${DEPS_DIR}/bin/protoc"
      PROTOC="${DEPS_DIR}/bin/protoc"
   fi
   export PROTOC
}

patch_apex_cuda_check() {
   local setup_py="${APEX_DIR}/setup.py"
   if grep -q 'if (bare_metal_version.major != torch_binary_version.major):' "${setup_py}"; then
      return
   fi
   if grep -q 'if (bare_metal_version != torch_binary_version):' "${setup_py}"; then
      sed -i 's/if (bare_metal_version != torch_binary_version):/if (bare_metal_version.major != torch_binary_version.major):/' "${setup_py}"
      return
   fi
   die "could not patch apex CUDA version check in ${setup_py}"
}

prepare_sources() {
   mkdir -p "${DEPS_DIR}"
   bash "${SCRIPT_DIR}/prepare_slime.sh" apply "${SLIME_DIR}"
   ensure_repo https://github.com/sgl-project/sglang.git "${SGLANG_DIR}" "${SGLANG_REF}"
   apply_repo_patch "${SGLANG_DIR}" "${SLIME_DIR}/docker/patch/${PATCH_VERSION}/sglang.patch" sglang
   apply_repo_patch "${SGLANG_DIR}" "${SLIME_DIR}/docker/patch/${PATCH_VERSION}/sglang-top_p.patch" sglang-top-p
   apply_repo_patch "${SGLANG_DIR}" "${REPO_ROOT}/tools/env/patches/sglang-qwen3-omni-epd.patch" sglang-qwen3-omni-epd
   ensure_repo https://github.com/NVIDIA/Megatron-LM.git "${MEGATRON_DIR}" "${MEGATRON_COMMIT}" 1
   apply_repo_patch "${MEGATRON_DIR}" "${SLIME_DIR}/docker/patch/${PATCH_VERSION}/megatron.patch" megatron
   ensure_repo https://github.com/NVIDIA/apex.git "${APEX_DIR}" "${APEX_COMMIT}"
   patch_apex_cuda_check
}

link_deps_dir() {
   local link="${REPO_ROOT}/.deps"
   local resolved_deps resolved_link
   resolved_deps="$(cd "${DEPS_DIR}" && pwd)"
   if [ -d "${link}" ] && [ ! -L "${link}" ]; then
      resolved_link="$(cd "${link}" && pwd)"
      if [ "${resolved_deps}" = "${resolved_link}" ]; then
         return 0
      fi
      die "${link} exists and is not a symlink"
   elif [ -e "${link}" ] && [ ! -L "${link}" ]; then
      die "${link} exists and is not a symlink"
   fi
   ln -sfn "${DEPS_DIR}" "${link}"
}

compile_lock() {
   SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="${SGLANG_REF#v}" \
      uv pip compile --project "${REPO_ROOT}" --group env -o "${RESOLVE_LOCK}"
}

validate_resolve_lock() {
   if [ "${CUDA_BACKEND}" = "cu130" ]; then
      if grep -E '^(transformer-engine-cu12|nvidia-cuda-nvcc-cu12|nvidia-cuda-runtime-cu12)==|\+cu12[0-9]' "${RESOLVE_LOCK}"; then
         die "resolved conflicting CUDA 12 package(s) in ${RESOLVE_LOCK} for CUDA_BACKEND=${CUDA_BACKEND}"
      fi
   fi
   grep -Eq "^torch==.*\+${CUDA_BACKEND}$" "${RESOLVE_LOCK}" || die "missing torch +${CUDA_BACKEND} pin in ${RESOLVE_LOCK}"
   grep -Eq "^transformer-engine==${TRANSFORMER_ENGINE_VERSION}$" "${RESOLVE_LOCK}" || die "missing transformer-engine ${TRANSFORMER_ENGINE_VERSION} in ${RESOLVE_LOCK}"
   grep -Eq "^nvidia-resiliency-ext==${NVRX_VERSION}$" "${RESOLVE_LOCK}" || die "missing nvidia-resiliency-ext ${NVRX_VERSION} in ${RESOLVE_LOCK}"
   grep -Eq '^flash-linear-attention==0\.5\.2$' "${RESOLVE_LOCK}" || die "missing flash-linear-attention 0.5.2 in ${RESOLVE_LOCK}"
   grep -Eq '^tilelang==0\.1\.12$' "${RESOLVE_LOCK}" || die "missing tilelang 0.1.12 in ${RESOLVE_LOCK}"
   grep -Eq '^blake3==' "${RESOLVE_LOCK}" || die "missing blake3 in ${RESOLVE_LOCK}"
   grep -Eq '^xxhash==' "${RESOLVE_LOCK}" || die "missing xxhash in ${RESOLVE_LOCK}"
   grep -Eq '^zstandard==' "${RESOLVE_LOCK}" || die "missing zstandard in ${RESOLVE_LOCK}"
}

install_build_env() {
   local build_env_lock="${BUILD_TMP_ROOT}/requirements.build-env.lock"
   grep -Ev '^(antlr4-python3-runtime|pylatexenc|transformer-engine-torch)==|^transformer-engine-torch @' \
      "${RESOLVE_LOCK}" >"${build_env_lock}"
   uv_install --no-deps -r "${build_env_lock}" \
      --extra-index-url "${PYTORCH_INDEX_URL}" \
      --extra-index-url "${SGLANG_WHL_INDEX_URL}"
   set_cuda_env
   ensure_rust
   ensure_protoc
}

reuse_wheel() {
   local pattern="$1"
   find "${WHEELHOUSE}" -maxdepth 1 -name "${pattern}" -print -quit | grep -q .
}

build_wheelhouse() {
   mkdir -p "${WHEELHOUSE}"
   [ "${CLEAR_WHEELHOUSE}" = "1" ] && rm -f "${WHEELHOUSE}"/*.whl "${WHEELHOUSE}"/requirements*.lock

   local binary_lock="${BUILD_TMP_ROOT}/requirements.binary.lock"
   grep -Ev '^(antlr4-python3-runtime|pylatexenc|transformer-engine-torch)==|^transformer-engine-torch @' \
      "${RESOLVE_LOCK}" >"${binary_lock}"
   pip_wheel --only-binary=:all: --no-deps -r "${binary_lock}" \
      --index-url "${PYPI_INDEX_URL}" \
      --extra-index-url "${PYTORCH_INDEX_URL}" \
      --extra-index-url "${SGLANG_WHL_INDEX_URL}"
   reuse_wheel 'antlr4_python3_runtime-4.9.3-*.whl' || \
      pip_wheel --no-build-isolation --no-deps antlr4-python3-runtime==4.9.3
   reuse_wheel 'pylatexenc-2.10-*.whl' || \
      pip_wheel --no-build-isolation --no-deps pylatexenc==2.10
   reuse_wheel "transformer_engine_torch-${TRANSFORMER_ENGINE_VERSION}-*.whl" || \
      NVTE_WITH_NCCL_EP=0 NVTE_BUILD_MAX_JOBS="${TE_BUILD_MAX_JOBS}" MAX_JOBS="${TE_BUILD_MAX_JOBS}" \
      CMAKE_BUILD_PARALLEL_LEVEL="${TE_BUILD_MAX_JOBS}" MAKEFLAGS="-j${TE_BUILD_MAX_JOBS}" \
      pip_wheel --no-build-isolation --no-deps "transformer-engine-torch==${TRANSFORMER_ENGINE_VERSION}"

   reuse_wheel 'flash_attn-2.8.3-*.whl' || \
      FLASH_ATTENTION_FORCE_BUILD=TRUE \
      MAX_JOBS="${FLASH_ATTN_MAX_JOBS}" NVCC_THREADS="${FLASH_ATTN_NVCC_THREADS}" \
      pip_wheel -v flash-attn==2.8.3 --no-build-isolation --no-cache-dir --no-deps --no-binary flash-attn

   reuse_wheel 'flash_qla-*.whl' || pip_wheel --no-deps --no-build-isolation git+https://github.com/QwenLM/FlashQLA.git@6ef4858b5446e05bd461d9658d877e548182dbcb
   reuse_wheel 'apex-*.whl' || NVCC_APPEND_FLAGS="--threads ${APEX_NVCC_THREADS}" \
      pip_wheel -v --no-cache-dir --no-build-isolation --no-deps \
      --config-setting=--build-option="--cpp_ext --cuda_ext --parallel ${APEX_PARALLEL}" "${APEX_DIR}"

   TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$("${PYTHON}" -c 'import torch; print(torch.version.cuda.split(".")[0])')}"
   export TMS_CUDA_MAJOR
   reuse_wheel 'torch_memory_saver-*.whl' || \
      pip_wheel -v --no-cache-dir --no-build-isolation --no-deps \
      "git+https://github.com/zhuzilin/torch_memory_saver.git@${TMS_COMMIT}"

   pip_wheel --no-deps --no-build-isolation "${SLIME_DIR}/slime/backends/megatron_utils/kernels/int4_qat"

   if [ -n "${ROUTER_WHEEL_NAME:-}" ] && [ ! -s "${WHEELHOUSE}/${ROUTER_WHEEL_NAME}" ]; then
      die "missing expected router wheel ${ROUTER_WHEEL_NAME} in ${WHEELHOUSE}"
   fi
}

write_wheelhouse_lock() {
   "${PYTHON}" - "${WHEELHOUSE}" <<'PY'
from pathlib import Path
from zipfile import ZipFile
import email
import sys
from packaging.utils import canonicalize_name

wheelhouse = Path(sys.argv[1])
seen = {}
for wheel in sorted(wheelhouse.glob("*.whl")):
    with ZipFile(wheel) as zf:
        meta_name = next(name for name in zf.namelist() if name.endswith(".dist-info/METADATA"))
        meta = email.message_from_bytes(zf.read(meta_name))
    seen[canonicalize_name(meta["Name"])] = (meta["Name"], meta["Version"])
with (wheelhouse / "requirements.lock").open("w", encoding="utf-8") as f:
    for _, (name, version) in sorted(seen.items()):
        f.write(f"{name}=={version}\n")
PY
   cp "${RESOLVE_LOCK}" "${WHEELHOUSE}/requirements.resolve.lock"
}

install_from_wheelhouse() {
   uv pip install --python "${PYTHON}" --link-mode "${UV_LINK_MODE}" --exact --no-deps \
      --no-index --find-links "${WHEELHOUSE}" -r "${WHEELHOUSE}/requirements.lock"
}

write_megatron_lm_path() {
   "${PYTHON}" - "${MEGATRON_DIR}" <<'PY'
from pathlib import Path
import site
import sys

megatron_dir = Path(sys.argv[1]).resolve()
site_packages = Path(site.getsitepackages()[0])
(site_packages / "slime_omni_megatron_lm.pth").write_text(f"{megatron_dir}\n", encoding="utf-8")
PY
}

install_editables() {
   SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="${SGLANG_REF#v}" \
      uv_install -e "${SGLANG_DIR}/python" --no-build-isolation --no-deps
   uv_install -e "${MEGATRON_DIR}" --no-build-isolation --no-deps
   write_megatron_lm_path
   uv_install -e "${SLIME_DIR}" --no-deps
   uv_install -e "${REPO_ROOT}" --no-build-isolation --no-deps
   "${PYTHON}" -m slime_ext.install_hook
}

smoke_test() {
   [ "${SKIP_SMOKE}" = "1" ] && return
   "${PYTHON}" - <<'PY'
import importlib
for name in ["torch", "sglang", "flash_attn_2_cuda", "slime", "slime_ext", "blake3", "xxhash", "zstandard"]:
    importlib.import_module(name)
import sglang_router
assert "slime" in getattr(sglang_router, "__version__", ""), sglang_router.__version__
print("smoke ok")
PY
}

main() {
   command -v uv >/dev/null 2>&1 || die "uv is required"
   log "repo: ${REPO_ROOT}"
   log "build root: ${BUILD_TMP_ROOT}"
   log "venv: ${VENV_DIR}"
   prepare_sources
   link_deps_dir
   create_venv
   compile_lock
   validate_resolve_lock
   install_build_env
   build_wheelhouse
   write_wheelhouse_lock
   install_from_wheelhouse
   install_editables
   smoke_test
}

main "$@"
