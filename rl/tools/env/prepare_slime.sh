#!/usr/bin/env bash
# Verify or apply the recorded Omni changes to the pinned upstream checkout.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/versions.env"
MODE=${1:-check}
SLIME_DIR=${2:-${SLIME_DIR:-${SCRIPT_DIR}/../../third_party/slime}}
PATCH=${SCRIPT_DIR}/patches/slime-omni.patch
case "${MODE}" in check|apply) ;; *) echo "usage: $0 [check|apply] [slime-dir]" >&2; exit 2 ;; esac
if [ "$(git -C "${SLIME_DIR}" rev-parse HEAD)" != "${SLIME_COMMIT}" ]; then
    echo "Slime must be checked out at ${SLIME_COMMIT}; existing files were not changed." >&2
    exit 1
fi
if git -C "${SLIME_DIR}" apply --reverse --check "${PATCH}" 2>/dev/null; then
    echo "Slime ${SLIME_COMMIT}: Omni patch present"
elif git -C "${SLIME_DIR}" apply --check "${PATCH}" 2>/dev/null; then
    if [ "${MODE}" = check ]; then
        echo "Omni patch is missing; run bash tools/env/prepare_slime.sh apply" >&2
        exit 1
    fi
    git -C "${SLIME_DIR}" apply "${PATCH}"
    echo "Slime ${SLIME_COMMIT}: Omni patch applied"
else
    echo "Slime checkout differs from the recorded Omni patch; inspect its local changes." >&2
    exit 1
fi
