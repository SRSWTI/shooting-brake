#!/usr/bin/env bash
# Route-frequency calibration, all-CUDA mode.
#
# setvars.sh calls exit internally, so it is sourced before set -e; it also
# strips the venv from PATH, so the venv is re-prepended after. Not strictly
# needed for an all-CUDA run, but keeping the environment identical to every
# other harness script costs nothing and avoids a second convention.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"

python phase7/calibrate_routes.py "$@"
