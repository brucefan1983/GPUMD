#!/usr/bin/env bash
# compute-sanitizer companion check, run manually outside pytest whenever kernel code changes.
#
# Kernel-level memory/race checks complement the numerical tests. A sanitizer
# error fails this script even when the executable exits successfully.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GPUMD_EXECUTABLE="${REPO_ROOT}/src/gpumd"
if [[ ! -x "${GPUMD_EXECUTABLE}" ]]; then
  GPUMD_EXECUTABLE="${REPO_ROOT}/gpumd"
fi
FIXTURE_DIR="${SCRIPT_DIR}/fixtures/sanitizer_inputs"
TOOLS=(memcheck racecheck initcheck synccheck)

echo "=== Generating sanitizer input fixtures ==="
(cd "${SCRIPT_DIR}" && python3 -m pytest --dump-fixtures -q)

for case_dir in "${FIXTURE_DIR}"/*/; do
  case_name="$(basename "${case_dir}")"
  for tool in "${TOOLS[@]}"; do
    echo "=== compute-sanitizer --tool ${tool} (${case_name}) ==="
    (cd "${case_dir}" && compute-sanitizer --error-exitcode 1 --tool "${tool}" "${GPUMD_EXECUTABLE}")
  done
done
