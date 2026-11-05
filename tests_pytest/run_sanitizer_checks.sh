#!/usr/bin/env bash
# compute-sanitizer companion check, run manually outside pytest whenever kernel code changes.
#
# Nothing in the pytest suite (including the standalone scripts/compare_gpunep_cpunep.py
# developer utility) reliably catches kernel-level memory/race errors, since those can produce
# numerically plausible-but-wrong output on a given run. Pass/fail here means "sanitizer
# reported no errors," not a numerical match -- a different, complementary signal to the rest of
# this suite.
#
# Sanitizer runs are substantially slower than the numerical tests, so this is not wired into
# any automated trigger (matching CI wiring being out of scope for this suite in general).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GPUMD_EXECUTABLE="${REPO_ROOT}/gpumd"
FIXTURE_DIR="${SCRIPT_DIR}/fixtures/sanitizer_inputs"
TOOLS=(memcheck racecheck initcheck synccheck)

echo "=== Generating sanitizer input fixtures ==="
(cd "${SCRIPT_DIR}" && python3 -m pytest --dump-fixtures -q)

for case_dir in "${FIXTURE_DIR}"/*/; do
  case_name="$(basename "${case_dir}")"
  for tool in "${TOOLS[@]}"; do
    echo "=== compute-sanitizer --tool ${tool} (${case_name}) ==="
    (cd "${case_dir}" && compute-sanitizer --tool "${tool}" "${GPUMD_EXECUTABLE}")
  done
done
