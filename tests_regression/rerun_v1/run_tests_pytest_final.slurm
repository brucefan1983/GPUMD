#!/bin/bash
#SBATCH --job-name=gpumd-pytest-e5
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
source /path/to/GPU_METAD_ENV
source /path/to/CONDA_ROOT/etc/profile.d/conda.sh
conda activate gpu-metad
root=/path/to/GPUMD_SOURCE
build=/path/to/RERUN_TEST_ROOT/latest_e5f1b4f/build_artifacts/bin-rerun-v1-e5f1b4f-sm90
expected=e5f1b4f6dc8072a29aaa13946755d07a6fd9b383
git -C "$root" merge-base --is-ancestor "$expected" HEAD
test -x "$build/gpumd" && test -x "$build/nep"
ln -sfn "$build/gpumd" "$root/src/gpumd"
ln -sfn "$build/nep" "$root/src/nep"
trap 'rm -f "$root/src/gpumd" "$root/src/nep"' EXIT
test "$(readlink -f "$root/src/gpumd")" = "$build/gpumd"
test "$(readlink -f "$root/src/nep")" = "$build/nep"
cd "$root/tests_pytest"
/path/to/PYTEST_ENV/bin/python -m pytest -q --junitxml=/path/to/RERUN_TEST_ROOT/latest_e5f1b4f/pytest_final.junit.xml 2>&1 | tee /path/to/RERUN_TEST_ROOT/latest_e5f1b4f/pytest_final.log
