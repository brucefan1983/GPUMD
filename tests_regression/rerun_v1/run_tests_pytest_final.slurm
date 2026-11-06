#!/bin/bash
#SBATCH --job-name=gpumd-pytest-e5
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --output=/data/home/user/PG/MD/GPUMD/950K/solid_left/Rerun_test/latest_e5f1b4f/pytest_final.slurm.out
#SBATCH --error=/data/home/user/PG/MD/GPUMD/950K/solid_left/Rerun_test/latest_e5f1b4f/pytest_final.slurm.err
set -euo pipefail
source /data/home/user/soft/GPU_MetaD_5.9/env.sh
source /data/home/user/miniconda3/etc/profile.d/conda.sh
conda activate gpu-metad
root=/data/home/user/soft/GPUMD-rerun-latest
build=/data/home/user/PG/MD/GPUMD/950K/solid_left/Rerun_test/latest_e5f1b4f/build_artifacts/bin-rerun-v1-e5f1b4f-sm90
expected=e5f1b4f6dc8072a29aaa13946755d07a6fd9b383
git -C "$root" merge-base --is-ancestor "$expected" HEAD
test -x "$build/gpumd" && test -x "$build/nep"
ln -sfn "$build/gpumd" "$root/src/gpumd"
ln -sfn "$build/nep" "$root/src/nep"
trap 'rm -f "$root/src/gpumd" "$root/src/nep"' EXIT
test "$(readlink -f "$root/src/gpumd")" = "$build/gpumd"
test "$(readlink -f "$root/src/nep")" = "$build/nep"
cd "$root/tests_pytest"
/data/home/user/soft/gpumd-pytest-e5/bin/python -m pytest -q --junitxml=/data/home/user/PG/MD/GPUMD/950K/solid_left/Rerun_test/latest_e5f1b4f/pytest_final.junit.xml 2>&1 | tee /data/home/user/PG/MD/GPUMD/950K/solid_left/Rerun_test/latest_e5f1b4f/pytest_final.log
