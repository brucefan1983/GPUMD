#!/bin/bash
#SBATCH --job-name=rerun-v1-e5-select
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=00:25:00
#SBATCH --output=/data/home/user/PG/MD/GPUMD/950K/solid_left/Rerun_test/latest_e5f1b4f/selection.slurm.out
#SBATCH --error=/data/home/user/PG/MD/GPUMD/950K/solid_left/Rerun_test/latest_e5f1b4f/selection.slurm.err
set -euo pipefail
source /data/home/user/soft/GPU_MetaD_5.9/env.sh
source /data/home/user/miniconda3/etc/profile.d/conda.sh
conda activate gpu-metad
export RERUN_GPUMD=/data/home/user/PG/MD/GPUMD/950K/solid_left/Rerun_test/latest_e5f1b4f/build_artifacts/bin-rerun-v1-e5f1b4f-sm90/gpumd
export RERUN_TEST_ROOT=/data/home/user/PG/MD/GPUMD/950K/solid_left/Rerun_test/latest_e5f1b4f/selection
export RERUN_TRAJECTORY=/data/home/user/PG/MD/GPUMD/950K/solid_left/Rerun_test/latest_e5f1b4f/energy_smoke/rerun_a/trajectory.xyz
python3 /data/home/user/soft/GPUMD-rerun-latest/tests_regression/rerun_v1/latest_selection_tests.py
