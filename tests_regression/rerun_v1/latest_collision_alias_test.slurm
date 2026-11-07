#!/bin/bash
#SBATCH --job-name=rerun-v1-alias-fix
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
source /path/to/GPU_METAD_ENV
source /path/to/CONDA_ROOT/etc/profile.d/conda.sh
conda activate gpu-metad
python3 /path/to/GPUMD_SOURCE/tests_regression/rerun_v1/latest_collision_alias_test.py
