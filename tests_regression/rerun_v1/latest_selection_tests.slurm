#!/bin/bash
#SBATCH --job-name=rerun-v1-latest-select
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=00:25:00
#SBATCH --output=/data/home/user/PG/MD/GPUMD/950K/solid_left/Rerun_test/latest_6f548c78/selection_final3.slurm.out
#SBATCH --error=/data/home/user/PG/MD/GPUMD/950K/solid_left/Rerun_test/latest_6f548c78/selection_final3.slurm.err
set -euo pipefail

source /data/home/user/soft/GPU_MetaD_5.9/env.sh
source /data/home/user/miniconda3/etc/profile.d/conda.sh
conda activate gpu-metad
python3 /data/home/user/soft/GPUMD-rerun-latest/tests_regression/rerun_v1/latest_selection_tests.py
