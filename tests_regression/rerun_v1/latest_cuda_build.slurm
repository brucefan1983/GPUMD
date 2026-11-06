#!/bin/bash
#SBATCH --job-name=rerun-v1-latest-build
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --output=/data/home/user/PG/MD/GPUMD/950K/solid_left/Rerun_test/latest_6f548c78/build_cuda.slurm.out
#SBATCH --error=/data/home/user/PG/MD/GPUMD/950K/solid_left/Rerun_test/latest_6f548c78/build_cuda.slurm.err
set -euo pipefail

source /data/home/user/soft/GPU_MetaD_5.9/env.sh
source /data/home/user/miniconda3/etc/profile.d/conda.sh
conda activate gpu-metad

root=/data/home/user/soft/GPUMD-rerun-latest
expected=6f548c78baaad2f80b8d1bcbf8ccc88301515727
actual=$(git -C "$root" rev-parse HEAD)
printf 'Source branch: %s\nSource commit: %s\n' "$(git -C "$root" branch --show-current)" "$actual"
test "$actual" = "$expected"
nvcc --version
nvidia-smi
build="$root/build-rerun-v1-6f548c78-sm90"
cmake -S "$root" -B "$build" -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" --target gpumd nep -j 12
install_dir="$root/bin-rerun-v1-6f548c78-sm90"
mkdir -p "$install_dir"
install -m 755 "$build/gpumd" "$install_dir/gpumd"
install -m 755 "$build/nep" "$install_dir/nep"
"$install_dir/gpumd" --version || true
