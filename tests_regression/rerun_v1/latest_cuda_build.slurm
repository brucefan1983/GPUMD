#!/bin/bash
#SBATCH --job-name=rerun-v1-latest-build
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail

source /path/to/GPU_METAD_ENV
source /path/to/CONDA_ROOT/etc/profile.d/conda.sh
conda activate gpu-metad

root=/path/to/GPUMD_SOURCE
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
