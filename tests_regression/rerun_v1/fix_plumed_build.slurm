#!/bin/bash
#SBATCH --job-name=rerun-v1-fix-plumed
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

root=/path/to/PLUMED_BUILD_SOURCE
expected=e5f1b4f6dc8072a29aaa13946755d07a6fd9b383
actual=$(git -C "$root" rev-parse HEAD)
printf 'PLUMED source commit: %s\n' "$actual"
test "$actual" = "$expected"
prefix=/path/to/PLUMED_PREFIX
export PLUMED_KERNEL="$prefix/lib/libplumedKernel.so"
export LD_LIBRARY_PATH="$prefix/lib:${LD_LIBRARY_PATH:-}"
printf 'PLUMED installation: %s\n' "$prefix"
cd "$root/src"
make gpumd -j12 \
  CUDA_ARCH=-arch=sm_90 \
  CFLAGS="-std=c++14 -O3 -arch=sm_90 -DUSE_PLUMED" \
  INC="-I./ -I$prefix/include" \
  LDFLAGS="-L$prefix/lib -lplumed -lplumedKernel"
install_dir="$root/bin-rerun-v1-e5f1b4f-plumed"
mkdir -p "$install_dir"
install -m 755 gpumd "$install_dir/gpumd"
ldd "$install_dir/gpumd"
