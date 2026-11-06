#!/bin/bash
#SBATCH --job-name=rerun-v1-fix-plumed
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --output=/data/home/user/PG/MD/GPUMD/950K/solid_left/Rerun_test/latest_e5f1b4f/build_plumed_fix.slurm.out
#SBATCH --error=/data/home/user/PG/MD/GPUMD/950K/solid_left/Rerun_test/latest_e5f1b4f/build_plumed_fix.slurm.err
set -euo pipefail

source /data/home/user/soft/GPU_MetaD_5.9/env.sh
source /data/home/user/miniconda3/etc/profile.d/conda.sh
conda activate gpu-metad

root=/data/home/user/soft/GPUMD-rerun-latest-plumed-e5f1b4f-src
expected=e5f1b4f6dc8072a29aaa13946755d07a6fd9b383
actual=$(git -C "$root" rev-parse HEAD)
printf 'PLUMED source commit: %s\n' "$actual"
test "$actual" = "$expected"
prefix=/data/home/user/soft/Lammps+deepmd+plumed_newversion/installs/plumed-2.10.1-official
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
