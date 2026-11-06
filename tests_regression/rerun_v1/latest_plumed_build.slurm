#!/bin/bash
#SBATCH --job-name=rerun-v1-latest-plumed
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --output=/data/home/user/PG/MD/GPUMD/950K/solid_left/Rerun_test/latest_6f548c78/build_plumed.slurm.out
#SBATCH --error=/data/home/user/PG/MD/GPUMD/950K/solid_left/Rerun_test/latest_6f548c78/build_plumed.slurm.err
set -euo pipefail

source /data/home/user/soft/GPU_MetaD_5.9/env.sh
source /data/home/user/miniconda3/etc/profile.d/conda.sh
conda activate gpu-metad

root=/data/home/user/soft/GPUMD-rerun-latest-plumed-src
expected=6f548c78baaad2f80b8d1bcbf8ccc88301515727
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
install_dir="$root/bin-rerun-v1-6f548c78-plumed"
mkdir -p "$install_dir"
install -m 755 gpumd "$install_dir/gpumd"
ldd "$install_dir/gpumd"
