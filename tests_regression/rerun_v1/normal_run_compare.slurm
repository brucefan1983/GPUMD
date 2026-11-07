#!/bin/bash
#SBATCH --job-name=rerun-v1-normal-run
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
source /path/to/GPU_METAD_ENV
source /path/to/CONDA_ROOT/etc/profile.d/conda.sh
conda activate gpu-metad
root=/path/to/GPUMD_SOURCE
base=/path/to/RERUN_TEST_ROOT/latest_e5f1b4f/normal_run_compare
model=$root/tests_regression/fixtures/systems/cu_grouped.xyz
potential=$root/potentials/eam/Cu_Zhou_2004.txt
mkdir -p "$base/fix" "$base/patched"
rm -f "$base/fix/output.xyz" "$base/fix/thermo.out" "$base/fix/gpumd.log"
rm -f "$base/patched/output.xyz" "$base/patched/thermo.out" "$base/patched/gpumd.log"
for flavor in fix patched; do
  cd "$base/$flavor"
  cp "$model" model.xyz
  cat > run.in <<EOF
potential $potential
velocity 300 seed 123
time_step 0.2
ensemble nve
dump_xyz 1 output.xyz precision double velocity force potential virial
dump_thermo 1
run 2
EOF
done
(cd "$base/fix" && "$base/../build_artifacts/bin-rerun-v1-6f548c78-sm90/gpumd" > gpumd.log 2>&1)
(cd "$base/patched" && "$base/../build_artifacts/bin-rerun-v1-e5f1b4f-sm90/gpumd" > gpumd.log 2>&1)
cmp "$base/fix/output.xyz" "$base/patched/output.xyz"
cmp "$base/fix/thermo.out" "$base/patched/thermo.out"
echo "PASS normal run output is byte-identical for 2-step NVE reference (xyz and thermo)"
