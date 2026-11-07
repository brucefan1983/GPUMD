#!/bin/bash
#SBATCH --job-name=rerun-v1-temperature
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
test_root=/path/to/RERUN_TEST_ROOT/temperature_boundary
mkdir -p "$test_root/no_ensemble" "$test_root/constant_ensemble"
model=$root/tests_regression/fixtures/systems/temperature_F_512.xyz
potential=$root/tests_regression/fixtures/potentials/temperature_F_nep.txt

for test_case in no_ensemble constant_ensemble; do
  cd "$test_root/$test_case"
  cp "$model" model.xyz
  python3 - "$model" <<'PY'
import sys

source = sys.argv[1]
with open(source) as stream:
    n = int(stream.readline())
    comment = stream.readline().rstrip("\n")
    atoms = [stream.readline() for _ in range(n)]
if len(atoms) != n or any(not line for line in atoms):
    raise SystemExit("temperature-dependent NEP fixture is incomplete")
with open("trajectory.xyz", "w") as out:
    for step in range(1, 101):
        out.write(f"{n}\nStep={step} Time={0.2 * step:.8f} {comment}\n")
        out.writelines(line.rstrip("\n") + "\n" for line in atoms)
PY
  {
    echo "potential $potential"
    echo "time_step 0.2"
    if [[ "$test_case" == constant_ensemble ]]; then
      echo "ensemble nvt_ber 950 950 100"
    fi
    echo "rerun trajectory.xyz"
  } > run.in
  set +e
  "$root/bin/gpumd" > gpumd.log 2>&1
  code=$?
  set -e
  echo "$test_case exit_code=$code" | tee result.txt
done
