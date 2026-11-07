#!/bin/bash
#SBATCH --job-name=rerun-v1-latest-temp
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail

source /path/to/GPU_METAD_ENV
source /path/to/CONDA_ROOT/etc/profile.d/conda.sh
conda activate gpu-metad

root=/path/to/GPUMD_SOURCE
gpumd=$root/bin-rerun-v1-6f548c78-sm90/gpumd
test_root=/path/to/RERUN_TEST_ROOT/latest_6f548c78/temperature_final
model=$root/tests_regression/fixtures/systems/temperature_F_512.xyz
potential=$root/tests_regression/fixtures/potentials/temperature_F_nep.txt
mkdir -p "$test_root/reference" "$test_root/constant" "$test_root/missing" "$test_root/ramp"

cd "$test_root/reference"
cp "$model" model.xyz
cat > run.in <<EOF
potential $potential
velocity 300 seed 123
time_step 0.2
ensemble nvt_ber 950 950 100
dump_xyz 1 reference.xyz precision double velocity potential force virial
run 1
EOF
"$gpumd" > gpumd.log 2>&1

python3 - "$test_root/reference/reference.xyz" "$test_root/constant/trajectory.xyz" <<'PY'
import re
import sys

source, target = sys.argv[1:]
with open(source) as stream:
    n = int(stream.readline())
    comment = stream.readline().strip()
    atoms = [stream.readline() for _ in range(n)]
if len(atoms) != n or any(not line.strip() for line in atoms):
    raise SystemExit("constant-temperature reference dump is incomplete")
comment = re.sub(r"(?<!\S)(?:Step|Time)=[^ \t]+", "", comment).strip()
with open(target, "w") as out:
    for step in range(1, 101):
        out.write(f"{n}\nStep={step} Time={0.2 * step:.8f} {comment}\n")
        out.writelines(line.rstrip("\n") + "\n" for line in atoms)
PY

for case in constant missing ramp; do
  cd "$test_root/$case"
  cp "$model" model.xyz
  if [[ "$case" != constant ]]; then
    cp "$test_root/constant/trajectory.xyz" trajectory.xyz
  fi
  {
    echo "potential $potential"
    echo "time_step 0.2"
    if [[ "$case" == constant ]]; then
      echo "ensemble nvt_ber 950 950 100"
      echo "dump_xyz 1 rerun.xyz precision double velocity force potential virial"
      echo "dump_thermo 1"
    elif [[ "$case" == ramp ]]; then
      echo "ensemble nvt_ber 300 950 100"
    fi
    echo "rerun trajectory.xyz"
  } > run.in
done

cd "$test_root/constant"
"$gpumd" > gpumd.log 2>&1
python3 "$root/tests_regression/rerun_v1/compare_rerun_xyz.py" \
  "$test_root/reference/reference.xyz" "$test_root/constant/rerun.xyz" 100 0.2
python3 - "$test_root/constant/thermo.out" <<'PY'
import sys
rows = []
for line in open(sys.argv[1]):
    fields = line.split()
    try:
        float(fields[0]); rows.append(fields)
    except (ValueError, IndexError):
        pass
if len(rows) != 100:
    raise SystemExit(f"constant T dump_thermo rows={len(rows)}, expected 100")
print("PASS constant-temperature TNEP: dump_thermo rows=100")
PY

for case in missing ramp; do
  cd "$test_root/$case"
  set +e
  "$gpumd" > gpumd.log 2>&1
  code=$?
  set -e
  if [[ "$code" -eq 0 ]]; then
    echo "FAIL $case: unsupported temperature configuration was accepted"
    exit 1
  fi
  if [[ "$case" == missing ]]; then
    grep -q "requires a positive temperature from" gpumd.log
  else
    grep -q "requires a constant-temperature ensemble" gpumd.log
  fi
  echo "PASS $case: rejected clearly (exit=$code)"
done
