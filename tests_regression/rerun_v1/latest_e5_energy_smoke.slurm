#!/bin/bash
#SBATCH --job-name=rerun-v1-energy
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
gpumd=/path/to/RERUN_TEST_ROOT/latest_e5f1b4f/build_artifacts/bin-rerun-v1-e5f1b4f-sm90/gpumd
test_root=/path/to/RERUN_TEST_ROOT/latest_e5f1b4f/energy_smoke
model=$root/tests_regression/fixtures/systems/cu_grouped.xyz
potential=$root/potentials/eam/Cu_Zhou_2004.txt
mkdir -p "$test_root/reference" "$test_root/rerun_a" "$test_root/rerun_b"

cd "$test_root/reference"
cp "$model" model.xyz
cat > run.in <<EOF
potential $potential
time_step 0.2
ensemble nve
dump_xyz 1 reference.xyz precision double velocity potential force virial
dump_thermo 1
run 1
EOF
"$gpumd" > gpumd.log 2>&1

python3 - "$test_root/reference/reference.xyz" "$test_root/rerun_a/trajectory.xyz" <<'PY'
import re
import sys

source, target = sys.argv[1:]
with open(source) as stream:
    n = int(stream.readline())
    comment = stream.readline().rstrip("\n")
    atoms = [stream.readline() for _ in range(n)]
if len(atoms) != n or any(not line.strip() for line in atoms):
    raise SystemExit("normal GPUMD reference dump is incomplete")
comment = re.sub(r"(?<!\S)(?:Step|Time)=[^ \t]+", "", comment).strip()
energy = float(re.search(r"(?:^| )energy=([^ ]+)", comment).group(1))
with open(target, "w") as out:
    for step in range(1, 101):
        frame_comment = f"Step={step} Time={0.2 * step:.8f} " + comment
        out.write(f"{n}\n{frame_comment}\n")
        out.writelines(line.rstrip("\n") + "\n" for line in atoms)
open(target + ".reference_energy", "w").write(f"{energy:.17g}\n")
PY

for run_name in rerun_a rerun_b; do
  cd "$test_root/$run_name"
  cp "$model" model.xyz
  if [[ "$run_name" == rerun_b ]]; then
    cp "$test_root/rerun_a/trajectory.xyz" trajectory.xyz
  fi
  cat > run.in <<EOF
potential $potential
time_step 0.2
dump_xyz 1 rerun.xyz precision double velocity force potential virial
dump_thermo 1
rerun trajectory.xyz
EOF
  "$gpumd" > gpumd.log 2>&1
done

python3 "$root/tests_regression/rerun_v1/compare_rerun_xyz.py" \
  "$test_root/reference/reference.xyz" "$test_root/rerun_a/rerun.xyz" 100 0.2
python3 "$root/tests_regression/rerun_v1/compare_rerun_xyz.py" \
  "$test_root/reference/reference.xyz" "$test_root/rerun_b/rerun.xyz" 100 0.2
python3 - "$test_root" <<'PY'
import os
import sys
root = sys.argv[1]
for name in ("rerun_a", "rerun_b"):
    with open(os.path.join(root, name, "thermo.out")) as stream:
        rows = []
        for line in stream:
            fields = line.split()
            try:
                float(fields[0])
                rows.append(fields)
            except (ValueError, IndexError):
                pass
    if len(rows) != 100:
        raise SystemExit(f"{name}: expected 100 dump_thermo rows, got {len(rows)}")
print("PASS dump_thermo: 100 rows per rerun; repeated execution logs and outputs retained")
PY

# Ordinary, temperature-independent NEP should also rerun without an ensemble.
nep_root=$test_root/nep_smoke
mkdir -p "$nep_root/reference" "$nep_root/rerun"
nep_model=$root/tests_regression/fixtures/systems/observer_carbon_159.xyz
nep_potential=$root/tests_regression/fixtures/potentials/C_2022_NEP4_MODIFIED.txt
cd "$nep_root/reference"
cp "$nep_model" model.xyz
cat > run.in <<EOF
potential $nep_potential
time_step 0.2
ensemble nve
dump_xyz 1 reference.xyz precision double velocity potential force virial
run 1
EOF
"$gpumd" > gpumd.log 2>&1

python3 - "$nep_root/reference/reference.xyz" "$nep_root/rerun/trajectory.xyz" <<'PY'
import re
import sys

source, target = sys.argv[1:]
with open(source) as stream:
    n = int(stream.readline())
    comment = stream.readline().rstrip("\n")
    atoms = [stream.readline() for _ in range(n)]
if len(atoms) != n or any(not line for line in atoms):
    raise SystemExit("normal NEP reference dump is incomplete")
comment = re.sub(r"(?<!\S)(?:Step|Time)=[^ \t]+", "", comment).strip()
energy = float(re.search(r"(?:^| )energy=([^ ]+)", comment).group(1))
with open(target, "w") as out:
    for step in range(1, 101):
        out.write(f"{n}\nStep={step} Time={0.2 * step:.8f} {comment}\n")
        out.writelines(line.rstrip("\n") + "\n" for line in atoms)
open(target + ".reference_energy", "w").write(f"{energy:.17g}\n")
PY

cd "$nep_root/rerun"
cp "$nep_model" model.xyz
cat > run.in <<EOF
potential $nep_potential
time_step 0.2
dump_xyz 1 rerun.xyz precision double velocity force potential virial
dump_thermo 1
rerun trajectory.xyz
EOF
"$gpumd" > gpumd.log 2>&1
python3 "$root/tests_regression/rerun_v1/compare_rerun_xyz.py" \
  "$nep_root/reference/reference.xyz" "$nep_root/rerun/rerun.xyz" 100 0.2
