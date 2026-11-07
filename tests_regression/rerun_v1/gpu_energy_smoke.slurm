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
test_root=/path/to/RERUN_TEST_ROOT/energy_smoke
model=$root/tests_regression/fixtures/systems/cu_grouped.xyz
potential=$root/potentials/eam/Cu_Zhou_2004.txt
mkdir -p "$test_root/reference" "$test_root/rerun_a" "$test_root/rerun_b"

cd "$test_root/reference"
cp "$model" model.xyz
cat > run.in <<EOF
potential $potential
time_step 0.2
ensemble nve
dump_xyz 1 reference.xyz precision double potential force virial
dump_thermo 1
run 1
EOF
"$root/bin/gpumd" > gpumd.log 2>&1

python3 - "$test_root/reference/reference.xyz" "$test_root/rerun_a/trajectory.xyz" <<'PY'
import re
import sys

source, target = sys.argv[1:]
with open(source) as stream:
    n = int(stream.readline())
    comment = stream.readline().rstrip("\n")
    atoms = [stream.readline() for _ in range(n)]
if len(atoms) != n or any(not line for line in atoms):
    raise SystemExit("normal GPUMD reference dump is incomplete")
energy = float(re.search(r"(?:^| )energy=([^ ]+)", comment).group(1))
with open(target, "w") as out:
    for step in range(1, 101):
        frame_comment = f"Step={step} Time={0.2 * step:.8f} " + comment
        out.write(f"{n}\n{frame_comment}\n")
        out.writelines(atoms)
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
rerun trajectory.xyz
EOF
  "$root/bin/gpumd" > gpumd.log 2>&1
done

python3 - "$test_root" <<'PY'
import math
import os
import sys

root = sys.argv[1]
reference = float(open(os.path.join(root, "rerun_a/trajectory.xyz.reference_energy")).read())
values = []
for name in ("rerun_a", "rerun_b"):
    with open(os.path.join(root, name, "rerun_energy.dat")) as stream:
        rows = [line.split() for line in stream if line.strip() and not line.startswith("#")]
    if len(rows) != 100:
        raise SystemExit(f"{name}: expected 100 energy rows, got {len(rows)}")
    values.append(float(rows[0][2]))
    if any(not math.isclose(float(row[2]), reference, rel_tol=1e-10, abs_tol=1e-8) for row in rows):
        raise SystemExit(f"{name}: rerun energy differs from the normal GPUMD reference")
if values[0] != values[1]:
    raise SystemExit("two rerun executions are not reproducible")
print(f"PASS: frames=100 reference_energy={reference:.17g} rerun_energy={values[0]:.17g} delta={abs(values[0]-reference):.6g} eV reproducibility=exact")
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
dump_xyz 1 reference.xyz precision double potential force virial
run 1
EOF
"$root/bin/gpumd" > gpumd.log 2>&1

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
energy = float(re.search(r"(?:^| )energy=([^ ]+)", comment).group(1))
with open(target, "w") as out:
    for step in range(1, 101):
        out.write(f"{n}\nStep={step} Time={0.2 * step:.8f} {comment}\n")
        out.writelines(atoms)
open(target + ".reference_energy", "w").write(f"{energy:.17g}\n")
PY

cd "$nep_root/rerun"
cp "$nep_model" model.xyz
cat > run.in <<EOF
potential $nep_potential
time_step 0.2
rerun trajectory.xyz
EOF
"$root/bin/gpumd" > gpumd.log 2>&1
python3 - "$nep_root/rerun" <<'PY'
import math
import os
import sys

root = sys.argv[1]
reference = float(open(os.path.join(root, "trajectory.xyz.reference_energy")).read())
with open(os.path.join(root, "rerun_energy.dat")) as stream:
    rows = [line.split() for line in stream if line.strip() and not line.startswith("#")]
if len(rows) != 100:
    raise SystemExit(f"ordinary NEP: expected 100 rows, got {len(rows)}")
if any(not math.isclose(float(row[2]), reference, rel_tol=1e-10, abs_tol=1e-8) for row in rows):
    raise SystemExit("ordinary NEP rerun energy differs from the normal GPUMD reference")
print(f"PASS: ordinary NEP without rerun ensemble; frames=100 reference_energy={reference:.17g} rerun_energy={float(rows[0][2]):.17g} delta={abs(float(rows[0][2])-reference):.6g} eV")
PY
