#!/bin/bash
#SBATCH --job-name=rerun-v1-latest-plumed-test
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=00:25:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail

source /path/to/GPU_METAD_ENV
source /path/to/CONDA_ROOT/etc/profile.d/conda.sh
conda activate gpu-metad

root=/path/to/GPUMD_SOURCE
plumed_root=/path/to/PLUMED_BUILD_SOURCE
plumed_prefix=/path/to/PLUMED_PREFIX
gpumd=/path/to/RERUN_TEST_ROOT/latest_e5f1b4f/build_artifacts/bin-rerun-v1-e5f1b4f-plumed/gpumd
export PLUMED_KERNEL="$plumed_prefix/lib/libplumedKernel.so"
export LD_LIBRARY_PATH="$plumed_prefix/lib:${LD_LIBRARY_PATH:-}"
test_root=/path/to/RERUN_TEST_ROOT/latest_e5f1b4f/plumed_final
source_trajectory=/path/to/RERUN_TEST_ROOT/latest_e5f1b4f/energy_smoke/rerun_a/trajectory.xyz
model=$root/tests_regression/fixtures/systems/cu_grouped.xyz
potential=$root/potentials/eam/Cu_Zhou_2004.txt
mkdir -p "$test_root/interval1" "$test_root/interval10" "$test_root/sparse"

python3 - "$source_trajectory" "$test_root" <<'PY'
import sys

source, root = sys.argv[1:]
with open(source) as stream:
    frames = []
    while True:
        count_line = stream.readline()
        if not count_line:
            break
        if not count_line.strip():
            continue
        n = int(count_line)
        comment = stream.readline().strip()
        atoms = [stream.readline() for _ in range(n)]
        frames.append((n, comment, atoms))
if len(frames) != 100:
    raise SystemExit(f"expected 100 source frames, found {len(frames)}")
for name, selected in (("interval1", frames), ("interval10", frames), ("sparse", [frames[19], frames[99]])):
    with open(f"{root}/{name}/trajectory.xyz", "w") as out:
        for n, comment, atoms in selected:
            out.write(f"{n}\n{comment}\n")
            out.writelines(line.rstrip("\n") + "\n" for line in atoms)
PY

for case in interval1 interval10 sparse; do
  cd "$test_root/$case"
  cp "$model" model.xyz
  cat > plumed.dat <<'EOF'
d: DISTANCE ATOMS=1,2
PRINT ARG=d STRIDE=1 FILE=COLVAR
EOF
  case_interval=1
  if [[ "$case" == interval10 || "$case" == sparse ]]; then
    case_interval=10
  fi
  cat > run.in <<EOF
potential $potential
time_step 0.2
ensemble nvt_ber 950 950 100
plumed plumed.dat $case_interval 0
dump_xyz 1 rerun.xyz precision double velocity force potential virial
dump_thermo 1
rerun trajectory.xyz
EOF
  "$gpumd" > gpumd.log 2>&1
done

python3 - "$test_root" <<'PY'
import math
import os
import re
import sys

root = sys.argv[1]
source = os.path.join(root, "interval1/trajectory.xyz")
with open(source) as stream:
    n = int(stream.readline())
    comment = stream.readline()
    rows = [stream.readline().split() for _ in range(n)]
properties = re.search(r"(?:^|\s)Properties=([^ \t]+)", comment).group(1).split(":")
offset, cursor = {}, 0
for i in range(0, len(properties), 3):
    name, width = properties[i], int(properties[i + 2])
    offset[name] = (cursor, width)
    cursor += width
start, _ = offset["pos"]
points = [[float(value) for value in row[start:start + 3]] for row in rows[:2]]
box_values = [float(value) for value in re.search(r'Lattice="([^\"]+)"', comment).group(1).split()]
lengths = (box_values[0], box_values[4], box_values[8])
delta = [points[1][i] - points[0][i] for i in range(3)]
delta = [delta[i] - round(delta[i] / lengths[i]) * lengths[i] for i in range(3)]
expected_cv = math.sqrt(sum(value * value for value in delta)) * 0.1

for case, expected_steps in (("interval1", list(range(1, 101))), ("interval10", list(range(10, 101, 10))), ("sparse", [20, 100])):
    path = os.path.join(root, case, "COLVAR")
    cv_rows = []
    for line in open(path):
        if line.strip() and not line.startswith("#"):
            cv_rows.append([float(value) for value in line.split()])
    if len(cv_rows) != len(expected_steps):
        raise SystemExit(f"{case}: CV row count={len(cv_rows)} expected={len(expected_steps)}")
    max_cv_error = max(abs(row[-1] - expected_cv) for row in cv_rows)
    if max_cv_error > 1e-6:
        raise SystemExit(f"{case}: independent distance CV error {max_cv_error} nm")
    expected_times_ps = [step * 0.0002 for step in expected_steps]
    max_time_error = max(abs(row[0] - expected) for row, expected in zip(cv_rows, expected_times_ps))
    if max_time_error > 1e-7:
        raise SystemExit(f"{case}: COLVAR physical-time error {max_time_error} ps")
    expected_outputs = 2 if case == "sparse" else 100
    xyz_count = sum(1 for line in open(os.path.join(root, case, "rerun.xyz")) if line.strip().isdigit())
    thermo_rows = 0
    for line in open(os.path.join(root, case, "thermo.out")):
        fields = line.split()
        try:
            float(fields[0]); thermo_rows += 1
        except (ValueError, IndexError):
            pass
    if xyz_count != expected_outputs or thermo_rows != expected_outputs:
        raise SystemExit(f"{case}: dump_xyz={xyz_count}, dump_thermo={thermo_rows}, expected={expected_outputs}")
    print(f"PASS {case}: CV_rows={len(cv_rows)} CV_max_error={max_cv_error:.8g} nm COLVAR_time_max_error={max_time_error:.8g} ps xyz_frames={xyz_count} thermo_rows={thermo_rows}")
PY
