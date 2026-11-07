#!/usr/bin/env python3
"""Verify the rerun input/output alias fix in the committed source build."""

import hashlib
import shutil
import subprocess
from pathlib import Path

root = Path("/path/to/GPUMD_SOURCE")
gpumd = Path("/path/to/RERUN_TEST_ROOT/latest_e5f1b4f/build_artifacts/bin-rerun-v1-e5f1b4f-sm90/gpumd")
test_dir = Path(
    "/path/to/RERUN_TEST_ROOT/latest_e5f1b4f/collision_alias"
)
source = Path(
    "/path/to/RERUN_TEST_ROOT/latest_6f548c78/energy_smoke/rerun_a/trajectory.xyz"
)
model = root / "tests_regression/fixtures/systems/cu_grouped.xyz"
potential = root / "potentials/eam/Cu_Zhou_2004.txt"
test_dir.mkdir(parents=True, exist_ok=True)
shutil.copyfile(model, test_dir / "model.xyz")
lines = source.read_text().splitlines(keepends=True)
# Preserve a small two-frame trajectory while retaining the full frame records.
n = int(lines[0])
frame_size = n + 2
(test_dir / "trajectory.xyz").write_text("".join(lines[: 2 * frame_size]))
trajectory = test_dir / "trajectory.xyz"
before = hashlib.sha256(trajectory.read_bytes()).hexdigest()
(test_dir / "run.in").write_text(
    f"potential {potential}\ntime_step 0.2\n"
    "dump_xyz 1 ./trajectory.xyz force potential virial\n"
    "rerun trajectory.xyz\n"
)
with open(test_dir / "gpumd.log", "w") as log:
    result = subprocess.run([str(gpumd)], cwd=test_dir, stdout=log, stderr=subprocess.STDOUT)
after = hashlib.sha256(trajectory.read_bytes()).hexdigest()
message = (test_dir / "gpumd.log").read_text()
if result.returncode == 0:
    raise SystemExit("FAIL: relative-path alias was accepted")
if before != after:
    raise SystemExit("FAIL: input trajectory changed after collision rejection")
if "must not overwrite the input trajectory" not in message:
    raise SystemExit("FAIL: collision did not produce the expected diagnostic")
print("PASS: relative-path alias rejected; input SHA-256 unchanged")
