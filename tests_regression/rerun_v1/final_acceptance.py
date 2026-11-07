#!/usr/bin/env python3
"""Negative, path-alias, and rerun lifecycle tests for patched GPUMD rerun."""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

root = Path("/path/to/GPUMD_SOURCE")
gpumd = Path("/path/to/RERUN_TEST_ROOT/latest_e5f1b4f/build_artifacts/bin-rerun-v1-e5f1b4f-sm90/gpumd")
potential = root / "potentials/eam/Cu_Zhou_2004.txt"
model = root / "tests_regression/fixtures/systems/cu_grouped.xyz"
base = Path("/path/to/RERUN_TEST_ROOT/latest_e5f1b4f")
energy = base / "energy_smoke/rerun_a/trajectory.xyz"
work = base / "final_acceptance"


def frames(path):
    lines = path.read_text().splitlines(keepends=True)
    n = int(lines[0])
    size = n + 2
    return ["".join(lines[i:i + size]) for i in range(0, len(lines), size)]


def run(case, trajectory_text, extra="", output="out.xyz", expects_success=False):
    d = work / case
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    shutil.copyfile(model, d / "model.xyz")
    (d / "trajectory.xyz").write_text(trajectory_text)
    (d / "run.in").write_text(
        f"potential {potential}\ntime_step 0.2\n{extra}"
        f"dump_xyz 1 {output} force potential virial\nrerun trajectory.xyz\n"
    )
    with open(d / "gpumd.log", "w") as log:
        result = subprocess.run([str(gpumd)], cwd=d, stdout=log,
                                stderr=subprocess.STDOUT, timeout=30)
    ok = result.returncode == 0
    if ok != expects_success:
        raise AssertionError(f"{case}: exit={result.returncode}, expected success={expects_success}; see {d}/gpumd.log")
    print(f"PASS {case}: exit={result.returncode}")
    return d


work.mkdir(parents=True, exist_ok=True)
fs = frames(energy)
two = "".join(fs[:2])
run("malformed_xyz", "32\nStep=1 Time=0\nCu 0 0\n")
run("invalid_step_time", two.replace("Step=1 Time=0.2", "Step=1 Time=NaN", 1))
run("atom_count_mismatch", two.replace("32\n", "31\n", 1))
run("invalid_box", two.replace("Lattice=\"", "Lattice=\"0 0 0 ", 1))

# Thermodynamics needs per-atom velocities. Remove velocity fields and check
# that requesting dump_thermo does not silently produce a plausible result.
no_velocity_lines = []
for line in two.splitlines():
    if "Properties=" in line:
        line = line.replace(
            "Properties=species:S:1:pos:R:3:vel:R:3:forces:R:3:energy_atom:R:1:virial:R:9",
            "Properties=species:S:1:pos:R:3",
        )
    elif line.startswith("Cu "):
        line = " ".join(line.split()[:4])
    no_velocity_lines.append(line)
no_velocity = "\n".join(no_velocity_lines) + "\n"
d = work / "missing_velocity"
if d.exists():
    shutil.rmtree(d)
d.mkdir(parents=True)
shutil.copyfile(model, d / "model.xyz")
(d / "trajectory.xyz").write_text(no_velocity)
(d / "run.in").write_text(
    f"potential {potential}\ntime_step 0.2\ndump_thermo 1\nrerun trajectory.xyz\n"
)
with open(d / "gpumd.log", "w") as log:
    result = subprocess.run([str(gpumd)], cwd=d, stdout=log,
                            stderr=subprocess.STDOUT, timeout=30)
if result.returncode == 0:
    print("FAIL missing_velocity: accepted; inspect thermo semantics in log")
else:
    print(f"PASS missing_velocity: rejected (exit={result.returncode})")

# Every alias resolves to the same existing trajectory. Verify both rejection
# and byte-for-byte preservation after each attempt.
aliases = {
    "alias_dot": "./trajectory.xyz",
    "alias_parent": "sub/../trajectory.xyz",
    "alias_absolute": str(work / "alias_absolute/trajectory.xyz"),
    "alias_symlink": "trajectory-link.xyz",
}
for case, name in aliases.items():
    d = work / case
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    shutil.copyfile(model, d / "model.xyz")
    (d / "trajectory.xyz").write_text(two)
    (d / "sub").mkdir()
    if case == "alias_symlink":
        (d / name).symlink_to(d / "trajectory.xyz")
    path = d / "trajectory.xyz"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    (d / "run.in").write_text(
        f"potential {potential}\ntime_step 0.2\n"
        f"dump_xyz 1 {name} force potential virial\nrerun trajectory.xyz\n"
    )
    with open(d / "gpumd.log", "w") as log:
        result = subprocess.run([str(gpumd)], cwd=d, stdout=log,
                                stderr=subprocess.STDOUT, timeout=30)
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    if result.returncode == 0 or before != after:
        raise AssertionError(f"{case}: alias not safely rejected or input modified")
    print(f"PASS {case}: rejected, input SHA-256 unchanged")

# Exercise repeated initialization/finalization in one process using 10 reruns.
d = work / "ten_sequential_reruns"
if d.exists():
    shutil.rmtree(d)
d.mkdir(parents=True)
shutil.copyfile(model, d / "model.xyz")
runin = [f"potential {potential}", "time_step 0.2"]
for i in range(10):
    (d / f"traj{i}.xyz").write_text(fs[i:i + 1][0])
    runin.extend([f"dump_xyz 1 out{i}.xyz force potential virial", f"rerun traj{i}.xyz"])
(d / "run.in").write_text("\n".join(runin) + "\n")
with open(d / "gpumd.log", "w") as log:
    result = subprocess.run([str(gpumd)], cwd=d, stdout=log,
                            stderr=subprocess.STDOUT, timeout=60)
if result.returncode:
    raise AssertionError(f"ten sequential reruns failed; see {d}/gpumd.log")
for i in range(10):
    if not (d / f"out{i}.xyz").is_file() or (d / f"out{i}.xyz").stat().st_size == 0:
        raise AssertionError(f"missing output from rerun {i}")
print("PASS ten sequential reruns: all outputs created, process exited cleanly")
