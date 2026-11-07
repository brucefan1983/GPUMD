#!/usr/bin/env python3
"""H100 runtime regression matrix for rerun selection and output scheduling."""

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path("/path/to/GPUMD_SOURCE")
GPUMD = Path(os.environ.get("RERUN_GPUMD", ROOT / "bin-rerun-v1-6f548c78-sm90/gpumd"))
TEST_ROOT = Path(os.environ.get(
    "RERUN_TEST_ROOT",
    "/path/to/RERUN_TEST_ROOT/latest_6f548c78/selection_final3",
))
SOURCE = Path(os.environ.get(
    "RERUN_TRAJECTORY",
    "/path/to/RERUN_TEST_ROOT/latest_6f548c78/energy_smoke/rerun_a/trajectory.xyz",
))
MODEL = ROOT / "tests_regression/fixtures/systems/cu_grouped.xyz"
POTENTIAL = ROOT / "potentials/eam/Cu_Zhou_2004.txt"


def parse_xyz(path):
    frames = []
    with open(path) as stream:
        while True:
            count_line = stream.readline()
            if not count_line:
                return frames
            if not count_line.strip():
                continue
            n = int(count_line)
            comment = stream.readline().strip()
            atoms = [stream.readline() for _ in range(n)]
            if any(not row.strip() for row in atoms):
                raise ValueError(f"truncated frame in {path}")
            frames.append((n, comment, atoms))


def set_clock(comment, step):
    comment = re.sub(r"(?<!\S)(?:Step|Time)=[^ \t]+", "", comment).strip()
    return f"Step={step} Time={0.2 * step:.8f} {comment}"


def write_trajectory(path, frames, steps=None):
    with open(path, "w") as out:
        for index, (n, comment, atoms) in enumerate(frames):
            step = index + 1 if steps is None else steps[index]
            out.write(f"{n}\n{set_clock(comment, step)}\n")
            out.writelines(row.rstrip("\n") + "\n" for row in atoms)


def run_case(name, frames, options="", xyz_interval=1, thermo_interval=None, steps=None):
    case_dir = TEST_ROOT / name
    case_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MODEL, case_dir / "model.xyz")
    trajectory = case_dir / "trajectory.xyz"
    write_trajectory(trajectory, frames, steps)
    run_lines = [f"potential {POTENTIAL}", "time_step 0.2"]
    run_lines.append(
        f"dump_xyz {xyz_interval} output.xyz force potential virial"
    )
    if thermo_interval is not None:
        run_lines.append(f"dump_thermo {thermo_interval}")
    suffix = " ".join(options.split())
    run_lines.append("rerun trajectory.xyz" + (f" {suffix}" if suffix else ""))
    (case_dir / "run.in").write_text("\n".join(run_lines) + "\n")
    completed = subprocess.run(
        [str(GPUMD)], cwd=case_dir, stdout=open(case_dir / "gpumd.log", "w"),
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{name}: GPUMD exited {completed.returncode}; see gpumd.log")
    return case_dir


def output_steps(path):
    return [
        int(re.search(r"(?:^|\s)Step=(\d+)", comment).group(1))
        for _, comment, _ in parse_xyz(path)
    ]


def thermo_row_count(path):
    rows = 0
    for line in open(path):
        fields = line.split()
        try:
            float(fields[0])
            rows += 1
        except (ValueError, IndexError):
            continue
    return rows


def expect(name, actual, expected):
    if actual != expected:
        raise AssertionError(f"{name}: got {actual}, expected {expected}")
    print(f"PASS {name}: {actual}")


def main():
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    frames = parse_xyz(SOURCE)
    if len(frames) != 100:
        raise SystemExit(f"source trajectory has {len(frames)} frames, expected 100")

    cases = [
        ("none", "", list(range(1, 101))),
        ("first", "first 20", list(range(20, 101))),
        ("last", "last 50", list(range(1, 51))),
        ("every", "every 30", [1, 30, 60, 90]),
        ("skip", "skip 3", list(range(1, 101, 3))),
        ("combined", "first 20\nlast 90\nevery 30\nskip 2", [20, 60]),
        ("empty", "first 101\nlast 200", []),
    ]
    for name, options, expected in cases:
        case_dir = run_case(name, frames, options)
        expect(name, output_steps(case_dir / "output.xyz"), expected)

    schedule_dir = run_case(
        "scheduling", frames, xyz_interval=20, thermo_interval=25
    )
    expect("dump_xyz interval 20", output_steps(schedule_dir / "output.xyz"), [20, 40, 60, 80, 100])
    expect("dump_thermo interval 25 row count", thermo_row_count(schedule_dir / "thermo.out"), 4)

    sparse_steps = [20, 100, 200, 300, 400, 500]
    sparse_dir = run_case(
        "sparse", [frames[0]] * len(sparse_steps),
        xyz_interval=100, thermo_interval=200, steps=sparse_steps,
    )
    expect("sparse dump_xyz schedule", output_steps(sparse_dir / "output.xyz"), [100, 200, 300, 400, 500])
    expect("sparse dump_thermo interval 200 row count", thermo_row_count(sparse_dir / "thermo.out"), 2)

    nonmonotonic_steps = [1, 2, 2, 1, 3]
    nonmonotonic_dir = run_case(
        "repeated_decreasing", [frames[0]] * len(nonmonotonic_steps),
        steps=nonmonotonic_steps,
    )
    expect("repeat/decrease skip", output_steps(nonmonotonic_dir / "output.xyz"), [1, 2, 3])

    zero_dir = run_case("step_zero", [frames[0]], steps=[0])
    expect("Step 0 output", output_steps(zero_dir / "output.xyz"), [0])

    # Sequential reruns must each finalize their output action and continue.
    seq_dir = TEST_ROOT / "sequential"
    seq_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MODEL, seq_dir / "model.xyz")
    write_trajectory(seq_dir / "first.xyz", frames[:2])
    write_trajectory(seq_dir / "second.xyz", frames[2:4], steps=[3, 4])
    (seq_dir / "run.in").write_text(
        f"potential {POTENTIAL}\ntime_step 0.2\n"
        "dump_xyz 1 first_out.xyz force potential virial\nrerun first.xyz\n"
        "dump_xyz 1 second_out.xyz force potential virial\nrerun second.xyz\n"
    )
    completed = subprocess.run(
        [str(GPUMD)], cwd=seq_dir, stdout=open(seq_dir / "gpumd.log", "w"),
        stderr=subprocess.STDOUT,
    )
    if completed.returncode:
        raise RuntimeError("sequential rerun commands failed; see sequential/gpumd.log")
    expect("sequential rerun 1", output_steps(seq_dir / "first_out.xyz"), [1, 2])
    expect("sequential rerun 2", output_steps(seq_dir / "second_out.xyz"), [3, 4])

    # Verify exact-name collision rejection, then reproduce a relative-path alias.
    collision_dir = TEST_ROOT / "collision_exact"
    collision_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MODEL, collision_dir / "model.xyz")
    write_trajectory(collision_dir / "trajectory.xyz", frames[:2])
    before = hashlib.sha256((collision_dir / "trajectory.xyz").read_bytes()).hexdigest()
    (collision_dir / "run.in").write_text(
        f"potential {POTENTIAL}\ntime_step 0.2\n"
        "dump_xyz 1 trajectory.xyz force potential virial\nrerun trajectory.xyz\n"
    )
    result = subprocess.run(
        [str(GPUMD)], cwd=collision_dir,
        stdout=open(collision_dir / "gpumd.log", "w"), stderr=subprocess.STDOUT,
    )
    after = hashlib.sha256((collision_dir / "trajectory.xyz").read_bytes()).hexdigest()
    if result.returncode == 0 or before != after:
        raise AssertionError("exact path collision was not safely rejected")
    print("PASS exact input/output collision rejection")

    alias_dir = TEST_ROOT / "collision_alias"
    alias_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MODEL, alias_dir / "model.xyz")
    write_trajectory(alias_dir / "trajectory.xyz", frames[:2])
    before = hashlib.sha256((alias_dir / "trajectory.xyz").read_bytes()).hexdigest()
    (alias_dir / "run.in").write_text(
        f"potential {POTENTIAL}\ntime_step 0.2\n"
        "dump_xyz 1 ./trajectory.xyz force potential virial\nrerun trajectory.xyz\n"
    )
    timed_out = False
    try:
        result = subprocess.run(
            [str(GPUMD)], cwd=alias_dir,
            stdout=open(alias_dir / "gpumd.log", "w"), stderr=subprocess.STDOUT,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        timed_out = True
        result = None
    after = hashlib.sha256((alias_dir / "trajectory.xyz").read_bytes()).hexdigest()
    if result is not None and result.returncode != 0:
        print("PASS relative-path alias collision rejection")
    elif timed_out or before != after:
        print("FAIL reproduced relative-path alias collision: accepted, modified input, or hung")
        (TEST_ROOT / "collision_alias.FAIL").write_text(
            "rerun trajectory.xyz with dump_xyz ./trajectory.xyz was not rejected safely\n"
        )
    else:
        raise AssertionError("relative-path alias unexpectedly accepted without changing file")


if __name__ == "__main__":
    main()
