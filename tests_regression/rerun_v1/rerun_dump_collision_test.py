#!/usr/bin/env python3
"""Integration checks for rerun dump_xyz input-trajectory protection."""

import argparse
import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_frame(path):
    lines = path.read_text().splitlines(keepends=True)
    count = int(lines[0])
    return lines[: count + 2], count


def frame_step(header, time_step):
    match = re.search(r"(?:^|\s)(?:Step|step|Timestep)=(-?\d+)(?:\s|$)", header)
    if match:
        return int(match.group(1))
    time_match = re.search(r"(?:^|\s)Time=([-+0-9.eE]+)", header)
    if not time_match:
        raise ValueError("source trajectory requires Step or Time metadata")
    return round(float(time_match.group(1)) / time_step)


def execute(gpumd, work, model, potential, trajectory, dump, expect_success, with_plumed):
    shutil.copyfile(model, work / "model.xyz")
    before = digest(trajectory)
    plumed_command = ""
    if with_plumed:
        (work / "plumed.dat").write_text(
            "DISTANCE ATOMS=1,2 LABEL=d\nPRINT ARG=d STRIDE=1 FILE=COLVAR\n"
        )
        plumed_command = "plumed plumed.dat 1 0\n"
    (work / "run.in").write_text(
        f"potential {potential}\ntime_step 0.2\n{plumed_command}{dump}\n"
        f"rerun {trajectory.name}\n"
    )
    result = subprocess.run(
        [str(gpumd)], cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    after = digest(trajectory)
    if not expect_success and before != after:
        raise AssertionError("input trajectory changed after collision attempt")
    if expect_success and before != after:
        raise AssertionError("input trajectory changed during noncolliding output")
    if expect_success != (result.returncode == 0):
        raise AssertionError(
            f"unexpected exit status {result.returncode}; output: {result.stdout[-1200:]}"
        )
    if not expect_success:
        if "must not overwrite the input trajectory" not in result.stdout:
            raise AssertionError("collision rejection diagnostic is missing")
    elif with_plumed:
        colvar = work / "COLVAR"
        if not colvar.exists():
            raise AssertionError("PLUMED did not create COLVAR")
        rows = [line for line in colvar.read_text().splitlines() if line and not line.startswith("#")]
        if len(rows) != 1:
            raise AssertionError(f"expected one PLUMED row, found {len(rows)}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpumd", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--potential", type=Path, required=True)
    parser.add_argument("--with-plumed", action="store_true")
    args = parser.parse_args()
    source_frame, _ = first_frame(args.trajectory)
    step = frame_step(source_frame[1], 0.2)

    with tempfile.TemporaryDirectory(prefix="gpumd-rerun-collision-") as temp:
        root = Path(temp)

        # Normal output is guarded by the early preflight check.
        work = root / "normal"
        work.mkdir()
        trajectory = work / "trajectory.xyz"
        trajectory.write_text("".join(source_frame))
        execute(
            args.gpumd, work, args.model, args.potential, trajectory,
            "dump_xyz 1 trajectory.xyz force potential virial", False, args.with_plumed,
        )
        print("PASS normal direct collision; input hash unchanged")

        for name, output_path, create_alias in (
            ("normal-dotdot", "nested/../trajectory.xyz", lambda work, traj: (work / "nested").mkdir()),
            ("normal-symlink", "alias.xyz", lambda work, traj: (work / "alias.xyz").symlink_to(traj.name)),
            ("normal-absolute", None, lambda work, traj: None),
        ):
            work = root / name
            work.mkdir()
            trajectory = work / "trajectory.xyz"
            trajectory.write_text("".join(source_frame))
            create_alias(work, trajectory)
            configured_path = output_path or str(trajectory.resolve())
            execute(
                args.gpumd, work, args.model, args.potential, trajectory,
                f"dump_xyz 1 {configured_path} force potential virial", False, args.with_plumed,
            )
            print(f"PASS normal {name} alias; input hash unchanged")

        # The per-frame suffix is the original timestep (step+1 in the
        # zero-based end_of_step interface).
        work = root / "per-frame-direct"
        work.mkdir()
        trajectory = work / f"trajectory.xyz{step}"
        trajectory.write_text("".join(source_frame))
        execute(
            args.gpumd, work, args.model, args.potential, trajectory,
            "dump_xyz 1 trajectory.xyz* force potential virial", False, args.with_plumed,
        )
        print("PASS per-frame direct collision; input hash unchanged")

        # A generated output symlink may point at the input through a relative
        # path containing '..'. The actual generated name must be checked.
        work = root / "per-frame-symlink"
        work.mkdir()
        trajectory = work / "trajectory.xyz"
        trajectory.write_text("".join(source_frame))
        alias_dir = work / "alias"
        alias_dir.mkdir()
        alias = alias_dir / f"frame{step}"
        alias.symlink_to(Path("..") / "trajectory.xyz")
        execute(
            args.gpumd, work, args.model, args.potential, trajectory,
            f"dump_xyz 1 alias/../alias/frame* force potential virial", False, args.with_plumed,
        )
        print("PASS per-frame symlink and relative alias; input hash unchanged")

        work = root / "per-frame-absolute-symlink"
        work.mkdir()
        trajectory = work / "trajectory.xyz"
        trajectory.write_text("".join(source_frame))
        alias_dir = work / "alias"
        alias_dir.mkdir()
        (alias_dir / f"frame{step}").symlink_to(Path("..") / "trajectory.xyz")
        execute(
            args.gpumd, work, args.model, args.potential, trajectory,
            f"dump_xyz 1 {alias_dir.resolve()}/frame* force potential virial", False, args.with_plumed,
        )
        print("PASS per-frame absolute symlink alias; input hash unchanged")

        work = root / "per-frame-hardlink"
        work.mkdir()
        trajectory = work / "trajectory.xyz"
        trajectory.write_text("".join(source_frame))
        (work / f"frame{step}").hardlink_to(trajectory)
        execute(
            args.gpumd, work, args.model, args.potential, trajectory,
            "dump_xyz 1 frame* force potential virial", False, args.with_plumed,
        )
        print("PASS per-frame hard-link alias; input hash unchanged")

        # split uses numbered names and deliberately advances past an existing
        # chunk. Confirm the input chunk is preserved and the next chunk works.
        work = root / "split-existing-input"
        work.mkdir()
        trajectory = work / "trajectory_0001.xyz"
        trajectory.write_text("".join(source_frame))
        execute(
            args.gpumd, work, args.model, args.potential, trajectory,
            "dump_xyz 1 trajectory.xyz split 1 force potential virial", True, args.with_plumed,
        )
        if not (work / "trajectory_0002.xyz").exists():
            raise AssertionError("split dump did not generate its next noncolliding chunk")
        print("PASS split collision candidate skipped; next chunk written safely")

        work = root / "split-symlink-input"
        work.mkdir()
        trajectory = work / "trajectory.xyz"
        trajectory.write_text("".join(source_frame))
        (work / "output_0001.xyz").symlink_to(trajectory.name)
        execute(
            args.gpumd, work, args.model, args.potential, trajectory,
            "dump_xyz 1 output.xyz split 1 force potential virial", True, args.with_plumed,
        )
        if not (work / "output_0002.xyz").exists():
            raise AssertionError("split dump did not skip an aliased chunk path")
        print("PASS split symlink collision candidate skipped safely")

        # Verify successful output in all three modes.
        for name, dump, expected in (
            ("normal-ok", "dump_xyz 1 output.xyz force potential virial", "output.xyz"),
            ("per-frame-ok", "dump_xyz 1 output.xyz* force potential virial", f"output.xyz{step}"),
            ("split-ok", "dump_xyz 1 output.xyz split 1 force potential virial", "output_0001.xyz"),
        ):
            work = root / name
            work.mkdir()
            trajectory = work / "trajectory.xyz"
            trajectory.write_text("".join(source_frame))
            execute(
                args.gpumd, work, args.model, args.potential, trajectory, dump, True,
                args.with_plumed,
            )
            if not (work / expected).exists():
                raise AssertionError(f"expected noncolliding output {expected} was not created")
            print(f"PASS noncolliding {name} output")


if __name__ == "__main__":
    main()
