#!/usr/bin/env python3
"""Validate the GPUMD NEP-spin force path against the frozen FP64 oracle."""

import argparse
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from spin_fixtures import write_xyz
from validate_spin_training import maximum_error, finite_values


ROOT = Path(__file__).resolve().parents[1]
GPUMD = Path(os.environ.get("GPUMD_COMMAND", ROOT / "src" / "gpumd"))


def read_frame(path):
    lines = path.read_text().splitlines()
    atom_count = int(lines[0])
    rows = [line.split() for line in lines[-atom_count:]]
    values = [[float(value) for value in row[1:]] for row in rows]
    finite_values(values)
    energy = float(
        re.search(r"\benergy=([-+0-9.eE]+)", lines[-atom_count - 1]).group(1))
    finite_values([energy])
    virial = []
    for row in values:
        # dump_xyz already serializes the internal 9-plane tensor in row-major
        # order (xx, xy, xz, yx, yy, yz, zx, zy, zz).
        virial.append(row[13:22])
    return {
        "spin": [row[6:9] for row in values],
        "energy": energy,
        "force": [row[3:6] for row in values],
        "mforce": [row[9:12] for row in values],
        "potential": [row[12] for row in values],
        "atom_virial": virial,
        "virial": [sum(row[index] for row in virial) for index in range(9)],
    }


def flatten(values):
    if not isinstance(values, list):
        return [values]
    result = []
    for value in values:
        result.extend(flatten(value))
    return result


max_error = maximum_error


def worst_difference(candidate, reference):
    left = flatten(candidate)
    right = flatten(reference)
    index = max(range(len(left)), key=lambda item: abs(left[item] - right[item]))
    finite_values(left)
    finite_values(right)
    return index, left[index], right[index]


def run_case(root, name, case, model):
    directory = root / name
    directory.mkdir()
    write_xyz(directory / "model.xyz", case)
    (directory / "nep.txt").write_text(model)
    (directory / "run.in").write_text(
        "potential nep.txt\n"
        "velocity 1\n"
        "ensemble nve\n"
        "time_step 0\n"
        "dump_xyz 1 result.xyz force potential spin mforce virial precision double\n"
        "run 1\n")
    result = subprocess.run(
        [str(GPUMD)], cwd=directory, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(
            f"{name}: GPUMD failed with {result.returncode}\n"
            f"{result.stdout}{result.stderr}")
    return read_frame(directory / "result.xyz")
