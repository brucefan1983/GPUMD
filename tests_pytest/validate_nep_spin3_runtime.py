#!/usr/bin/env python3
"""Validate the GPUMD nep4_spin3 force path against the frozen FP64 oracle."""

import argparse
import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path
from spin_model_format import block_model
from spin_fixtures import selective_type_model, write_xyz, reshape, read_text_oracle, load_fixture


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "nep_spin3"
GPUMD = Path(os.environ.get("GPUMD_COMMAND", ROOT / "src" / "gpumd"))












def read_frame(path):
    lines = path.read_text().splitlines()
    atom_count = int(lines[0])
    rows = [line.split() for line in lines[-atom_count:]]
    values = [[float(value) for value in row[1:]] for row in rows]
    energy = float(
        re.search(r"\benergy=([-+0-9.eE]+)", lines[-atom_count - 1]).group(1))
    virial = []
    for row in values:
        # dump_xyz already serializes the internal 9-plane tensor in row-major
        # order (xx, xy, xz, yx, yy, yz, zx, zy, zz).
        virial.append(row[13:22])
    return {
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


def max_error(candidate, reference):
    left = flatten(candidate)
    right = flatten(reference)
    if len(left) != len(right):
        raise AssertionError(f"shape mismatch: {len(left)} != {len(right)}")
    return max(abs(a - b) for a, b in zip(left, right))


def worst_difference(candidate, reference):
    left = flatten(candidate)
    right = flatten(reference)
    index = max(range(len(left)), key=lambda item: abs(left[item] - right[item]))
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


def validate_removed_spin2(root):
    model, oracle = load_fixture("o3c2_uniform")
    case = next(iter(oracle["cases"].values()))
    directory = root / "removed_spin2"
    directory.mkdir()
    write_xyz(directory / "model.xyz", case)
    (directory / "nep.txt").write_text(
        model.replace("nep4_spin3", "nep4_spin2", 1))
    (directory / "run.in").write_text(
        "potential nep.txt\nensemble nve\ntime_step 0\nrun 1\n")
    result = subprocess.run(
        [str(GPUMD)], cwd=directory, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        raise AssertionError("removed nep4_spin2 model was accepted")


def validate_energy_reference(root):
    legacy, oracle = load_fixture("o3c2_uniform")
    model = block_model(legacy)
    case = oracle["cases"]["noncollinear"]
    base = run_case(root, "reference-base", case, model)
    old = next(line for line in model.splitlines() if line.startswith("energy_reference "))
    references = list(map(float, old.split()[1:]))
    shifts = [100000.123456789, -200000.987654321]
    changed = "energy_reference " + " ".join(format(a + b, '.17g') for a, b in zip(references, shifts))
    shifted = run_case(root, "reference-shift", case, model.replace(old, changed))
    expected = base["energy"] + sum(shifts[0 if t == "Fe" else 1] for t in case["types"])
    if abs(shifted["energy"] - expected) > 2e-6:
        raise AssertionError(f"energy reference mismatch: actual={shifted['energy']:.17g}, expected={expected:.17g}")
    for field in ("force", "mforce", "virial"):
        if max_error(base[field], shifted[field]) > 1e-10:
            raise AssertionError(f"constant energy reference changed {field}")
    print("Spin3 double energy-reference shift and derivative invariance passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases", nargs="*", default=None,
        help="oracle cases to run; default runs every case")
    parser.add_argument(
        "--fixtures", nargs="*",
        choices=("o3c2_uniform", "o3c2_typewise", "o3c3_uniform"),
        default=("o3c2_uniform", "o3c2_typewise", "o3c3_uniform"),
        help="frozen Spin3 fixtures to run")
    parser.add_argument("--tolerance", type=float, default=2.0e-4)
    parser.add_argument("--block-format", action="store_true", help="Validate converted block-format fixtures against the same frozen oracle")
    args = parser.parse_args()

    errors = {}
    with tempfile.TemporaryDirectory(prefix="gpumd-spin3-") as temp:
        root = Path(temp)
        validate_removed_spin2(root)
        if args.block_format:
            validate_energy_reference(root)
        for fixture in args.fixtures:
            model, oracle = load_fixture(fixture)
            if args.block_format:
                model = block_model(model)
            selected = args.cases or list(oracle["cases"])
            fixture_root = root / fixture
            fixture_root.mkdir()
            for name in selected:
                if name not in oracle["cases"]:
                    continue
                reference = oracle["cases"][name]
                candidate = run_case(fixture_root, name, reference, model)
                case_errors = {
                    "energy": abs(candidate["energy"] - sum(reference["potential"])),
                    "potential": max_error(candidate["potential"], reference["potential"]),
                    "force": max_error(candidate["force"], reference["force"]),
                    "mforce": max_error(candidate["mforce"], reference["mforce"]),
                    "virial": max_error(
                        candidate["virial"],
                        [sum(row[index] for row in reference["atom_virial"])
                         for index in range(9)]),
                }
                key = f"{fixture}/{name}"
                errors[key] = case_errors
                print(key + ": " + " ".join(
                    f"{field}={error:.3e}" for field, error in case_errors.items()))
                failed = {
                    field: error for field, error in case_errors.items()
                    if error > args.tolerance
                }
                if failed:
                    for field in failed:
                        if field == "energy":
                            target = sum(reference["potential"])
                        elif field == "virial":
                            target = [sum(row[index] for row in reference["atom_virial"])
                                      for index in range(9)]
                        else:
                            target = reference[field]
                        index, actual, expected = worst_difference(candidate[field], target)
                        print(
                            f"{key} {field} worst[{index}]: "
                            f"actual={actual:.10e} expected={expected:.10e}",
                            flush=True)
                    raise AssertionError(
                        f"{key}: tolerance {args.tolerance:.3e} exceeded: {failed}")

    worst = max(error for case in errors.values() for error in case.values())
    print(
        f"spin3 oracle validation passed: cases={len(errors)} "
        f"max_error={worst:.3e}; nep4_spin2 rejected")


if __name__ == "__main__":
    main()
