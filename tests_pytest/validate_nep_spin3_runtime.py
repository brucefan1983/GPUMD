#!/usr/bin/env python3
"""Validate the GPUMD NEP-spin force path against the frozen FP64 oracle."""

import argparse
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from spin_fixtures import write_xyz, load_fixture
from validate_main_nep_spin3_training import maximum_error, finite_values


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "nep_spin3"
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


def validate_energy_reference(root):
    model, oracle = load_fixture("o3c2_uniform")
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



def validate_io(root):
    import copy
    import shutil
    model, oracle = load_fixture("o3c2_uniform")
    case = copy.deepcopy(oracle["cases"]["noncollinear"])
    # Include an excluded type with deliberately nonzero input spin.
    case["types"][-1] = "Ge"
    case["spins"][-1] = [0.8, -0.2, 0.4]
    raw = run_case(root, "inactive-raw", case, model)
    case["spins"][-1] = [0.0, 0.0, 0.0]
    zero = run_case(root, "inactive-zero", case, model)
    for field in ("energy", "potential", "force", "mforce", "virial", "spin"):
        assert max_error(raw[field], zero[field]) < 1e-10, field
    assert raw["spin"][-1] == [0.0] * 3 and raw["mforce"][-1] == [0.0] * 3

    # Selection and validation must share the same lexical and ordering rules.
    variants = {
        "uppercase": model.replace("model nep_spin", "MODEL NEP_SPIN"),
        "inline-comment": model.replace("model nep_spin", "model nep_spin#comment"),
        "model-after-blocks": model.replace("model nep_spin\n", "") + "\nmodel nep_spin\n",
    }
    for name, changed in variants.items():
        alternate = run_case(root, name, case, changed)
        for field in ("energy", "force", "mforce", "virial"):
            assert max_error(alternate[field], zero[field]) < 1e-10, (name, field)

    # Small-cell lists must count and grow rather than truncate a short initial allocation.
    reference = oracle["cases"]["noncollinear"]
    grown = run_case(root, "small-capacity-growth", reference,
                     model.replace("cutoff 6 5 128 128", "cutoff 6 5 1 1"))
    regular = run_case(root, "small-capacity-reference", reference, model)
    for field in ("energy", "force", "mforce", "virial"):
        assert max_error(grown[field], regular[field]) < 1e-10, field

    # Restart and replication preserve the fixed-spin model, including type masking.
    source = root / "io-source"
    source.mkdir()
    (source / "nep.txt").write_text(model)
    write_xyz(source / "model.xyz", case)
    (source / "run.in").write_text(
        "replicate 2 1 1\npotential nep.txt\nvelocity 1\nensemble nve\ntime_step 0\n"
        "dump_restart 1\ndump_xyz 1 result.xyz force potential spin mforce virial precision double\nrun 1\n")
    result = subprocess.run([str(GPUMD)], cwd=source, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    before = read_frame(source / "result.xyz")
    expected_spin = case["spins"] * 2
    assert max_error(before["spin"], expected_spin) < 1e-10
    resume = root / "io-resume"
    resume.mkdir()
    shutil.copy(source / "nep.txt", resume / "nep.txt")
    shutil.copy(source / "restart.xyz", resume / "model.xyz")
    (resume / "run.in").write_text(
        "potential nep.txt\nensemble nve\ntime_step 0\n"
        "dump_xyz 1 result.xyz force potential spin mforce virial precision double\nrun 1\n")
    result = subprocess.run([str(GPUMD)], cwd=resume, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    after = read_frame(resume / "result.xyz")
    for field in ("energy", "potential", "force", "mforce", "virial", "spin"):
        assert max_error(before[field], after[field]) < 2e-4, field
    # Use upstream split output with the same Spin quantities and reader.
    (resume / "run.in").write_text(
        "potential nep.txt\nensemble nve\ntime_step 0\n"
        "dump_xyz 1 split.xyz split 1 force potential spin mforce virial precision double\nrun 2\n")
    result = subprocess.run([str(GPUMD)], cwd=resume, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    for path in (resume / "split_0001.xyz", resume / "split_0002.xyz"):
        split = read_frame(path)
        for field in ("energy", "force", "mforce", "virial", "spin"):
            assert max_error(split[field], after[field]) < 2e-4, field
    print("model selection, neighbor growth, masking, replication, restart and split output passed")

def validate_thin_cell(root):
    # 10,000 atoms and two periodic images previously reserved ~19.2 GB.
    model, _ = load_fixture("o3c2_uniform")
    positions = [[1.5 + 3 * i, 2 + 4 * j, 1 + 2 * k]
                 for i in range(2) for j in range(50) for k in range(100)]
    case = {"types": ["Fe"] * len(positions), "positions": positions,
            "spins": [[0.2, -0.1, 2.0]] * len(positions),
            "cell": [6., 0., 0., 0., 200., 0., 0., 0., 200.]}
    small = run_case(root, "thin-cell", case, model)
    expanded = dict(case)
    expanded["cell"] = [18., 0., 0., 0., 200., 0., 0., 0., 200.]
    expanded["positions"] = [[x + 6 * shift, y, z] for shift in range(3) for x, y, z in positions]
    expanded["types"] = case["types"] * 3
    expanded["spins"] = case["spins"] * 3
    large = run_case(root, "thin-cell-expanded", expanded, model)
    for field in ("potential", "force", "mforce"):
        assert max_error(small[field], large[field][:len(positions)]) < 2e-4, field
    assert max_error(small["virial"], [value / 3 for value in large["virial"]]) < 2e-2
    print("10000-atom thin-cell allocation and equivalent expanded-cell comparison passed")


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
    args = parser.parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("tolerance must be finite and positive")

    errors = {}
    with tempfile.TemporaryDirectory(prefix="gpumd-spin3-") as temp:
        root = Path(temp)
        validate_energy_reference(root)
        validate_io(root)
        validate_thin_cell(root)
        for fixture in args.fixtures:
            model, oracle = load_fixture(fixture)
            selected = args.cases or list(oracle["cases"])
            fixture_root = root / fixture
            fixture_root.mkdir()
            for name in selected:
                if name not in oracle["cases"]:
                    continue
                reference = oracle["cases"][name]
                candidate = run_case(fixture_root, name, reference, model)
                case_errors = {
                    "energy": max_error(candidate["energy"], sum(reference["potential"])),
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
        f"max_error={worst:.3e}")


if __name__ == "__main__":
    main()
