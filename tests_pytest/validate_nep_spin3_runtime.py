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


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "nep_spin3"
GPUMD = Path(os.environ.get("GPUMD_COMMAND", ROOT / "src" / "gpumd"))


def selective_type_model():
    """Return the production C=2 fixture with Fe DOFs and Fe/Ge environments."""
    return (FIXTURE / "nep4_spin3_o3c2_uniform.nep").read_text()


def write_xyz(path, case):
    lattice = " ".join(f"{value:.17g}" for value in case["cell"])
    lines = [
        str(len(case["types"])),
        f'Lattice="{lattice}" '
        'Properties=species:S:1:pos:R:3:spin:R:3 pbc="T T T"',
    ]
    for symbol, position, spin in zip(
            case["types"], case["positions"], case["spins"]):
        values = position + spin
        lines.append(symbol + " " + " ".join(f"{value:.17g}" for value in values))
    path.write_text("\n".join(lines) + "\n")


def reshape(values, width):
    if len(values) % width:
        raise ValueError(f"cannot reshape {len(values)} values with width {width}")
    return [values[index:index + width] for index in range(0, len(values), width)]


def read_text_oracle(path, model):
    lines = path.read_text().splitlines()
    if lines[0] != "spin3_oc_oracle_v1":
        raise ValueError(f"{path}: unsupported oracle format")
    model_sha256 = lines[1].split()
    if model_sha256[:1] != ["model_sha256"]:
        raise ValueError(f"{path}: missing model hash")
    actual_hash = hashlib.sha256(model.encode()).hexdigest()
    if actual_hash != model_sha256[1]:
        raise ValueError(
            f"{path}: model hash mismatch: {actual_hash} != {model_sha256[1]}")
    descriptor_dim = int(lines[2].split()[1])
    spin_descriptor_dim = int(lines[3].split()[1])
    if descriptor_dim - spin_descriptor_dim != 30 or spin_descriptor_dim not in (55, 91):
        raise ValueError(
            f"{path}: unexpected descriptor dimensions "
            f"{descriptor_dim}/{spin_descriptor_dim}")

    cases = {}
    current = None
    for line in lines[4:]:
        fields = line.split()
        if fields[0] == "case":
            current = {"atom_count": int(fields[2])}
            cases[fields[1]] = current
        elif fields[0] == "types":
            current["types"] = fields[1:]
        elif fields[0] == "end":
            current = None
        else:
            if current is None:
                raise ValueError(f"{path}: data outside a case: {fields[0]}")
            count = int(fields[1])
            values = [float(value) for value in fields[2:]]
            if len(values) != count:
                raise ValueError(
                    f"{path}: {fields[0]} count mismatch: {len(values)} != {count}")
            current[fields[0]] = values

    for name, case in cases.items():
        atom_count = case.pop("atom_count")
        if len(case["types"]) != atom_count:
            raise ValueError(f"{path}: {name} type count mismatch")
        case["positions"] = reshape(case["positions"], 3)
        case["spins"] = reshape(case["spins"], 3)
        case["force"] = reshape(case.pop("forces"), 3)
        case["mforce"] = reshape(case.pop("mforces"), 3)
        case["atom_virial"] = reshape(case.pop("virial"), 9)
        case.pop("tau")
        case.pop("descriptor")
    return {"cases": cases}


def load_fixture(name):
    fixtures = {
        "o3c2_uniform": ("nep4_spin3_o3c2_uniform.nep", "spin3_o3c2_uniform_oracle.txt"),
        "o3c2_typewise": ("nep4_spin3_o3c2_typewise.nep", "spin3_o3c2_typewise_oracle.txt"),
        "o3c3_uniform": ("nep4_spin3_o3c3_uniform.nep", "spin3_o3c3_uniform_oracle.txt"),
    }
    if name not in fixtures:
        raise ValueError(f"unknown fixture: {name}")
    model_name, oracle_name = fixtures[name]
    model = (FIXTURE / model_name).read_text()
    oracle = read_text_oracle(FIXTURE / oracle_name, model)
    return model, oracle


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
