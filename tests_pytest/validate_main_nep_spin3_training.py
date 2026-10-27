#!/usr/bin/env python3
"""Validate spin3 training, checkpoint prediction, and restart parity."""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEP = Path(os.environ.get("NEP_COMMAND", ROOT / "src" / "nep"))
GPUMD = Path(os.environ.get("GPUMD_COMMAND", ROOT / "src" / "gpumd"))

NEP_IN = """\
type 2 Fe Ge
version 4
spin_version 1
spin_dof_type Fe
spin_env_type Fe Ge
cutoff 6.0 5.0
n_max 0 0
basis_size 0 0
l_max 2 0 0
neuron 4
spin_compress 2
spin_basis_size 8 0
spin_l_max 2 0 0
spin_cutoff 6.0
spin_order 3
spin_soc 1
lambda_m 1.0
lambda_tau 0.5
population 10
batch 3
generation 1
output_interval 1
save_potential 1 0 0
"""

TRAIN_XYZ = """\
4
Lattice="24 0 0 0 24 0 0 0 24" Properties=species:S:1:pos:R:3:force:R:3:spin:R:3:mforce:R:3 energy=-4.0 virial="0 0 0 0 0 0 0 0 0"
Fe 1.0 1.0 1.0 0 0 0 1.0 0.1 0.0 0.2 -0.1 0.05
Fe 2.2 1.1 1.0 0 0 0 0.8 -0.2 0.1 -0.1 0.2 -0.05
Ge 1.4 2.3 1.1 0 0 0 0.0 0.0 0.0 9.0 9.0 9.0
Ge 2.8 2.1 1.4 0 0 0 0.1 0.0 0.0 9.0 9.0 9.0
4
Lattice="24 0 0 0 24 0 0 0 24" Properties=species:S:1:pos:R:3:force:R:3:spin:R:3:mforce:R:3 energy=-2.0 virial="0 0 0 0 0 0 0 0 0"
Fe 1.2 1.0 1.0 0 0 0 0.9 0.0 0.2 0.1 0.0 -0.2
Ge 2.1 1.4 1.0 0 0 0 0.0 0.0 0.0 8.0 8.0 8.0
Ge 1.5 2.4 1.2 0 0 0 0.0 0.1 0.0 8.0 8.0 8.0
Ge 2.8 2.5 1.5 0 0 0 0.0 0.0 0.1 8.0 8.0 8.0
4
Lattice="24 0 0 0 24 0 0 0 24" Properties=species:S:1:pos:R:3:force:R:3:spin:R:3:mforce:R:3 energy=-6.0 virial="0 0 0 0 0 0 0 0 0"
Fe 1.0 1.0 1.0 0 0 0 1.0 0.0 0.0 0.0 0.1 0.0
Fe 2.0 1.0 1.0 0 0 0 0.8 0.1 0.0 0.1 0.0 0.0
Fe 1.5 2.0 1.0 0 0 0 0.9 0.0 0.1 0.0 0.0 0.1
Ge 2.5 2.0 1.2 0 0 0 0.0 0.0 0.0 7.0 7.0 7.0
"""


def run(binary, cwd):
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return subprocess.run(
        [str(binary)], cwd=cwd, env=env, capture_output=True, text=True,
        check=False)


def write_case(directory, nep_in=NEP_IN, train_xyz=TRAIN_XYZ):
    directory.mkdir()
    (directory / "nep.in").write_text(nep_in)
    (directory / "train.xyz").write_text(train_xyz)


def response_xyz():
    lines = TRAIN_XYZ.splitlines()
    output = []
    atom_count = int(lines[0])
    for coordinate in (-1.0, 0.0, 1.0):
        output.append(lines[0])
        output.append(
            f"{lines[1]} response_probe=rotation response_group=scan-a "
            f"response_coordinate={coordinate}")
        for atom in range(atom_count):
            fields = lines[2 + atom].split()
            if atom == 0:
                fields[7:10] = [
                    f"{math.cos(coordinate):.16e}",
                    f"{math.sin(coordinate):.16e}",
                    "0.0",
                ]
            output.append(" ".join(fields))
    return "\n".join(output) + "\n"


def columns(path, count, limit=None):
    rows = [
        [float(item) for item in line.split()[:count]]
        for line in path.read_text().splitlines()]
    return rows if limit is None else rows[:limit]


def flatten(values):
    if not isinstance(values, list):
        return [values]
    result = []
    for value in values:
        result.extend(flatten(value))
    return result


def maximum_error(left, right):
    left = flatten(left)
    right = flatten(right)
    if len(left) != len(right):
        raise AssertionError(f"shape mismatch: {len(left)} != {len(right)}")
    return max(abs(a - b) for a, b in zip(left, right))


def parity_tolerance(left, right):
    scale = max(abs(value) for value in flatten(left) + flatten(right))
    return 5.0e-5 + 5.0e-6 * scale


def validate_parser(root):
    root.mkdir()
    cases = {
        "invalid_basis": NEP_IN.replace("spin_basis_size 8 0", "spin_basis_size 7 0"),
        "invalid_second_basis": NEP_IN.replace("spin_basis_size 8 0", "spin_basis_size 8 1"),
        "invalid_order": NEP_IN.replace("spin_order 3", "spin_order 4"),
        "invalid_soc": NEP_IN.replace("spin_soc 1", "spin_soc 2"),
        "oversized_spin_shape": NEP_IN.replace("spin_compress 2", "spin_compress 4"),
        "removed_spin_mode": NEP_IN.replace("spin_version 1", "spin_mode 3"),
        "invalid_spin_cutoff_arity": NEP_IN.replace(
            "spin_cutoff 6.0", "spin_cutoff 4.0 5.0 6.0"),
        "removed_spin_chiral": NEP_IN.replace("spin_soc 1", "spin_soc 1\nspin_chiral 1"),
        "removed_curriculum": NEP_IN.replace("spin_order 3", "spin_order 2").replace(
            "lambda_tau 0.5", "lambda_tau 0.5\nspin_curriculum 1"),
        "response_requires_metadata": NEP_IN.replace(
            "lambda_tau 0.5", "lambda_tau 0.5\nlambda_spin_response 0.3"),
    }
    for value in ("0", "2", "1.0", "nan", "1 2", ""):
        cases[f"invalid_spin_version_{value}"] = NEP_IN.replace("spin_version 1", f"spin_version {value}")
    cases["duplicate_spin_version"] = NEP_IN + "spin_version 1\n"
    cases["mixed_spin_selectors"] = NEP_IN + "spin_mode 3\n"
    for keyword in ("model_type 1", "charge_mode 1", "vdw 1", "charge_vdw 1"):
        for first in (False, True):
            # The complete selector input is checked before reading Spin keywords.
            cases[f"conflicting_{keyword}_{first}"] = (
                keyword + "\n" + NEP_IN if first else NEP_IN + keyword + "\n")
    cases["removed_mforce_mode"] = NEP_IN + "spin_mforce_mode full\n"
    results = {}
    for name, text in cases.items():
        case = root / name
        write_case(case, text.replace("generation 1", "generation 0"))
        result = run(NEP, case)
        results[name] = result.returncode
        if result.returncode == 0:
            raise AssertionError(f"{name} did not fail closed")
    # Validate after all keywords: both orders reject oversized uniform/typewise
    # cutoffs and accept a final radial cutoff larger than its default of 8 A.
    for spin_cutoff in ("7.0", "5.0 7.0", "9.0"):
        for spin_first in (False, True):
            name = f"cutoff_bound_{spin_cutoff}_{spin_first}"
            radial = "9.0 5.0" if spin_cutoff == "9.0" else "6.0 5.0"
            text = NEP_IN.replace("cutoff 6.0 5.0\n", "").replace("spin_cutoff 6.0\n", "")
            cutoffs = [f"cutoff {radial}", f"spin_cutoff {spin_cutoff}"]
            text += "\n".join(reversed(cutoffs) if spin_first else cutoffs) + "\n"
            case = root / name
            write_case(case, text)
            result = run(NEP, case)
            results[name] = result.returncode
            if spin_cutoff == "9.0":
                assert result.returncode == 0, result.stdout + result.stderr
            else:
                assert result.returncode != 0
                assert "spin_cutoff must not exceed the NEP radial cutoff" in result.stdout + result.stderr
    tangent_case = root / "spin_tangent_is_not_a_label"
    tangent_input = NEP_IN.replace(
        "lambda_tau 0.5", "lambda_tau 0.5\nlambda_spin_response 0.3",
    ).replace("generation 1", "generation 0")
    tangent_xyz = response_xyz().replace(
        "mforce:R:3", "mforce:R:3:spin_tangent:R:3")
    tangent_xyz = "\n".join(line + " 0 0 0" if i % 6 >= 2 else line
                            for i, line in enumerate(tangent_xyz.splitlines())) + "\n"
    write_case(tangent_case, tangent_input, tangent_xyz)
    result = run(NEP, tangent_case)
    results["spin_tangent_is_not_a_label"] = result.returncode
    if result.returncode == 0 or "not an input label" not in result.stdout + result.stderr:
        raise AssertionError("spin_tangent input was not rejected explicitly")
    return results


def validate_response_training(root):
    nep_in = NEP_IN.replace(
        "lambda_tau 0.5",
        "lambda_tau 0.5\nlambda_spin_response 0.3",
    ).replace("generation 1", "generation 3")
    write_case(root, nep_in.replace("batch 3", "batch 1"))
    (root / "response.xyz").write_text(response_xyz())
    result = run(NEP, root)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    loss = (root / "loss.out").read_text().splitlines()[-1]
    if re.search(r"\b(?:nan|inf)\b", loss, re.IGNORECASE):
        raise AssertionError("response training produced a non-finite loss")
    return {"training_loss": loss}


def validate_flexible_zbl_rejected(root):
    from spin_fixtures import load_fixture

    root.mkdir()
    training = root / "training"
    write_case(training, NEP_IN + "zbl 2.5\n")
    # Malformed contents must never reach the flexible parameter scanner.
    (training / "zbl.in").write_text("not ZBL parameters\n")
    result = run(NEP, training)
    message = "NEP-spin does not support flexible ZBL"
    assert result.returncode != 0 and message in result.stdout + result.stderr
    model, _ = load_fixture("o3c2_uniform")
    model = model.replace("model nep_spin\n", "model nep_spin\nzbl 0 0\n")
    case = root / "prediction"
    write_case(case, NEP_IN + "zbl 2.5\nprediction 1\n")
    (case / "nep.txt").write_text(model)
    result = run(NEP, case)
    assert result.returncode != 0 and message in result.stdout + result.stderr
    return "training and prediction reject flexible ZBL"



def validate_training_and_prediction(
        root, enable_zbl=False, spin_compress=2, typewise_spin_cutoff=False,
        all_spin_types=False, check_restart=False, torque_only=False, small_cell=False):
    root.mkdir()
    training = root / "training"
    nep_in = NEP_IN.replace("version 4", "version 4\nzbl 2.5") \
        if enable_zbl else NEP_IN
    nep_in = nep_in.replace(
        "spin_compress 2", f"spin_compress {spin_compress}")
    if torque_only:
        nep_in = nep_in.replace("lambda_m 1.0", "lambda_m 0")
    if all_spin_types:
        nep_in = nep_in.replace("spin_dof_type Fe\n", "spin_dof_type Fe Ge\n")
    if typewise_spin_cutoff:
        nep_in = nep_in.replace("spin_cutoff 6.0", "spin_cutoff 5.0 6.0")
    if check_restart:
        # Nontrivial channel/basis axes expose incorrect parameter transposes.
        nep_in = nep_in.replace("n_max 0 0", "n_max 4 4").replace(
            "basis_size 0 0", "basis_size 2 2")
    xyz = TRAIN_XYZ
    if small_cell:
        nep_in = nep_in.replace("cutoff 6.0 5.0", "cutoff 7.0 5.0").replace(
            "spin_cutoff 6.0", "spin_cutoff 7.0").replace("batch 3", "batch 1")
        xyz = xyz.replace("24 0 0 0 24 0 0 0 24", "13 0 0 0 13 0 0 0 13").replace("Ge 2.8", "Ge 8.0")
    write_case(training, nep_in, xyz)
    result = run(NEP, training)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)

    checkpoint = (training / "nep.txt").read_text()
    if not checkpoint.startswith("nep 2 Fe Ge\nformat_version 1\nmodel nep_spin\n"):
        raise AssertionError("invalid block-format spin3 checkpoint")
    if "\nspin_version 1\n" not in checkpoint or "\nspin_mode " in checkpoint:
        raise AssertionError("Spin models must declare spin_version instead of spin_mode")
    if "\nenergy_reference " not in checkpoint or "spin_baseline" in checkpoint:
        raise AssertionError("Spin3 must use the shared energy reference")
    if enable_zbl and "\nzbl 1.25 2.5\n" not in checkpoint:
        raise AssertionError("spin3 ZBL checkpoint is missing its zbl line")
    required = {
        "spin_basis_size": "8", "spin_l_max": "2", "spin_order": "3",
        "spin_soc": "1", "spin_scaler": "1", "spin_compress": str(spin_compress),
        "spin_projection_size": str(4 * spin_compress * spin_compress),
        "spin_cutoff": "5 6" if typewise_spin_cutoff else "7" if small_cell else "6",
    }
    if not all_spin_types:
        required.update(spin_dof_type="Fe", spin_env_type="Fe Ge")
    else:
        assert "spin_dof_type" not in checkpoint and "spin_env_type" not in checkpoint, (
            "all-active masks should use the canonical implicit representation")
    for key, value in required.items():
        assert f"\n{key} {value}\n" in checkpoint, f"missing checkpoint field: {key} {value}"

    loss = (training / "loss.out").read_text().splitlines()[-1]
    if re.search(r"\b(?:nan|inf)\b", loss, re.IGNORECASE):
        raise AssertionError("spin3 training produced a non-finite loss")
    loss_values = [float(value) for value in loss.split()]
    regularization_only = loss_values[2] + loss_values[3]
    if abs(loss_values[1] - regularization_only) > 2.0e-5:
        raise AssertionError(
            "generation zero must use upstream regularization-only fitness")

    spin_descriptor_dims = {1: 21, 2: 55, 3: 91}
    structural_dim = 15 if check_restart else 3
    descriptor_dim = structural_dim + spin_descriptor_dims[spin_compress]
    q_scaler = [
        float(value) for value in checkpoint.splitlines()[-descriptor_dim:]]
    # The production magnetic scaler downscales but never amplifies channels.
    spin_scaler = q_scaler[structural_dim:]
    if any(not math.isfinite(value) or value <= 0 for value in q_scaler):
        raise AssertionError("spin3 q_scaler must be positive and finite")
    if max(spin_scaler) > 1.0:
        raise AssertionError(
            f"spin3 magnetic q_scaler must not amplify channels: max={max(spin_scaler)}")

    prediction = root / "prediction"
    prediction.mkdir()
    shutil.copy(training / "nep.txt", prediction / "nep.txt")
    shutil.copy(training / "train.xyz", prediction / "train.xyz")
    prediction_in = re.sub(
        r"^(population|generation|output_interval|save_potential).*$",
        "", nep_in, flags=re.MULTILINE)
    prediction_in = prediction_in.replace(
        "version 4", "version 4\nprediction 1")
    (prediction / "nep.in").write_text(prediction_in)
    result = run(NEP, prediction)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)

    report = {"training_loss": loss, "descriptor_dim": descriptor_dim}
    for name, width in (("energy", 2), ("force", 6), ("mforce", 6), ("virial", 12)):
        rows = columns(prediction / (name + "_train.out"), width)
        assert rows and all(math.isfinite(x) for x in flatten(rows)), name
    if check_restart:
        # Zero search variance freezes the saved parameter vector. A restart must
        # round-trip structural and magnetic coefficients without reordering them
        # or refitting the checkpoint's elemental energy baseline.
        checkpoint_lines = checkpoint.splitlines()
        first_parameter = next(
            i + 1 for i, line in enumerate(checkpoint_lines) if line.startswith("@ parameters "))
        parameters = checkpoint_lines[first_parameter:-descriptor_dim - 1]
        resumed = root / "resumed"
        write_case(resumed, nep_in, TRAIN_XYZ.replace("energy=-4.0", "energy=40.0"))
        (resumed / "nep.txt").write_text(checkpoint)
        (resumed / "nep.restart").write_text(
            "nep_restart\nenergy_reference 2 " + next(line.split(" ", 1)[1] for line in checkpoint_lines if line.startswith("energy_reference ")) + "\nparameters " + str(len(parameters)) + "\n" + "".join(f"{value} 0.0\n" for value in parameters))
        result = run(NEP, resumed)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        resumed_lines = (resumed / "nep.txt").read_text().splitlines()
        if resumed_lines[:first_parameter] != checkpoint_lines[:first_parameter]:
            raise AssertionError("restart changed the Spin3 header or energy baseline")
        error = maximum_error(
            [float(value) for value in parameters],
            [float(line.split()[0]) for line in (resumed / "nep.restart").read_text().splitlines()[3:]])
        if error > 1.0e-7:
            raise AssertionError(f"zero-variance restart changed parameters: {error}")
        if resumed_lines[-descriptor_dim:] != checkpoint_lines[-descriptor_dim:]:
            raise AssertionError("restart changed saved descriptor scaling")
        report["restart_parameter_error"] = error
        report["restart_scaler"] = "preserved"
        obsolete = root / "obsolete-restart"
        write_case(obsolete, nep_in)
        (obsolete / "nep.txt").write_text(checkpoint)
        (obsolete / "nep.restart").write_text("".join(f"{v} 0\n" for v in parameters))
        result = run(NEP, obsolete)
        assert result.returncode != 0, "Old headerless restarts must be rejected"

    return report


def validate_native_compiled_checkpoint(root):
    """Compare full outputs at fixed parameters, including cross-block scatter."""
    root.mkdir()
    # 36 centers span multiple fused-force tiles; Fe and Ge both carry spin.
    atoms = TRAIN_XYZ.splitlines()[2:6]
    expanded = ["36", TRAIN_XYZ.splitlines()[1]]
    for tile in range(9):
        for atom in atoms:
            fields = atom.split()
            fields[1] = str(float(fields[1]) + 4.0 * (tile % 3))
            fields[2] = str(float(fields[2]) + 4.0 * (tile // 3))
            expanded.append(" ".join(fields))
    xyz = "\n".join(expanded) + "\n"
    report = {}
    for channels, order in ((1, 3), (2, 3), (3, 3), (9, 1)):
        config = NEP_IN.replace("spin_compress 2", f"spin_compress {channels}")
        config = config.replace("spin_order 3", f"spin_order {order}")
        config = config.replace("spin_dof_type Fe\n", "spin_dof_type Fe Ge\n")
        config = config.replace("n_max 0 0", "n_max 4 4").replace(
            "basis_size 0 0", "basis_size 2 2")
        if channels == 3:
            config = config.replace("spin_cutoff 6.0", "spin_cutoff 5.0 6.0")
        initial = root / f"c{channels}-initial"
        write_case(initial, config + "nep_compile 0\n", xyz)
        result = run(NEP, initial)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        checkpoint = (initial / "nep.txt").read_text()
        restart_lines = (initial / "nep.restart").read_text().splitlines()
        frozen = "\n".join(restart_lines[:3]) + "\n" + "".join(
            f"{line.split()[0]} 0\n" for line in restart_lines[3:])
        cases = []
        for compiled in (0, 1):
            case = root / f"c{channels}-compiled-{compiled}"
            # Report test outputs at fixed parameters after the first physical evaluation.
            output_config = config.replace("generation 1\n", "generation 2\n")
            write_case(case, output_config + f"nep_compile {compiled}\n", xyz)
            (case / "test.xyz").write_text(xyz)
            (case / "nep.txt").write_text(checkpoint)
            (case / "nep.restart").write_text(frozen)
            result = run(NEP, case)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            if compiled and (
                    "Compile specialized Spin3 training kernels" not in result.stdout or
                    "Spin layout JIT ready:" not in result.stdout or
                    "specialization disabled" in (result.stdout + result.stderr).lower()):
                raise AssertionError("Spin3 JIT did not compile successfully")
            cases.append(case)
        errors = {}
        for name, width in (("energy", 1), ("force", 3), ("virial", 6), ("mforce", 3)):
            values = [columns(case / f"{name}_test.out", width) for case in cases]
            error = maximum_error(*values)
            tolerance = parity_tolerance(*values)
            if not math.isfinite(error) or error > tolerance:
                raise AssertionError(f"C{channels} JIT {name}: {error} > {tolerance}")
            errors[name] = {"max_abs": error, "tolerance": tolerance}
        report[f"o{order}c{channels}"] = errors
    return report


def validate_default_shape(root):
    """Exercise model defaults and compare omitted vs explicit C2 end to end."""
    root.mkdir()
    # Only select the elements/model and bound the training budget and output.
    # Leave all structural, Spin descriptor, network and loss settings implicit.
    config = (
        "type 2 Fe Ge\nspin_version 1\nseed 20261025\n"
        "population 12\nbatch 3\ngeneration 3\noutput_interval 1\n"
        "save_potential 100000 1 0\n")
    cases = [root / "implicit", root / "explicit"]
    configs = [config, config + "spin_compress 2\n"]
    for case, text in zip(cases, configs):
        write_case(case, text)
    # Without ZBL enabled, a stray zbl.in must not affect any lifecycle stage.
    (cases[0] / "zbl.in").write_text("not ZBL parameters\n")
    report = {}
    for stage in ("training", "prediction", "restart"):
        for case, text in zip(cases, configs):
            if stage == "prediction":
                text += "prediction 1\n"
            elif stage == "restart":
                text = text.replace("generation 3\n", "generation 4\n")
            (case / "nep.in").write_text(text)
            result = run(NEP, case)
            (case / f"{stage}.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            checkpoint = (case / "nep.txt").read_text()
            for line in ("spin_compress 2", "spin_order 3", "spin_soc 1",
                         "spin_l_max 2", "spin_basis_size 8",
                         "spin_cutoff 8", "spin_projection_size 16"):
                if f"\n{line}\n" not in checkpoint:
                    raise AssertionError(f"default checkpoint missing {line}")
            if "@ q_scaler 97\n" not in checkpoint:
                raise AssertionError("default descriptor must have 42 + 55 coordinates")
            loss = [float(value) for value in
                    (case / "loss.out").read_text().splitlines()[-1].split()]
            if not all(math.isfinite(value) for value in loss):
                raise AssertionError(f"{stage}: non-finite default-shape loss")
        files = ["nep.txt", "nep.restart"]
        if stage == "prediction":
            for name, width, rows in (("energy", 2, 3), ("force", 6, 12),
                                      ("mforce", 6, 12), ("virial", 12, 3)):
                filename = f"{name}_train.out"
                files.append(filename)
                values = columns(cases[0] / filename, width)
                if len(values) != rows or any(len(row) != width for row in values):
                    raise AssertionError(f"unexpected default prediction shape: {filename}")
                if not all(math.isfinite(value) for value in flatten(values)):
                    raise AssertionError(f"non-finite default prediction: {filename}")
            energies = columns(cases[0] / "energy_train.out", 2)
            report["energy_rmse_per_atom"] = math.sqrt(
                sum((predicted - reference) ** 2 for predicted, reference in energies)
                / len(energies))
        for filename in files:
            if (cases[0] / filename).read_bytes() != (cases[1] / filename).read_bytes():
                raise AssertionError(f"{stage}: implicit/explicit C2 differ in {filename}")
        report[stage] = "implicit/explicit C2 byte-identical"
    report["descriptor_dim"] = 97
    return report


def validate_shared_nep_io(root):
    """Guard the ordinary NEP callers of the shared writer and restart reader."""
    config = "\n".join(line for line in NEP_IN.splitlines()
        if not line.startswith(("spin_", "lambda_m", "lambda_tau"))) + "\n"
    write_case(root, config)
    result = run(NEP, root)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    first = (root / "nep.restart").read_text().splitlines()[1]
    result = run(NEP, root)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    if (root / "nep.restart").read_text().splitlines()[1] != first:
        raise AssertionError("ordinary NEP restart changed energy references")
    (root / "nep.in").write_text(config + "prediction 1\n")
    result = run(NEP, root)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    predictions = [(root / (name + "_train.out")).read_text() for name in ("energy", "force", "virial")]
    (root / "nep.in").write_text(config + "prediction 1\n")
    result = run(NEP, root)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    assert predictions == [(root / (name + "_train.out")).read_text() for name in ("energy", "force", "virial")]
    return "training/restart/prediction passed"


def validate_fixed_models(root):
    """Use the frozen FP64 oracle, independent of the model reader and writer."""
    from spin_fixtures import load_fixture

    root.mkdir()
    report = {}
    for fixture in ("o3c2_uniform", "o3c2_typewise", "o3c3_uniform"):
        model, oracle = load_fixture(fixture)
        fields = {line.split()[0]: line.split()[1:] for line in model.splitlines()
                  if line and not line.startswith('@')}
        config = ("type 2 Fe Ge\nversion 4\nspin_version 1\nprediction 1\n"
                  "n_max 4 4\nbasis_size 8 8\nl_max 4 2 0\nneuron 30\n"
                  "cutoff 6 5\nspin_basis_size 8 0\nspin_l_max 2 0 0\n"
                  "spin_order 3\nspin_soc 1\nspin_compress " + fields['spin_compress'][0] +
                  "\nspin_cutoff " + ' '.join(fields['spin_cutoff']) + "\n")
        cases = list(oracle['cases'].items())
        xyz = []
        for _, case in cases:
            xyz += [str(len(case['types'])), 'Lattice="' + ' '.join(map(str, case['cell'])) +
                    '" Properties=species:S:1:pos:R:3:force:R:3:spin:R:3:mforce:R:3 '
                    'energy=0 virial="0 0 0 0 0 0 0 0 0"']
            for symbol, position, spin in zip(case['types'], case['positions'], case['spins']):
                xyz.append(symbol + ' ' + ' '.join(map(str, position + [0]*3 + spin + [0]*3)))
        directory = root / fixture
        write_case(directory, config, '\n'.join(xyz) + '\n')
        (directory / 'nep.txt').write_text(model)
        result = run(NEP, directory)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        energy = columns(directory / 'energy_train.out', 1)
        force = columns(directory / 'force_train.out', 3)
        mforce = columns(directory / 'mforce_train.out', 3)
        virial = columns(directory / 'virial_train.out', 6)
        assert len(energy) == len(virial) == len(cases)
        assert len(force) == len(mforce) == sum(len(c['types']) for _, c in cases)
        atom = 0
        for index, (name, case) in enumerate(cases):
            count = len(case['types'])
            errors = {
                'energy': abs(energy[index][0] * count - sum(case['potential'])),
                'force': maximum_error(force[atom:atom+count], case['force']),
                'mforce': maximum_error(mforce[atom:atom+count], case['mforce']),
                'virial': maximum_error([v * count for v in virial[index]],
                    [sum(row[j] for row in case['atom_virial']) for j in (0, 4, 8, 1, 5, 6)]),
            }
            assert all(math.isfinite(v) for v in errors.values()), (fixture, name, errors)
            assert all(errors[k] <= 2e-4 for k in ('energy', 'mforce', 'virial')), (fixture, name, errors)
            # NEP stores positions in float32. The cutoff-boundary fixture also
            # contains a 0.002 A pair: its stored separation differs by 7.25e-5
            # relatively from the FP64 oracle. Allow that coordinate precision
            # in forces; the other frozen cases retain the absolute-only bound.
            force_rtol = 1e-4 if name == 'cutoff_boundary' else 0
            for actual, expected in zip(flatten(force[atom:atom+count]), flatten(case['force'])):
                assert abs(actual - expected) <= max(2e-4, force_rtol * abs(expected)), (
                    fixture, name, actual, expected)
            report[fixture + '/' + name] = errors
            atom += count
    return report


def main():
    global NEP
    parser = argparse.ArgumentParser()
    parser.add_argument("--nep", type=Path, default=NEP)
    parser.add_argument("--check-jit", action="store_true",
                        help="Require the CUDA compiler and validate specialized Spin3 training")
    args = parser.parse_args()
    NEP = args.nep.resolve()
    with tempfile.TemporaryDirectory(prefix="gpumd-main-nep-spin3-") as tmp:
        root = Path(tmp)
        report = {
            "fixed_models": validate_fixed_models(root / "fixed-models"),
            "default_shape": validate_default_shape(root / "defaults"),
            "flexible_zbl_rejected": validate_flexible_zbl_rejected(root / "flexible-zbl"),
            "ordinary_nep_io": validate_shared_nep_io(root / "ordinary-nep"),
            "parser_negative_exit_codes": validate_parser(root / "parser"),
            "response_training": validate_response_training(root / "response"),
            "training_prediction": validate_training_and_prediction(root / "correctness"),
            "rank_one_training_prediction": validate_training_and_prediction(
                root / "rank-one", spin_compress=1,
                typewise_spin_cutoff=True, torque_only=True),
            "zbl_training_prediction": validate_training_and_prediction(
                root / "zbl", enable_zbl=True),
            "magnetic_cutoff_small_cell": validate_training_and_prediction(
                root / "small-cell", all_spin_types=True, small_cell=True),
            "channel_layout_restart": validate_training_and_prediction(
                root / "restart", spin_compress=3,
                all_spin_types=True, check_restart=True),
        }
        if args.check_jit:
            report["native_compiled_checkpoint"] = validate_native_compiled_checkpoint(root / "jit")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
