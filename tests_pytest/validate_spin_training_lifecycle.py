#!/usr/bin/env python3
"""Exercise Spin checkpoint/export contracts through the real NEP CLI."""
import json
import math
import re
import tempfile
from pathlib import Path

import validate_main_nep_spin3_training as base

CONFIG = base.NEP_IN + "nep_compile 0\nseed 20261021\n"
NO_MASKS = re.sub(r"^spin_type.*\n", "", CONFIG, flags=re.M)


def execute(directory, config, xyz=base.TRAIN_XYZ, model=None, restart=None, error=None, test_xyz=None):
    base.write_case(directory, config, xyz)
    if test_xyz is not None:
        (directory / "test.xyz").write_text(test_xyz)
    if model is not None:
        (directory / "nep.txt").write_text(model)
    if restart is not None:
        (directory / "nep.restart").write_text(restart)
    result = base.run(base.NEP, directory)
    output = result.stdout + result.stderr
    (directory / "run.log").write_text(output)
    if error:
        assert result.returncode and error in output, output
    elif result.returncode:
        raise AssertionError(output)
    return output


def predict(directory, model, xyz=base.TRAIN_XYZ, config=NO_MASKS):
    execute(directory, config + "prediction 1\n", xyz, model)
    return {name: base.columns(directory / f"{name}_train.out", width)
            for name, width in (("energy", 1), ("force", 3), ("virial", 6), ("mforce", 3))}


def compare(left, right):
    errors = {}
    for name in left:
        error = base.maximum_error(left[name], right[name])
        assert math.isfinite(error) and error <= base.parity_tolerance(left[name], right[name]), (name, error)
        errors[name] = error
    return errors


def metadata(model):
    lines = model.splitlines()
    stop = next(i for i, line in enumerate(lines) if line.startswith('@ '))
    return dict((line.split()[0], line.split()[1:]) for line in lines[:stop]
                if line.startswith(('spin_', 'energy_reference ')))




def strip_labels(xyz, frames):
    lines = xyz.splitlines()
    for frame in frames:
        start = frame * 6
        lines[start+1] = lines[start+1].replace(":mforce:R:3", "")
        for a in range(start+2, start+6):
            lines[a] = " ".join(lines[a].split()[:-3])
    return "\n".join(lines) + "\n"


def validate_inactive_input(root, config, model, restart):
    root.mkdir()
    raw = base.TRAIN_XYZ.splitlines()
    for i, line in enumerate(raw):
        fields = line.split()
        if fields and fields[0] == "Ge":
            fields[4:7] = ["0.125", "-0.25", "0.375"]
            fields[7:13] = ["0.3", "-0.2", "1e-100", "123", "-456", "789"]
            raw[i] = " ".join(fields)
    raw = "\n".join(raw) + "\n"
    cleared = []
    for line in raw.splitlines():
        fields = line.split()
        if fields and fields[0] == "Ge":
            fields[7:13] = ["0"] * 6
            line = " ".join(fields)
        cleared.append(line)
    cleared = "\n".join(cleared) + "\n"
    for mode, extra, saved, state in (
            ("train", "", None, None),
            ("predict", "prediction 1\n", model, None),
            ("restart", "", model, restart),
            ("import", "import_q_scaler 1\n", model, None)):
        dirs = [root / (mode + suffix) for suffix in ("_raw", "_cleared")]
        for directory, xyz in zip(dirs, (raw, cleared)):
            execute(directory, config + extra, xyz, model=saved, restart=state, test_xyz=xyz)
            assert (directory / "train.xyz").read_text() == xyz
            assert (directory / "test.xyz").read_text() == xyz
        part = "train" if mode == "predict" else "test"
        outputs = [f"{kind}_{part}.out" for kind in ("energy", "force", "virial", "mforce")]
        if mode != "predict":
            outputs += ["nep.txt", "nep.restart", "loss.out"]
        for filename in outputs:
            assert (dirs[0] / filename).read_bytes() == (dirs[1] / filename).read_bytes(), (mode, filename)
        if mode == "predict":
            forces = base.columns(dirs[0] / "force_train.out", 6)
            magnetic = base.columns(dirs[0] / "mforce_train.out", 6)
            atoms = [line.split() for line in raw.splitlines() if line.startswith(("Fe ", "Ge "))]
            for i, fields in enumerate(atoms):
                assert forces[i][3:] == [float(x) for x in fields[4:7]]
                if fields[0] == "Ge":
                    assert magnetic[i] == [0.0] * 6
            energies = base.columns(dirs[0] / "energy_train.out", 2)
            for row, target in zip(energies, (-1.0, -0.5, -1.5)):
                assert abs(row[1] - target) < 1e-6, (row, target)
    return "raw and zeroed inputs agree; energy/atomic-force labels and input files preserved"


def validate(root):
    report = {}
    for name, config in (("fe_only", CONFIG), ("all_types", NO_MASKS)):
        fresh = root / name
        execute(fresh, config)
        model = (fresh / "nep.txt").read_text()
        assert "\nmodel nep_spin\n" in model and "\nspin_version 1\n" in model
        assert "\nspin_mode " not in model
        old = predict(root / (name + "_before"), model)
        # Freeze the exported elite in the shared reference-aware restart format.
        lines = model.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("@ parameters "))
        count = int(lines[start].split()[2])
        reference = metadata(model)["energy_reference"]
        frozen = ("nep_restart\nenergy_reference " + str(len(reference)) + " " + " ".join(reference)
                  + f"\nparameters {count}\n"
                  + "".join(f"{x} 0\n" for x in lines[start+1:start+1+count]))
        resumed = root / (name + "_resume")
        execute(resumed, NO_MASKS.replace("generation 1", "generation 2"), model=model, restart=frozen)
        snapshots = list(resumed.glob("nep_gen*.txt"))
        assert len(snapshots) == 2, list(resumed.iterdir())
        saved = (resumed / "nep.txt").read_text()
        assert metadata(saved) == metadata(model), (metadata(saved), metadata(model))
        after = predict(root / (name + "_after"), saved)
        report[name] = compare(old, after)
        imported = root / (name + "_import")
        execute(imported, NO_MASKS + "import_q_scaler 1\n", model=model)
        assert metadata((imported / "nep.txt").read_text()) == metadata(model)
        # An exported import model must also be accepted with omitted input masks.
        predict(root / (name + "_import_reload"), (imported / "nep.txt").read_text())
        if name == "fe_only":
            execute(root / "unsupported_version", NO_MASKS + "prediction 1\n",
                    model=model.replace("spin_version 1\n", "spin_version 2\n"),
                    error="Unsupported spin_version")
            execute(root / "conflicting_dof", NO_MASKS + "spin_type Ge\n",
                    model=model, restart=frozen, error="spin_type in nep.in conflicts")
            report["inactive_input"] = validate_inactive_input(root / "inactive_input", config, model, frozen)
            execute(root / "renamed_type", config.replace("spin_type Fe", "spin_dof_type Fe"),
                    error="spin_dof_type")
            execute(root / "removed_env", config + "spin_env_type Fe Ge\n",
                    error="spin_env_type")

    model = (root / "fe_only" / "nep.txt").read_text()
    reference = predict(root / "labeled", model)
    for name, frames in (("unlabeled", (0, 1, 2)), ("mixed", (0, 2))):
        xyz = strip_labels(base.TRAIN_XYZ, frames)
        directory = root / name
        report[name] = compare(reference, predict(directory, model, xyz))
        rows = base.columns(directory / "mforce_train.out", 6)
        assert len(rows) == 12
        for frame in frames:
            assert all(row[3:] == [0, 0, 0] for row in rows[frame*4:frame*4+4])
        # Training diagnostics continue to contain only supervised configurations.
        trained = root / (name + "_training")
        execute(trained, CONFIG, xyz, test_xyz=xyz)
        assert len((trained / "mforce_test.out").read_text().splitlines()) == 4 * (3-len(frames))

    # Small periodic cells force image neighbors. pbc strings are deliberately ignored.
    small = base.TRAIN_XYZ.replace('24 0 0 0 24 0 0 0 24', '4 0 0 0 4 0 0 0 4')
    periodic = predict(root / "pbc_default", model, small)
    for flag in ("F F F", "T T F"):
        xyz = small.replace("Properties=", f'pbc="{flag}" Properties=')
        report[flag] = compare(periodic, predict(root / flag.replace(" ", ""), model, xyz))
    report["ediff"] = validate_ediff(root / "ediff")
    return report


def validate_ediff(directory):
    # Interleaved groups force grouped batching to reorder frames, while spins
    # distinguish equal geometries. Distinct targets detect label-map mistakes.
    frames = []
    for index, name in enumerate(("a", "c", "b", "d")):
        frame = base.TRAIN_XYZ.splitlines()[:6]
        frame[1] = frame[1].replace("energy=-4.0", f"energy={-4+index}.0") + f" name={name}"
        fields = frame[2].split()
        fields[7:10] = [str(math.cos(index)), str(math.sin(index)), "0"]
        fields[10:13] = [str(index+1), str(index+2), str(index+3)]
        frame[2] = " ".join(fields)
        frames.append(frame)
    xyz = "\n".join(line for frame in frames for line in frame) + "\n"
    base.write_case(directory, CONFIG.replace("batch 3", "batch 2").replace("generation 1", "generation 2") + "lambda_d 0.1\n", xyz)
    (directory / "ediff.in").write_text("a - b\nc - d\n")
    (directory / "test.xyz").write_text(xyz)
    result = base.run(base.NEP, directory)
    assert result.returncode == 0, result.stdout + result.stderr
    loss = [[float(x) for x in line.split()] for line in
            (directory / "loss.out").read_text().splitlines()
            if line.strip() and not line.startswith("#")][-1]
    energy = base.columns(directory / "energy_test.out", 2)
    residuals = [4 * ((energy[a][0]-energy[b][0]) - (energy[a][1]-energy[b][1]))
                 for a, b in ((0, 2), (1, 3))]
    expected = math.sqrt(sum(x*x for x in residuals)/2)
    assert abs(loss[-1] - expected) < 2e-4, (loss, expected)
    targets = base.columns(directory / "mforce_test.out", 6)
    # Reconstruct the final training batch loss independently from predictions
    # in file order. Both ediff and magnetic errors must identify the same pair.
    pair_errors = []
    for a, b in ((0, 2), (1, 3)):
        error2 = sum((row[c] - row[c+3])**2
                     for i in (a, b) for row in targets[4*i:4*i+2] for c in range(3))
        pair_errors.append(math.sqrt(error2/12))
    assert any(abs(loss[7]-m) < 2e-4 and abs(loss[14]-abs(e)) < 2e-4
               for m, e in zip(pair_errors, residuals)), (loss, pair_errors, residuals)
    assert [targets[4*i][3:] for i in range(4)] == [[i+1, i+2, i+3] for i in range(4)]
    saved = (directory / "nep.txt").read_text()
    prediction = predict(directory / "prediction", saved, xyz)
    compare({"mforce": [row[:3] for row in targets]}, {"mforce": prediction["mforce"]})
    # No special acceptance of genuinely identical geometry+spin pairs.
    frames[2][2] = frames[0][2]
    (directory / "train.xyz").write_text("\n".join(line for frame in frames for line in frame) + "\n")
    result = base.run(base.NEP, directory)
    assert result.returncode and "same geometry and spins" in result.stdout + result.stderr
    return {"test_rmse": loss[-1], "independent_rmse": expected, "original_label_order": "passed"}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="spin-lifecycle-") as temporary:
        print(json.dumps(validate(Path(temporary)), indent=2, sort_keys=True))
