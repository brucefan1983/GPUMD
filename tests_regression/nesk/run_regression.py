#!/usr/bin/env python3
"""Compare pre-change and post-change NeSK executables on identical real data."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

from output_compare import check, check_band_output, check_training_outputs, compare_outputs


FIXTURES = Path(__file__).resolve().parent / "fixtures"
DATASETS = {"diamond": (("C",), 32), "graphene": (("C",), 72),
            "silicon": (("Si",), 28), "sige": (("Si", "Ge"), 28)}
BASES = ("sp3", "sp3sstar", "sp3d5", "sp3d5sstar")


def binary_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_input(directory, name, basis, dispersion):
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(FIXTURES / (name + ".xyz"), directory / "train.xyz")
    elements, band_max = DATASETS[name]
    metadata = {"elements": elements, "band_max": band_max, "basis": basis, "structures": []}
    # Read the fixed fixture layout to check identities and reference bands in the output.
    with (directory / "train.xyz").open() as stream:
        for line in stream:
            count = int(line)
            fields = dict(token.split("=", 1) for token in shlex.split(next(stream)))
            nk, nb = int(fields["nkpoints"]), int(fields["nbands"])
            ks, es = [list(map(float, fields[key].split())) for key in ("kpoints", "bands")]
            for _ in range(count):
                next(stream)
            check(len(ks) == 3*nk and len(es) == nb*nk and nb >= band_max, "invalid fixture bands")
            metadata["structures"].append({
                "kpoints": [ks[3*i:3*i+3] for i in range(nk)],
                "reference": [es[nb*i:nb*i+band_max] for i in range(nk)]})
    (directory / "nesk.in").write_text(f"""type {len(elements)} {' '.join(elements)}
basis {basis}
cutoff 3.2 3.2 3.2
n_max 1 1
basis_size 2 2
l_max 2
neuron 4 4
epoch 2
seed 17
batch 1 3
band_range 0 {band_max}
band_curriculum 0
band_weight 0 2 0.5
band_weight 2 {band_max} 1.5
learning_rate 0.003
lr_scheduler plateau 0.5 20 3.0e-4 1.0e-6
dispersion {dispersion} 0.9
""")
    return metadata


def run_executable(executable, directory, timeout):
    log = directory / "run.log"
    try:
        result = subprocess.run([str(executable)], cwd=directory, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or b""
        log.write_text(output.decode(errors="replace") if isinstance(output, bytes) else output)
        raise AssertionError(f"timeout ({timeout}s); see {log}") from exc
    log.write_text("$ " + shlex.quote(str(executable)) + "\n" + result.stdout)
    check(result.returncode == 0, f"exit {result.returncode}; see {log}\n{result.stdout[-2500:]}")


def run_training(executable, directory, name, basis, dispersion, timeout):
    metadata = prepare_input(directory, name, basis, dispersion)
    run_executable(executable, directory, timeout)
    check_training_outputs(directory, metadata)
    check(not (directory / "nesk.restart").exists(), "unexpected restart output")


def run_prediction(executable, directory, basis, model, model_hash, timeout):
    metadata = prepare_input(directory, "sige", basis, .2)
    with (directory / "nesk.in").open("a") as stream:
        stream.write("prediction 1\n")
    copied_model = directory / "nesk.txt"
    shutil.copyfile(model, copied_model)
    check(binary_hash(copied_model) == model_hash, "prediction model differs from baseline")
    run_executable(executable, directory, timeout)
    check(binary_hash(copied_model) == model_hash, "prediction changed the model file")
    check(not (directory / "loss.out").exists(), "unexpected prediction loss output")
    check(not (directory / "nesk.restart").exists(), "unexpected prediction restart output")
    check_band_output(directory / "band_train.out", metadata)


def main():
    parser = argparse.ArgumentParser(description="Compare old/new NeSK binaries")
    parser.add_argument("--old", required=True, type=Path, help="executable before the change")
    parser.add_argument("--new", required=True, type=Path, help="executable after the change")
    parser.add_argument("--dataset", action="append", choices=tuple(DATASETS),
                        help="select training data; sige also selects fixed-model prediction")
    parser.add_argument("--basis", action="append", choices=BASES,
                        help="select bases for training and fixed-model prediction")
    parser.add_argument("--timeout", type=int, default=300, help="seconds per process")
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--workdir", type=Path, help="new directory, always retained")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0 or not all(math.isfinite(x) and x >= 0 for x in (args.atol, args.rtol)):
        parser.error("timeout must be positive; tolerances must be finite and nonnegative")
    binaries = {"old": args.old.resolve(), "new": args.new.resolve()}
    for name, path in binaries.items():
        if not path.is_file() or not os.access(path, os.X_OK):
            parser.error(f"{name} executable is missing or not executable: {path}")
    datasets = list(dict.fromkeys(args.dataset or DATASETS))
    bases = list(dict.fromkeys(args.basis or BASES))
    for name in datasets:
        path = FIXTURES / (name + ".xyz")
        if not path.is_file():
            parser.error(f"fixture missing: {path}")
    work = args.workdir.resolve() if args.workdir else Path(tempfile.mkdtemp(prefix="nesk-exe-tests-"))
    if args.workdir:
        work.mkdir(parents=True, exist_ok=False)
    print(f"Work directory: {work}", flush=True)
    report = {"mode": "old_new", "atol": args.atol, "rtol": args.rtol,
              "executables": {name: {"path": str(path), "sha256": binary_hash(path)}
                              for name, path in binaries.items()}, "cases": []}
    prediction_models = {}

    def record_result(record):
        report["cases"].append(record)
        detail = ": " + record["error"] if "error" in record else ""
        print(f"{record['status']} {record['name']}{detail}", flush=True)
        (work / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    for name in datasets:
        for basis in bases:
            for dispersion in (0, .2):
                label = f"train_{name}_{basis}_dispersion_{dispersion}"
                record = {"name": label}
                try:
                    outputs = {}
                    for version, executable in binaries.items():
                        directory = work / label / version / "train"
                        run_training(executable, directory, name, basis, dispersion, args.timeout)
                        outputs[version] = directory
                        if name == "sige" and dispersion == .2 and version == "old":
                            model = directory / "nesk.txt"
                            prediction_models[basis] = (model, binary_hash(model))
                    # Allow for GPU run-to-run variation in this case's model parameters.
                    factor = 5 if label == "train_graphene_sp3sstar_dispersion_0.2" else 1
                    record["parameter_atol"] = factor * args.atol
                    record["max_abs_difference"] = compare_outputs(
                        outputs["old"], outputs["new"], args.atol, args.rtol,
                        parameter_atol=record["parameter_atol"])
                    record["status"] = "PASS"
                except Exception as exc:
                    record.update(status="FAIL", error=str(exc))
                record_result(record)

    if "sige" in datasets:
        for basis in bases:
            label = f"predict_sige_{basis}_dispersion_0.2"
            training_case = f"train_sige_{basis}_dispersion_0.2"
            record = {"name": label, "training_case": training_case}
            if basis not in prediction_models:
                record.update(status="SKIP", error=f"baseline training failed: {training_case}")
            else:
                model, model_hash = prediction_models[basis]
                record.update(model_source=str(model), model_sha256=model_hash)
                try:
                    outputs = {}
                    for version, executable in binaries.items():
                        directory = work / label / version / "predict"
                        run_prediction(executable, directory, basis, model, model_hash, args.timeout)
                        outputs[version] = directory
                    record["max_abs_difference"] = compare_outputs(
                        outputs["old"], outputs["new"], args.atol, args.rtol, prediction=True)
                    record["status"] = "PASS"
                except Exception as exc:
                    record.update(status="FAIL", error=str(exc))
            record_result(record)

    passed = sum(c["status"] == "PASS" for c in report["cases"])
    failed = sum(c["status"] == "FAIL" for c in report["cases"])
    skipped = sum(c["status"] == "SKIP" for c in report["cases"])
    report["summary"] = {"passed": passed, "failed": failed, "skipped": skipped}
    (work / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped", flush=True)
    if failed or skipped or args.keep or args.workdir:
        print(f"Inputs, logs and summary.json retained in: {work}", flush=True)
    else:
        shutil.rmtree(work)
    return int(failed > 0 or skipped > 0)


if __name__ == "__main__":
    raise SystemExit(main())
