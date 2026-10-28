#!/usr/bin/env python3
"""Recompute frozen cases with fixed external FP64/autograd reference sources."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

import numpy as np
import torch

TORCHNEP_REV = "3e2a5b3b31803f5cd8733d1fe67180cc7b144ad5"
ADAPTER_REV = "0ab9f452330e43a21ba76c9969d5a6859a6de31f"
GENERATOR_SHA = "5be4782db6df7d25d630866c3d666be29aea0f5bed9936f7e65dfb76b1abad59"
FIXTURES = Path(__file__).parent / "fixtures/nep_spin3"


def git(repository, *args):
    return subprocess.check_output(["git", "-C", str(repository), *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torchnep", required=True, type=Path, help="TorchNEP Git checkout")
    parser.add_argument("--adapters", required=True, type=Path, help="NEPAdapters Git checkout")
    parser.add_argument("--output", required=True, type=Path, help="New directory outside fixtures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"torchnep": TORCHNEP_REV, "adapters": ADAPTER_REV,
              "torch": torch.__version__, "numpy": np.__version__, "max_abs": {}}
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="spin-fp64-reference-") as temporary:
        root = Path(temporary)
        # Export committed sources, never import a dirty reference working tree.
        archive = git(args.torchnep, "archive", TORCHNEP_REV, "torchnep")
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(root)
        generator = git(args.adapters, "show", ADAPTER_REV + ":tests/generate_spin2_oc_oracle.py")
        assert hashlib.sha256(generator).hexdigest() == GENERATOR_SHA
        path = root / "reference_generator.py"
        path.write_bytes(generator)
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("reference_generator", path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        for name in ("o3c2_uniform", "o3c2_typewise", "o3c3_uniform"):
            model = root / (name + ".nep")
            model.write_bytes(git(args.adapters, "show", ADAPTER_REV +
                                 f":tests/fixtures/nep4_spin3_{name}.nep"))
            calculator = reference.NEPCalculator(str(model), dtype=torch.float64)
            lines = (FIXTURES / f"spin3_{name}_oracle.txt").read_text().splitlines()
            output = args.output / f"spin3_{name}_oracle.txt"
            errors = {}
            with output.open("w") as handle:
                handle.write("\n".join(lines[:4]) + "\n")
                for line in lines[4:]:
                    fields = line.split()
                    key = fields[0]
                    if key == "case":
                        case = {}
                        heading = line
                    elif key == "types":
                        species = fields[1:]
                    elif key == "end":
                        prediction = calculator.compute(
                            species, case["positions"].reshape(-1, 3),
                            case["cell"].reshape(3, 3), case["spins"].reshape(-1, 3),
                            compute_descriptor=True)
                        handle.write(heading + "\ntypes " + " ".join(species) + "\n")
                        for field, expected in case.items():
                            result_key = "energy" if field == "potential" else field
                            values = (expected if field in ("cell", "positions", "spins") else
                                      prediction[result_key].detach().cpu().numpy().reshape(-1))
                            assert values.shape == expected.shape, (name, field)
                            assert np.isfinite(expected).all() and np.isfinite(values).all()
                            error = float(np.max(np.abs(values - expected)))
                            errors[field] = max(errors.get(field, 0.0), error)
                            reference.write_array(handle, field, values)
                        handle.write("end\n")
                    else:
                        case[key] = np.array([float(v) for v in fields[2:]], dtype=np.float64)
                        assert len(case[key]) == int(fields[1]), (name, key)
            report["max_abs"][name] = errors
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert all(error <= 1e-10 for case in report["max_abs"].values() for error in case.values())


if __name__ == "__main__":
    main()
