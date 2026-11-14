"""Frozen Spin model/oracle loaders; independent of runtime and integrators."""
import hashlib
import math
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "nep_spin"

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

def load_fixture(name="c3l2s2_fege"):
    model = (FIXTURE / (name + ".nep")).read_text()
    if name == "c3l2s2_fege":
        import json
        source = json.loads((FIXTURE / "c3l2s2.json").read_text())["cases"]["large"]
        case = {key: source[key] for key in ("cell", "positions", "spins")}
        case["types"] = ["Fe" if t == "Fe" else "Ge" for t in source["types"]]
        return model, {"cases": {"noncollinear": case}}
    import json
    reference = json.loads((FIXTURE / (name + ".json")).read_text())
    assert hashlib.sha256(model.encode()).hexdigest() == reference["model_sha256"]
    return model, reference
