"""Frozen Spin model/oracle loaders; independent of runtime and integrators."""
import hashlib
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "nep_spin3"

def selective_type_model():
    """Return the production C=2 fixture with Fe spin DOFs and geometric environments from all types."""
    return (FIXTURE / "nep_spin_o3c2_uniform.nep").read_text()

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
        "o3c2_uniform": ("nep_spin_o3c2_uniform.nep", "spin3_o3c2_uniform_oracle.txt"),
        "o3c2_typewise": ("nep_spin_o3c2_uniform.nep", "spin3_o3c2_typewise_oracle.txt"),
        "o3c3_uniform": ("nep_spin_o3c3_uniform.nep", "spin3_o3c3_uniform_oracle.txt"),
    }
    if name not in fixtures:
        raise ValueError(f"unknown fixture: {name}")
    model_name, oracle_name = fixtures[name]
    model = (FIXTURE / model_name).read_text()
    if name == "o3c2_typewise":
        # Identical parameters; the frozen oracle hash also checks this header edit.
        model = model.replace("spin_cutoff 6.0\n", "spin_cutoff 4.0 6.0\n", 1)
    oracle = read_text_oracle(FIXTURE / oracle_name, model)
    return model, oracle
