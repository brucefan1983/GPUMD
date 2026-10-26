"""Fixed-spin model evaluation helpers."""
import os, shutil, subprocess
from pathlib import Path
from spin_fixtures import FIXTURE
ROOT = Path(__file__).resolve().parents[1]
GPUMD = Path(os.environ.get("GPUMD_COMMAND", ROOT / "src/gpumd"))

def base_model():
    lines = (FIXTURE / "model_large_box.xyz").read_text().splitlines()
    lines[1] = lines[1].replace(
        "Properties=species:S:1:pos:R:3:spin:R:3",
        "Properties=species:S:1:pos:R:3:vel:R:3:spin:R:3")
    for index in range(2, len(lines)):
        fields = lines[index].split()
        lines[index] = " ".join(fields[:4] + ["0", "0", "0"] + fields[4:])
    return "\n".join(lines) + "\n"

def run_case(root, name, run_input, model_text=None):
    case = root / name
    case.mkdir()
    shutil.copy(FIXTURE / "nep4_spin3_o3c2_uniform.nep", case / "nep.txt")
    (case / "model.xyz").write_text(model_text or base_model())
    (case / "run.in").write_text(run_input)
    result = subprocess.run(
        [str(GPUMD)], cwd=case, capture_output=True, text=True, check=False)
    return case, result

def read_last_frame(path):
    lines = path.read_text().splitlines()
    number_of_atoms = int(lines[0])
    values = [
        [float(value) for value in line.split()[1:]]
        for line in lines[-number_of_atoms:]
    ]
    return {
        "spin": [row[4:7] for row in values],
        "mforce": [row[7:10] for row in values],
    }

def evaluate(root, name, model_text):
    case, result = run_case(
        root,
        name,
        "potential nep.txt\n"
        "ensemble nve\n"
        "time_step 0\n"
        "dump_xyz 1 state.xyz mass spin mforce\n"
        "run 1\n",
        model_text)
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)
    return read_last_frame(case / "state.xyz")
