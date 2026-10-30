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
    shutil.copy(FIXTURE / "nep_spin_o3c2_uniform.nep", case / "nep.txt")
    (case / "model.xyz").write_text(model_text or base_model())
    (case / "run.in").write_text(run_input)
    result = subprocess.run(
        [str(GPUMD)], cwd=case, capture_output=True, text=True, check=False)
    return case, result
