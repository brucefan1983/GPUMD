"""Spin correctors retain Action increments without sampling them twice."""
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from spin_analytic_model import analytic_model, structure
from validate_spin_multigpu import frame

GPUMD = Path(os.environ.get("GPUMD_COMMAND", Path(__file__).parents[1] / "src/gpumd")).resolve()


@pytest.mark.parametrize("ensemble", [
    "nve_sib alpha 0 gamma 3 stemp -1 seed 731",
    "nve_glsd damping .01 gamma 3 stemp -1 seed 731",
])
@pytest.mark.parametrize("length", [8, 20], ids=["small-box", "large-box"])
def test_corrector_preserves_force_modifiers(tmp_path, ensemble, length):
    if not GPUMD.is_file():
        pytest.skip("GPUMD executable is not built")
    lines = structure().splitlines()
    lines[1] = lines[1].replace("20 0 0 0 20 0 0 0 20", f"{length} 0 0 0 {length} 0 0 0 {length}")
    lines[1] = lines[1].replace("spin_vel:R:3", "spin_vel:R:3:group:I:1")
    lines[2:] = [line + " 0" for line in lines[2:]]
    model = "\n".join(lines) + "\n"

    def run(name, integrator, dt):
        case = tmp_path / name
        case.mkdir()
        (case / "nep.txt").write_text(analytic_model(False))
        (case / "model.xyz").write_text(model)
        (case / "run.in").write_text(
            "potential nep.txt\n"
            f"ensemble {integrator}\ntime_step {dt}\n"
            "add_random_force .05\nadd_force 0 0 .02 -.01 .03\n"
            "add_mfield 0 0 4 -2 3\nadd_mfield region 0 1 0 1 0 1 8 -5 28\n"
            "dump_xyz 3 state.xyz precision double force potential spin mforce\nrun 3\n"
            "ensemble nve\ntime_step 0\n"
            "dump_xyz 1 cleared.xyz precision double force potential spin mforce\nrun 1\n")
        result = subprocess.run([str(GPUMD)], cwd=case, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        state, cleared = frame(case / "state.xyz"), frame(case / "cleared.xyz")
        assert all(np.isfinite(value).all() for data in (state, cleared) for value in data.values())
        for key in ("force", "mforce", "potential"):
            np.testing.assert_allclose(cleared[key], 0, rtol=0, atol=1e-14)
        return state

    # NVE at dt=0 has the same three Action samples but no Spin corrector.
    reference = run("reference", "nve", 0)
    actual = run("spin", ensemble, .1)
    np.testing.assert_allclose(actual["force"], reference["force"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual["force"].mean(axis=0), [.02, -.01, .03], rtol=0, atol=1e-12)
    assert np.max(np.abs(actual["force"] - [.02, -.01, .03])) > 1e-3
    field = np.array([12., -7., 31.]) * 5.7883818060e-5
    np.testing.assert_allclose(actual["mforce"], np.tile(field, (4, 1)), rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual["potential"].ravel(), -actual["spin"] @ field, rtol=0, atol=1e-12)
    assert np.max(np.abs(actual["spin"] - reference["spin"])) > 1e-12
