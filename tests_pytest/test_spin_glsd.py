"""GLSD conservation and self-consistent midpoint regressions."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from spin_analytic_model import REFERENCE_MODEL, analytic_model, structure
from validate_spin_glsd_runtime import damping
from validate_spin_multigpu import frame

GPUMD = Path(os.environ.get("GPUMD_COMMAND", Path(__file__).parents[1] / "src/gpumd")).resolve()
INITIAL = np.array([[1., .2, .3], [.4, -.3, .7], [0., 0., 0.], [.6, .1, -.4]])
FIELD = np.array([0., 0., 10000.]) * 5.7883818060e-5


def run(root, name, spins, ensemble, steps, dt, onsite=True, field=True, model=None):
    if not GPUMD.is_file():
        pytest.skip("GPUMD executable is not built")
    case = root / name
    case.mkdir()
    (case / "nep.txt").write_text(model if model is not None else analytic_model(onsite))
    (case / "model.xyz").write_text(structure(spins.tolist()))
    (case / "run.in").write_text(
        "potential nep.txt\n" +
        ("add_mfield region 0 1 0 1 0 1 0 0 10000\n" if field else "") +
        f"ensemble {ensemble}\ntime_step {dt}\n"
        "dump_xyz 1 state.xyz precision double spin mforce potential\n"
        f"run {steps}\n")
    result = subprocess.run([str(GPUMD)], cwd=case, capture_output=True, text=True)
    (case / "run.log").write_text(result.stdout + result.stderr)
    return case, result


def test_glsd_pure_precession_preserves_norm_and_radial_energy(tmp_path):
    case, result = run(tmp_path, "precession", INITIAL, "nve_glsd damping 0 gamma 3 stemp -1 midpoint_tol 1e-8 midpoint_max_iter 100", 200, 50)
    assert result.returncode == 0, result.stdout + result.stderr
    state = frame(case / "state.xyz")
    norm_error = float(np.max(np.abs(np.linalg.norm(state["spin"], axis=1) - np.linalg.norm(INITIAL, axis=1))))
    initial_energy = np.sum(np.tanh(.1 * np.sum(INITIAL**2, axis=1)) - INITIAL @ FIELD)
    energy_error = float(abs(np.sum(state["potential"]) - initial_energy))
    errors = {"norm_error": norm_error, "energy_error_eV": energy_error}
    (tmp_path / "conservation.json").write_text(json.dumps(errors, indent=2))
    print(errors)
    assert norm_error < 1e-11, errors
    # The ANN energy/field backend uses float descriptors.
    assert energy_error < 5e-7, errors


@pytest.mark.parametrize("mobility", [0., 2.])
def test_glsd_satisfies_nonlinear_midpoint_equation(tmp_path, mobility):
    dt, gamma = 20., 3.
    case, result = run(tmp_path, "step", INITIAL, f"nve_glsd {damping(mobility, gamma)} gamma {gamma} stemp -1 midpoint_tol 1e-8 midpoint_max_iter 100", 1, dt)
    assert result.returncode == 0, result.stdout + result.stderr
    state = frame(case / "state.xyz")
    midpoint = .5 * (INITIAL + state["spin"])
    middle, result = run(tmp_path, "midpoint", midpoint, "nve", 1, 0)
    assert result.returncode == 0, result.stdout + result.stderr
    field = frame(middle / "state.xyz")["mforce"]
    residual = state["spin"] - INITIAL - dt / 1000 * (gamma * np.cross(field, midpoint) + mobility * field)
    np.testing.assert_allclose(residual, 0, rtol=0, atol=2e-8)
    endpoint, result = run(tmp_path, "endpoint", state["spin"], "nve", 1, 0)
    assert result.returncode == 0, result.stdout + result.stderr
    np.testing.assert_allclose(state["mforce"], frame(endpoint / "state.xyz")["mforce"], rtol=0, atol=1e-12)


def test_glsd_dissipative_midpoint_converges_at_second_order(tmp_path):
    # Independent scalar ODE oracle: E=tanh(.1 r^2), dr/dt=-.4 r sech^2(.1 r^2).
    radii = np.linalg.norm(INITIAL, axis=1)
    h = 1e-4
    def drift(r):
        return -.4 * r / np.cosh(.1 * r*r)**2
    for _ in range(10000):
        a = drift(radii)
        b = drift(radii + .5*h*a)
        c = drift(radii + .5*h*b)
        d = drift(radii + h*c)
        radii += h/6 * (a + 2*b + 2*c + d)
    errors = []
    for dt in (40, 20, 10):
        case, result = run(tmp_path, str(dt), INITIAL, f"nve_glsd {damping(2, 3)} gamma 3 stemp -1 midpoint_max_iter 100", 1000//dt, dt, field=False)
        assert result.returncode == 0, result.stdout + result.stderr
        state = frame(case / "state.xyz")
        errors.append(np.max(np.abs(np.linalg.norm(state["spin"], axis=1) - radii)))
        assert np.sum(state["potential"]) < np.sum(np.tanh(.1 * np.sum(INITIAL**2, axis=1)))
    assert all(3.5 < x/y < 4.5 for x, y in zip(errors, errors[1:])), errors


def test_glsd_rejects_nonfinite_midpoint(tmp_path):
    _, result = run(tmp_path, "nonfinite", INITIAL, "nve_glsd damping 0 gamma 1e200 stemp -1", 1, 1, onsite=False)
    assert result.returncode != 0
    assert "GLSD midpoint predictor is non-finite" in result.stdout + result.stderr


def test_glsd_default_four_corrections_accepts_last_finite_update(tmp_path):
    options = f"nve_glsd {damping(2, 3)} gamma 3 stemp -1 midpoint_tol 1e-8"
    default, result = run(tmp_path, "default", INITIAL, options, 1, 200, field=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "midpoint_max_iter = 4." in result.stdout
    assert "mean 4.000, max 4." in result.stdout
    assert "steps accepted at iteration cap before convergence: 1 of 1." in result.stdout
    explicit, result = run(tmp_path, "explicit", INITIAL, options + " midpoint_max_iter 4", 1, 200, field=False)
    assert result.returncode == 0, result.stdout + result.stderr
    state = frame(default / "state.xyz")
    assert np.isfinite(state["spin"]).all()
    np.testing.assert_array_equal(state["spin"], frame(explicit / "state.xyz")["spin"])


def test_glsd_nep_energy_error_decreases_with_timestep(tmp_path):
    model = REFERENCE_MODEL.read_text().replace("spin_type Fe\n", "spin_type Fe Ge\n")
    initial, result = run(tmp_path, "initial", INITIAL, "nve", 1, 0, model=model)
    assert result.returncode == 0, result.stdout + result.stderr
    energy = np.sum(frame(initial / "state.xyz")["potential"])
    errors = []
    for dt in (2., 1., .5):
        case, result = run(tmp_path, str(dt), INITIAL, "nve_glsd damping 0 gamma 300 stemp -1 midpoint_tol 1e-8 midpoint_max_iter 100", round(200/dt), dt, model=model)
        assert result.returncode == 0, result.stdout + result.stderr
        lines = (case / "state.xyz").read_text().splitlines()
        states = np.array([[list(map(float, row.split()[1:])) for row in lines[i+2:i+6]]
                           for i in range(0, len(lines), 6)])
        norms = np.linalg.norm(states[:, :, 3:6], axis=2)
        assert np.max(np.abs(norms - np.linalg.norm(INITIAL, axis=1))) < 1e-11
        errors.append(float(np.max(np.abs(states[:, :, -1].sum(axis=1) - energy))))
    (tmp_path / "energy-convergence.json").write_text(json.dumps({"dt_fs": [2., 1., .5], "max_energy_error_eV": errors}, indent=2))
    print("NEP energy errors:", errors)
    assert errors[0] > 2 * errors[1] and errors[1] > 2 * errors[2], errors


def test_glsd_configurable_iteration_limit_and_summary(tmp_path):
    options = f"nve_glsd {damping(2, 3)} gamma 3 stemp -1 midpoint_tol 1e-8"
    _, limited = run(tmp_path, "limited", INITIAL, options + " midpoint_max_iter 1", 1, 40, field=False)
    assert limited.returncode == 0, limited.stdout + limited.stderr
    assert "steps accepted at iteration cap before convergence: 1 of 1." in limited.stdout
    _, loose = run(tmp_path, "loose", INITIAL, options.replace("1e-8", "1e-2") + " midpoint_max_iter 1", 1, 40, field=False)
    assert loose.returncode == 0, loose.stdout + loose.stderr
    assert "steps accepted at iteration cap before convergence: 0 of 1." in loose.stdout
    _, converged = run(tmp_path, "converged", INITIAL, options + " midpoint_max_iter 20", 1, 40, field=False)
    assert converged.returncode == 0, converged.stdout + converged.stderr
    assert "GLSD midpoint iterations per step:" in converged.stdout
    assert "steps accepted at iteration cap before convergence: 0 of 1." in converged.stdout
    # A constant field is solved in one correction, regardless of the cap.
    _, constant = run(tmp_path, "constant", INITIAL, options + " midpoint_max_iter 1", 3, 40, onsite=False)
    assert constant.returncode == 0, constant.stdout + constant.stderr
    assert "mean 1.000, max 1" in constant.stdout


@pytest.mark.parametrize("options", [
    "midpoint_tol 0", "midpoint_tol -1", "midpoint_tol nan", "midpoint_tol inf",
    "midpoint_tol 1e-8 midpoint_tol 1e-8", "midpoint_tol",
    "midpoint_max_iter 0", "midpoint_max_iter -1", "midpoint_max_iter 1.5",
    "midpoint_max_iter 2 midpoint_max_iter 2", "midpoint_max_iter",
])
def test_glsd_rejects_invalid_midpoint_controls(tmp_path, options):
    _, result = run(tmp_path, "invalid", INITIAL, "nve_glsd damping 0 " + options, 1, .1)
    assert result.returncode != 0
    assert "Input Error" in result.stdout + result.stderr


@pytest.mark.parametrize("ensemble", [
    "nvt_glsd 300 300 100 lattice off midpoint_tol 1e-8 midpoint_max_iter 10 damping 0",
    "npt_glsd temp 300 300 iso 0 0 midpoint_max_iter 10 midpoint_tol 1e-8 damping 0",
])
def test_glsd_midpoint_controls_in_thermal_ensembles(tmp_path, ensemble):
    _, result = run(tmp_path, "thermal", INITIAL, ensemble, 1, .01, onsite=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("options", ["midpoint_tol 1e-8", "midpoint_max_iter 10"])
def test_sib_rejects_glsd_midpoint_controls(tmp_path, options):
    _, result = run(tmp_path, "sib", INITIAL, "nve_sib alpha 0 " + options, 1, .1)
    assert result.returncode != 0
    assert "Input Error" in result.stdout + result.stderr


def test_glsd_coupled_timestep_convergence(tmp_path):
    if not GPUMD.is_file():
        pytest.skip("GPUMD executable is not built")
    from validate_spin_glsd_coupled import validate
    validate(tmp_path, GPUMD)


def test_glsd_accumulated_periodic_images(tmp_path):
    if not GPUMD.is_file():
        pytest.skip("GPUMD executable is not built")
    from validate_spin_glsd_images import validate
    validate(tmp_path, GPUMD)
