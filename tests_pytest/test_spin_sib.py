"""Semi-implicit B spin-integrator regression tests."""

import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


FIXTURE = Path(__file__).parent / "fixtures" / "nep_spin3"
GPUMD = Path(os.environ.get("GPUMD_COMMAND", Path(__file__).parents[1] / "src" / "gpumd"))
BOLTZMANN = 8.617343e-5
MASK64 = 0xFFFFFFFFFFFFFFFF


def _base_model():
    lines = (FIXTURE / "model_large_box.xyz").read_text().splitlines()
    lines[1] = lines[1].replace(
        "Properties=species:S:1:pos:R:3:spin:R:3",
        "Properties=species:S:1:pos:R:3:vel:R:3:spin:R:3")
    for index in range(2, len(lines)):
        fields = lines[index].split()
        lines[index] = " ".join(fields[:4] + ["0", "0", "0"] + fields[4:])
    return "\n".join(lines) + "\n"


def _replace_spins(model_text, spins):
    lines = model_text.splitlines()
    for row, spin in enumerate(spins, start=2):
        fields = lines[row].split()
        fields[-3:] = [f"{value:.17g}" for value in spin]
        lines[row] = " ".join(fields)
    return "\n".join(lines) + "\n"


def _run(case_dir, run_input, model_text=None):
    case_dir.mkdir()
    shutil.copy(FIXTURE / "nep4_spin3_o3c2_uniform.nep", case_dir / "nep.txt")
    (case_dir / "model.xyz").write_text(model_text or _base_model())
    (case_dir / "run.in").write_text(run_input)
    return subprocess.run(
        [str(GPUMD)], cwd=case_dir,
        capture_output=True, text=True, check=False)


def _read_last_frame(path):
    lines = path.read_text().splitlines()
    number_of_atoms = int(lines[0])
    values = np.array([
        [float(value) for value in line.split()[1:]]
        for line in lines[-number_of_atoms:]
    ])
    return {"spin": values[:, 4:7], "mforce": values[:, 7:10]}


def _evaluate(case_dir, model_text):
    result = _run(
        case_dir,
        "potential nep.txt\n"
        "ensemble nve\n"
        "time_step 0\n"
        "dump_xyz 1 state.xyz mass spin mforce\n"
        "run 1\n",
        model_text)
    assert result.returncode == 0, result.stdout + result.stderr
    return _read_last_frame(case_dir / "state.xyz")


def _cross(left, right):
    return np.cross(left, right)


def _cayley(direction, omega):
    half = 0.5 * omega
    half_squared = np.sum(half * half, axis=1)[:, None]
    half_dot = np.sum(half * direction, axis=1)[:, None]
    return (
        (1.0 - half_squared) * direction
        + 2.0 * _cross(half, direction)
        + 2.0 * half * half_dot
    ) / (1.0 + half_squared)


def _splitmix64(value):
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def _sib_gaussian(seed, atom_index, step, component):
    """Replay the counter-based Gaussian of the SIB kernel."""
    state = seed
    state ^= (atom_index * 0xD1B54A32D192ED03) & MASK64
    state ^= (step * 0x9E3779B97F4A7C15) & MASK64
    state ^= (component * 0x94D049BB133111EB) & MASK64

    def uniform_01():
        nonlocal state
        state = _splitmix64(state)
        return (state >> 11) * (1.0 / 9007199254740992.0)

    u1 = 0.0
    while u1 <= 0.0:
        u1 = uniform_01()
    u2 = uniform_01()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def test_sib_one_step_matches_two_field_cayley_oracle(tmp_path):
    model = _base_model()
    initial = _evaluate(tmp_path / "initial", model)
    magnitude = np.linalg.norm(initial["spin"], axis=1)[:, None]
    direction = initial["spin"] / magnitude
    alpha = 0.17
    gamma = 1200.0
    time_step_fs = 0.1
    drift = gamma * time_step_fs / 1000.0 / (1.0 + alpha * alpha)

    predictor_omega = drift * (
        initial["mforce"] + alpha * _cross(direction, initial["mforce"]))
    predictor = _cayley(direction, predictor_omega)
    midpoint_direction = 0.5 * (direction + predictor)
    midpoint_spin = magnitude * midpoint_direction
    midpoint = _evaluate(
        tmp_path / "midpoint", _replace_spins(model, midpoint_spin))

    corrector_omega = drift * (
        midpoint["mforce"]
        + alpha * _cross(midpoint_direction, midpoint["mforce"]))
    expected_spin = magnitude * _cayley(direction, corrector_omega)

    result = _run(
        tmp_path / "sib",
        "potential nep.txt\n"
        "ensemble nve_sib "
        f"alpha {alpha} gamma {gamma} stemp -1 seed 2468\n"
        f"time_step {time_step_fs}\n"
        "dump_xyz 1 state.xyz mass spin mforce\n"
        "run 1\n",
        model)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = _read_last_frame(tmp_path / "sib" / "state.xyz")
    np.testing.assert_allclose(actual["spin"], expected_spin, rtol=0.0, atol=3.0e-7)

    # The public output must be the corrected endpoint state, not the field
    # left over from the arithmetic midpoint evaluation.
    endpoint = _evaluate(
        tmp_path / "endpoint", _replace_spins(model, actual["spin"]))
    np.testing.assert_allclose(
        actual["mforce"], endpoint["mforce"], rtol=0.0, atol=3.0e-6)
    np.testing.assert_allclose(
        np.linalg.norm(actual["spin"], axis=1), magnitude[:, 0],
        rtol=0.0, atol=3.0e-7)


@pytest.mark.parametrize("alpha", [0.37, 1.0])
def test_sib_thermal_increment_matches_fdt_oracle(tmp_path, alpha):
    """The seeded thermal increment must carry the fluctuation-dissipation variance.

    A single step is replayed from the dumped initial and chord-midpoint fields,
    so only the integrator is under test: the same countered Gaussian increment
    enters the predictor and the corrector, exactly as in the kernels. The
    increment variance must be 2 alpha gamma k_B T dt / (mu_s (1 + alpha**2));
    dividing its amplitude by (1 + alpha**2) once more, as the pre-fix
    normalization did, cools the bath to T / (1 + alpha**2) and fails here.
    """
    seed = 987654
    gamma = 1200.0
    temperature = 300.0
    time_step_fs = 0.1
    model = _base_model()
    initial = _evaluate(tmp_path / "initial", model)
    magnitude = np.linalg.norm(initial["spin"], axis=1)
    direction = initial["spin"] / magnitude[:, None]

    drift = gamma * time_step_fs / 1000.0 / (1.0 + alpha * alpha)
    sigma = np.sqrt(
        2.0 * alpha * gamma * BOLTZMANN * temperature * (time_step_fs / 1000.0)
        / (magnitude * (1.0 + alpha * alpha)))
    increment = sigma[:, None] * np.array([
        [_sib_gaussian(seed, atom_index, 0, component) for component in range(3)]
        for atom_index in range(1, magnitude.size + 1)])

    predictor_omega = drift * (
        initial["mforce"] + alpha * _cross(direction, initial["mforce"])) + increment
    predictor = _cayley(direction, predictor_omega)
    midpoint_direction = 0.5 * (direction + predictor)
    midpoint = _evaluate(
        tmp_path / "midpoint",
        _replace_spins(model, magnitude[:, None] * midpoint_direction))

    corrector_omega = drift * (
        midpoint["mforce"]
        + alpha * _cross(midpoint_direction, midpoint["mforce"])) + increment
    expected_spin = magnitude[:, None] * _cayley(direction, corrector_omega)

    result = _run(
        tmp_path / "sib",
        "potential nep.txt\n"
        f"ensemble nve_sib alpha {alpha} gamma {gamma} "
        f"stemp {temperature} seed {seed}\n"
        f"time_step {time_step_fs}\n"
        "dump_xyz 1 state.xyz mass spin mforce\n"
        "run 1\n",
        model)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = _read_last_frame(tmp_path / "sib" / "state.xyz")
    np.testing.assert_allclose(actual["spin"], expected_spin, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(
        np.linalg.norm(actual["spin"], axis=1), magnitude, rtol=0.0, atol=3.0e-7)


def test_sib_noise_is_seeded_and_preserves_spin_magnitude(tmp_path):
    command = (
        "potential nep.txt\n"
        "ensemble nvt_sib 300 300 100 lattice off alpha 0.1 seed {seed}\n"
        "time_step 0.1\n"
        "dump_xyz 5 state.xyz mass spin mforce\n"
        "run 5\n")
    model = _base_model()
    initial_spin = np.array([
        [float(value) for value in line.split()[-3:]]
        for line in model.splitlines()[2:]
    ])
    frames = []
    for name, seed in (("same_a", 314159), ("same_b", 314159), ("different", 271828)):
        result = _run(tmp_path / name, command.format(seed=seed), model)
        assert result.returncode == 0, result.stdout + result.stderr
        frames.append(_read_last_frame(tmp_path / name / "state.xyz")["spin"])

    np.testing.assert_allclose(frames[0], frames[1], rtol=0.0, atol=0.0)
    assert np.max(np.abs(frames[0] - frames[2])) > 1.0e-8
    np.testing.assert_allclose(
        np.linalg.norm(frames[0], axis=1),
        np.linalg.norm(initial_spin, axis=1),
        rtol=0.0,
        atol=3.0e-7)


@pytest.mark.parametrize(
    "ensemble",
    [
        "nvt_sib 300 300 100 alpha",
        "nvt_sib 300 300 100 alpha -1",
        "nvt_sib 300 300 100 alpha nan",
        "nvt_sib 300 300 100 gamma 0",
        "nvt_sib 300 300 100 gamma nope",
        "nvt_sib 300 300 100 stemp -2",
        "nvt_sib 300 300 100 stemp -0.5",
        "nvt_sib 300 300 100 seed 0",
        "nvt_sib 300 300 100 lattice maybe",
        "nvt_sib 300 300 100 seed 1 seed 2",
        "nvt_sib 300 300 100 future 1",
        "nve_sib stemp 0",
        "nve_sib lattice off",
        "nve_sib alpha",
        "npt_sib temp 300 300 iso 0 0 lattice off",
        "npt_sib temp 300 300 iso 0 0 alpha 0.1 alpha 0.2",
    ],
)
def test_sib_parser_fails_closed(tmp_path, ensemble):
    result = _run(
        tmp_path / "case",
        "potential nep.txt\n"
        f"ensemble {ensemble}\n"
        "run 1\n")
    assert result.returncode != 0


def test_npt_sib_composes_mttk_and_sib(tmp_path):
    result = _run(
        tmp_path / "npt",
        "potential nep.txt\n"
        "ensemble npt_sib temp 300 300 iso 0 0 "
        "tperiod 100 pperiod 1000 alpha 0.1 stemp -1 seed 123\n"
        "time_step 0.01\n"
        "dump_xyz 1 state.xyz mass spin mforce\n"
        "run 1\n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Integrate spins with the semi-implicit B (SIB) method" in result.stdout
    assert "Use Nose-Hoover thermostat and Parrinello-Rahman barostat" in result.stdout
    state = _read_last_frame(tmp_path / "npt" / "state.xyz")
    assert np.all(np.isfinite(state["spin"]))
    assert np.all(np.isfinite(state["mforce"]))


def test_sib_requires_spin_potential(tmp_path):
    case = tmp_path / "nonspin"
    case.mkdir()
    shutil.copy(Path(__file__).parent / "fixtures" / "models" / "nep_C.txt", case / "nep.txt")
    shutil.copy(
        Path(__file__).parent / "fixtures" / "structures" / "C-nat16-rattled.xyz",
        case / "model.xyz")
    (case / "run.in").write_text(
        "potential nep.txt\n"
        "ensemble nvt_sib 300 300 100 alpha 0.1\n"
        "run 1\n")
    result = subprocess.run(
        [str(GPUMD)], cwd=case,
        capture_output=True, text=True, check=False)
    assert result.returncode != 0
