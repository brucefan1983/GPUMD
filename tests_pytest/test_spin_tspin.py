"""TSPIN parser, initialization, splitting, and restart regression tests."""

import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from spin_runtime_helpers import base_model as _model_with_zero_lattice_velocity

from spin_fixtures import load_fixture, selective_type_model


FIXTURE = Path(__file__).parent / "fixtures" / "nep_spin3"
GPUMD = Path(os.environ.get("GPUMD_COMMAND", Path(__file__).parents[1] / "src" / "gpumd"))
K_B = 8.617343e-5
PI = 3.14159265358979
TIME_UNIT_CONVERSION = 1.018051e1
MASK64 = (1 << 64) - 1


def _prepare_case(case_dir, run_input, model_text=None):
    case_dir.mkdir()
    shutil.copy(FIXTURE / "nep4_spin3_o3c2_uniform.nep", case_dir / "nep.txt")
    (case_dir / "model.xyz").write_text(
        model_text if model_text is not None else _model_with_zero_lattice_velocity())
    (case_dir / "run.in").write_text(run_input)
    return subprocess.run(
        [str(GPUMD)],
        cwd=case_dir,
        capture_output=True,
        text=True,
        check=False)


def _read_spin_frame(path):
    lines = path.read_text().splitlines()
    number_of_atoms = int(lines[0])
    values = np.array([
        [float(value) for value in line.split()[1:]]
        for line in lines[-number_of_atoms:]
    ])
    return {
        "mass": values[:, 3],
        "spin": values[:, 4:7],
        "mforce": values[:, 7:10],
        "spin_velocity": values[:, 10:13] * TIME_UNIT_CONVERSION,
    }


def _read_last_lattice(path):
    comments = [
        line for line in path.read_text().splitlines()
        if re.search(r'\bLattice="', line)
    ]
    values = re.search(r'Lattice="([^"]+)"', comments[-1]).group(1)
    return np.array([float(value) for value in values.split()])


def _splitmix64(value):
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def _uniform_01(state):
    state = _splitmix64(state)
    return state, (state >> 11) / 9007199254740992.0


def _gaussian(seed, atom_index, component):
    state = seed
    state ^= (atom_index * 0xD1B54A32D192ED03) & MASK64
    state ^= (component * 0x9E3779B97F4A7C15) & MASK64
    state, u1 = _uniform_01(state)
    while u1 <= 0.0:
        state, u1 = _uniform_01(state)
    state, u2 = _uniform_01(state)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * PI * u2)


def _initial_spin_velocity(seed, mass, temperature, mass_factor):
    number_of_atoms = len(mass)
    velocity = np.array([
        [0.5 * _gaussian(seed, atom + 1, component) for component in range(3)]
        for atom in range(number_of_atoms)
    ])
    twice_kinetic_energy = np.sum(
        mass[:, None] * mass_factor * velocity * velocity)
    target = 3.0 * number_of_atoms * K_B * temperature
    return velocity * math.sqrt(target / twice_kinetic_energy)


def _nhc(position, velocity, mass, twice_kinetic_energy,
         thermal_energy, degrees_of_freedom, half_time_step):
    weights = (
        0.784513610477560,
        0.235573213359357,
        -1.17767998417887,
        1.31518632068391,
        -1.17767998417887,
        0.235573213359357,
        0.784513610477560,
    )
    factor = 1.0
    for weight in weights:
        dt2 = half_time_step * weight / 4.0
        dt4 = dt2 * 0.5
        dt8 = dt4 * 0.5
        for _ in range(4):
            acceleration = velocity[-2] ** 2 / mass[-2] - thermal_energy
            velocity[-1] += dt4 * acceleration
            for index in range(len(velocity) - 2, -1, -1):
                scale = math.exp(-dt8 * velocity[index + 1] / mass[index + 1])
                if index == 0:
                    acceleration = (
                        twice_kinetic_energy -
                        degrees_of_freedom * thermal_energy)
                else:
                    acceleration = (
                        velocity[index - 1] ** 2 / mass[index - 1] -
                        thermal_energy)
                velocity[index] = scale * (
                    scale * velocity[index] + dt4 * acceleration)
            for index in range(len(position) - 1, -1, -1):
                position[index] += dt2 * velocity[index] / mass[index]
            local_factor = math.exp(-dt2 * velocity[0] / mass[0])
            twice_kinetic_energy *= local_factor * local_factor
            factor *= local_factor
            for index in range(len(velocity) - 1):
                scale = math.exp(-dt8 * velocity[index + 1] / mass[index + 1])
                if index == 0:
                    acceleration = (
                        twice_kinetic_energy -
                        degrees_of_freedom * thermal_energy)
                else:
                    acceleration = (
                        velocity[index - 1] ** 2 / mass[index - 1] -
                        thermal_energy)
                velocity[index] = scale * (
                    scale * velocity[index] + dt4 * acceleration)
            acceleration = velocity[-2] ** 2 / mass[-2] - thermal_energy
            velocity[-1] += dt4 * acceleration
    return factor


@pytest.mark.parametrize("pressure_control", [False, True], ids=["nvt", "npt"])
def test_tspin_one_step_matches_public_formula_and_gpumd_nhc_oracle(
        tmp_path, pressure_control):
    # Obtain the initial magnetic force from a fixed-spin run. This keeps the
    # integrator oracle independent of the NEP_Spin derivative implementation.
    static_model = _model_with_zero_lattice_velocity()
    lines = static_model.splitlines()
    lines[1] = lines[1].replace(":spin:R:3", ":spin:R:3:spin_vel:R:3")
    for index in range(2, len(lines)):
        lines[index] += " 0 0 0"
    static_result = _prepare_case(
        tmp_path / "static",
        "potential nep.txt\n"
        "ensemble nve\n"
        "time_step 0\n"
        "dump_xyz 1 state.xyz mass spin mforce spin_velocity\n"
        "run 1\n",
        "\n".join(lines) + "\n")
    assert static_result.returncode == 0, static_result.stdout + static_result.stderr
    initial = _read_spin_frame(tmp_path / "static" / "state.xyz")

    temperature = 300.0
    coupling = 100.0
    mass_factor = 1.5
    seed = 24681357
    time_step_fs = 0.1
    ensemble = (
        f"ensemble npt_tspin temp {temperature} {temperature} iso 0 0 "
        f"tperiod {coupling} pperiod 1000 mass_factor {mass_factor} seed {seed}\n"
        if pressure_control else
        f"ensemble nvt_tspin {temperature} {temperature} {coupling} "
        f"mass_factor {mass_factor} seed {seed}\n"
    )
    result = _prepare_case(
        tmp_path / "tspin",
        "potential nep.txt\n"
        + ensemble
        + f"time_step {time_step_fs}\n"
        "dump_xyz 1 state.xyz mass spin mforce spin_velocity\n"
        "run 1\n")
    assert result.returncode == 0, result.stdout + result.stderr
    actual = _read_spin_frame(tmp_path / "tspin" / "state.xyz")

    dt = time_step_fs / TIME_UNIT_CONVERSION
    number_of_atoms = len(actual["mass"])
    velocity = _initial_spin_velocity(
        seed, actual["mass"], temperature, mass_factor)
    position_nhc = np.zeros(4)
    velocity_nhc = np.array([1.0, -1.0, 1.0, -1.0])
    thermal_energy = K_B * temperature
    mass_nhc = np.full(4, thermal_energy * (dt * coupling) ** 2)
    mass_nhc[0] *= 3.0 * number_of_atoms

    twice_kinetic_energy = np.sum(
        actual["mass"][:, None] * mass_factor * velocity * velocity)
    factor = _nhc(
        position_nhc, velocity_nhc, mass_nhc, twice_kinetic_energy,
        thermal_energy, 3.0 * number_of_atoms, 0.5 * dt)
    velocity *= factor
    velocity += (
        0.5 * dt * initial["mforce"] /
        (actual["mass"][:, None] * mass_factor))
    expected_spin = initial["spin"] + dt * velocity

    velocity += (
        0.5 * dt * actual["mforce"] /
        (actual["mass"][:, None] * mass_factor))
    twice_kinetic_energy = np.sum(
        actual["mass"][:, None] * mass_factor * velocity * velocity)
    factor = _nhc(
        position_nhc, velocity_nhc, mass_nhc, twice_kinetic_energy,
        thermal_energy, 3.0 * number_of_atoms, 0.5 * dt)
    expected_velocity = velocity * factor

    np.testing.assert_allclose(actual["spin"], expected_spin, rtol=0.0, atol=2.0e-7)
    np.testing.assert_allclose(
        actual["spin_velocity"], expected_velocity, rtol=0.0, atol=2.0e-7)


@pytest.mark.parametrize(
    "ensemble",
    [
        "nvt_tspin 300 300 100 mass_factor",
        "nvt_tspin 300 300 100 mass_factor 0",
        "nvt_tspin 300 300 100 mass_factor nan",
        "nvt_tspin 300 300 100 seed 0",
        "nvt_tspin 300 300 100 seed nope",
        "nvt_tspin 300 300 100 seed 1 seed 2",
        "nvt_tspin 300 300 100 mass_factor 1 mass_factor 2",
        "nvt_tspin 300 300 100 mass_factor X 0.001 Fe 0",
        "nvt_tspin 300 300 100 mass_factor Fe -1 Ge 0.001",
        "nvt_tspin 300 300 100 mass_factor Fe 0 Ge 0",
        "nvt_tspin 300 300 100 lattice maybe",
        "nvt_tspin 300 300 100 lattice off lattice on",
        "nvt_tspin 300 300 100 future 1",
        "npt_tspin temp 300 300 iso 0 0 mass_factor",
        "npt_tspin temp 300 300 iso 0 0 seed 1 seed 2",
        "npt_tspin temp 300 300 iso 0 0 future 1",
    ],
)
def test_tspin_parser_fails_closed(tmp_path, ensemble):
    result = _prepare_case(
        tmp_path / "case",
        "potential nep.txt\n"
        f"ensemble {ensemble}\n"
        "run 1\n")
    assert result.returncode != 0


def test_element_mass_factor_requires_every_present_element(tmp_path):
    directory = tmp_path / "missing_element"
    directory.mkdir()
    (directory / "nep.txt").write_text(selective_type_model())
    (directory / "model.xyz").write_text(
        '2\nLattice="16 0 0 0 16 0 0 0 16" '
        'Properties=species:S:1:pos:R:3:vel:R:3:spin:R:3 pbc="T T T"\n'
        'Fe 0 0 0 0 0 0 1 0.2 -0.1\n'
        'Ge 3 0 0 0 0 0 0.4 -0.3 0.7\n')
    (directory / "run.in").write_text(
        "potential nep.txt\n"
        "ensemble nvt_tspin 300 300 100 mass_factor Fe 0.001\n"
        "run 1\n")
    result = subprocess.run(
        [str(GPUMD)], cwd=directory,
        capture_output=True, text=True, check=False)
    assert result.returncode != 0


def test_npt_tspin_composes_mttk_pressure_and_spin_integrators(tmp_path):
    initial_lattice = _read_last_lattice(FIXTURE / "model_large_box.xyz")
    initial_spin = np.array([
        [float(value) for value in line.split()[-3:]]
        for line in (FIXTURE / "model_large_box.xyz").read_text().splitlines()[2:]
    ])
    result = _prepare_case(
        tmp_path / "npt",
        "potential nep.txt\n"
        "ensemble npt_tspin temp 300 300 iso 0 0 tperiod 100 pperiod 1000 "
        "mass_factor 1.5 seed 2468\n"
        "time_step 0.1\n"
        "dump_xyz 1 state.xyz mass spin mforce spin_velocity\n"
        "run 1\n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Integrate spins with TSPIN" in result.stdout
    assert "Use Nose-Hoover thermostat and Parrinello-Rahman barostat" in result.stdout

    actual = _read_spin_frame(tmp_path / "npt" / "state.xyz")
    final_lattice = _read_last_lattice(tmp_path / "npt" / "state.xyz")
    assert np.max(np.abs(actual["spin"] - initial_spin)) > 1.0e-10
    assert np.all(np.isfinite(final_lattice))
    assert np.max(np.abs(final_lattice - initial_lattice)) > 0.0


def test_nvt_tspin_lattice_off_only_integrates_spins(tmp_path):
    lines = (FIXTURE / "model_large_box.xyz").read_text().splitlines()
    lines[1] = lines[1].replace(
        "Properties=species:S:1:pos:R:3:spin:R:3",
        "Properties=species:S:1:pos:R:3:vel:R:3:spin:R:3")
    initial_position = []
    initial_velocity = []
    initial_spin = []
    for index in range(2, len(lines)):
        fields = lines[index].split()
        velocity = [0.01 * index, -0.02 * index, 0.03 * index]
        initial_position.append([float(value) for value in fields[1:4]])
        initial_velocity.append(velocity)
        initial_spin.append([float(value) for value in fields[4:7]])
        lines[index] = " ".join(
            fields[:4] + [str(value) for value in velocity] + fields[4:])

    result = _prepare_case(
        tmp_path / "spin_only",
        "potential nep.txt\n"
        "ensemble nvt_tspin 300 300 100 lattice off "
        "mass_factor 0.001 seed 2468\n"
        "time_step 0.01\n"
        "dump_xyz 5 state.xyz velocity spin spin_velocity\n"
        "run 5\n",
        "\n".join(lines) + "\n")
    assert result.returncode == 0, result.stdout + result.stderr

    rows = np.array([
        [float(value) for value in line.split()[1:]]
        for line in (tmp_path / "spin_only" / "state.xyz").read_text().splitlines()
        [-len(initial_position):]
    ])
    np.testing.assert_allclose(rows[:, 0:3], initial_position, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(rows[:, 3:6], initial_velocity, rtol=0.0, atol=0.0)
    assert np.max(np.abs(rows[:, 6:9] - np.array(initial_spin))) > 1.0e-12


def test_tspin_integrates_coordinates_excluded_by_potential_response_mask(tmp_path):
    directory = tmp_path / "potential_mask"
    directory.mkdir()
    (directory / "nep.txt").write_text(selective_type_model())
    (directory / "model.xyz").write_text(
        '2\nLattice="16 0 0 0 16 0 0 0 16" '
        'Properties=species:S:1:pos:R:3:vel:R:3:spin:R:3:spin_vel:R:3 '
        'pbc="T T T"\n'
        'Fe 0 0 0 0 0 0 1 0.2 -0.1 0.08 -0.03 0.04\n'
        'Ge 3 0 0 0 0 0 0.4 -0.3 0.7 0.15 -0.05 0.1\n')
    (directory / "run.in").write_text(
        "potential nep.txt\n"
        "ensemble nvt_tspin 300 300 100 mass_factor 1.5 seed 97531\n"
        "time_step 0.1\n"
        "dump_xyz 1 state.xyz mass spin mforce spin_velocity\n"
        "run 1\n")
    result = subprocess.run(
        [str(GPUMD)], cwd=directory,
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr

    actual = _read_spin_frame(directory / "state.xyz")
    initial_excluded_spin = np.array([0.4, -0.3, 0.7])
    assert np.max(np.abs(actual["mforce"][1])) <= 1.0e-12
    assert np.max(np.abs(actual["spin"][1] - initial_excluded_spin)) > 1.0e-12
    assert np.max(np.abs(actual["spin_velocity"][1])) > 1.0e-12


def test_element_mass_factor_zero_freezes_only_selected_spins(tmp_path):
    directory = tmp_path / "element_mass"
    directory.mkdir()
    (directory / "nep.txt").write_text(selective_type_model())
    (directory / "model.xyz").write_text(
        '2\nLattice="16 0 0 0 16 0 0 0 16" '
        'Properties=species:S:1:pos:R:3:vel:R:3:spin:R:3:spin_vel:R:3 '
        'pbc="T T T"\n'
        'Fe 0 0 0 0 0 0 1 0.2 -0.1 0.08 -0.03 0.04\n'
        'Ge 3 0 0 0 0 0 0.4 -0.3 0.7 0.15 -0.05 0.1\n')
    (directory / "run.in").write_text(
        "potential nep.txt\n"
        "ensemble nvt_tspin 300 300 100 mass_factor Fe 0 Ge 1.5 seed 97531\n"
        "time_step 0.1\n"
        "dump_xyz 1 state.xyz mass spin mforce spin_velocity\n"
        "run 1\n")
    result = subprocess.run(
        [str(GPUMD)], cwd=directory,
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr

    actual = _read_spin_frame(directory / "state.xyz")
    np.testing.assert_allclose(actual["spin"][0], [1.0, 0.2, -0.1], atol=0.0)
    np.testing.assert_allclose(actual["spin_velocity"][0], 0.0, atol=0.0)
    assert np.max(np.abs(actual["spin"][1] - [0.4, -0.3, 0.7])) > 1.0e-12
    assert "spin mass factor for Fe is 0" in result.stdout
    assert "spin mass factor for Ge is 1.5" in result.stdout


def test_npt_tspin_runs_with_spin3_force_path(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "nep_spin3"
    _, oracle = load_fixture("o3c2_uniform")
    case = oracle["cases"]["chiral_soc"]
    directory = tmp_path / "spin3"
    directory.mkdir()
    shutil.copy(fixture / "nep4_spin3_o3c2_uniform.nep", directory / "nep.txt")
    lattice = " ".join(str(value) for value in case["cell"])
    lines = [
        str(len(case["types"])),
        f'Lattice="{lattice}" Properties=species:S:1:pos:R:3:vel:R:3:spin:R:3 '
        'pbc="T T T"',
    ]
    for symbol, position, spin in zip(
            case["types"], case["positions"], case["spins"]):
        values = position + [0.0, 0.0, 0.0] + spin
        lines.append(symbol + " " + " ".join(str(value) for value in values))
    (directory / "model.xyz").write_text("\n".join(lines) + "\n")
    (directory / "run.in").write_text(
        "potential nep.txt\n"
        "ensemble npt_tspin temp 300 300 iso 0 0 "
        "tperiod 100 pperiod 1000 seed 13579\n"
        "time_step 0.01\n"
        "dump_xyz 1 state.xyz mass spin mforce spin_velocity\n"
        "run 1\n")
    result = subprocess.run(
        [str(GPUMD)], cwd=directory,
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = _read_spin_frame(directory / "state.xyz")
    assert np.all(np.isfinite(actual["spin"]))
    assert np.all(np.isfinite(actual["mforce"]))


def test_tspin_requires_spin_potential(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    shutil.copy(
        Path(__file__).parent / "fixtures" / "models" / "nep_C.txt",
        case / "nep.txt")
    shutil.copy(
        Path(__file__).parent / "fixtures" / "structures" / "C-nat16-rattled.xyz",
        case / "model.xyz")
    (case / "run.in").write_text(
        "potential nep.txt\n"
        "ensemble nvt_tspin 300 300 100\n"
        "run 1\n")
    result = subprocess.run(
        [str(GPUMD)], cwd=case,
        capture_output=True, text=True, check=False)
    assert result.returncode != 0


def test_tspin_reuses_velocity_across_runs_and_restart(tmp_path):
    first = _prepare_case(
        tmp_path / "first",
        "potential nep.txt\n"
        "ensemble nvt_tspin 300 300 100 seed 17\n"
        "time_step 0.1\n"
        "dump_restart 1\n"
        "run 1\n"
        "ensemble nvt_tspin 300 300 100 seed 999\n"
        "run 1\n")
    assert first.returncode == 0, first.stdout + first.stderr
    assert first.stdout.count("Initialized TSPIN velocities with seed") == 1
    assert "Reuse initialized spin velocities; TSPIN seed is not used." in first.stdout

    restart_model = (tmp_path / "first" / "restart.xyz").read_text()
    restarted = _prepare_case(
        tmp_path / "restarted",
        "potential nep.txt\n"
        "ensemble nvt_tspin 300 300 100 seed 123\n"
        "time_step 0.1\n"
        "run 1\n",
        restart_model)
    assert restarted.returncode == 0, restarted.stdout + restarted.stderr
    assert "Reuse initialized spin velocities; TSPIN seed is not used." in restarted.stdout
    assert "Initialized TSPIN velocities with seed" not in restarted.stdout


@pytest.mark.slow
def test_tspin_short_temperature_statistics(tmp_path):
    result = _prepare_case(tmp_path / "thermal", "replicate 5 5 5\npotential nep.txt\n"
        "ensemble nvt_tspin 300 300 100 mass_factor 1.5 seed 314159\n"
        "time_step .1\ndump_xyz 10 trajectory.xyz mass spin spin_velocity\nrun 500\n")
    assert result.returncode == 0, result.stdout + result.stderr
    lines = (tmp_path / "thermal/trajectory.xyz").read_text().splitlines()
    n = int(lines[0])
    temperatures = []
    for start in range(0, len(lines), n + 2):
        rows = np.array([[float(x) for x in line.split()[1:]] for line in lines[start+2:start+n+2]])
        kinetic = np.sum(rows[:, 3] * 1.5 * np.sum((rows[:, 7:10] * TIME_UNIT_CONVERSION)**2, axis=1))
        temperatures.append(kinetic / (3 * n * K_B))
    assert temperatures and np.isfinite(temperatures).all()
    assert 240 <= np.mean(temperatures) <= 360


def test_ordinary_nhc_keeps_spins_fixed(tmp_path):
    model = _model_with_zero_lattice_velocity()
    initial = np.array([[float(x) for x in row.split()[7:10]] for row in model.splitlines()[2:]])
    result = _prepare_case(tmp_path / "fixed", "potential nep.txt\n"
        "ensemble nvt_nhc 300 300 100\ntime_step .1\n"
        "dump_xyz 10 state.xyz mass spin mforce\nrun 10\n", model)
    assert result.returncode == 0, result.stdout + result.stderr
    np.testing.assert_array_equal(_read_spin_frame(tmp_path / "fixed/state.xyz")["spin"], initial)
