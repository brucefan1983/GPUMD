"""Tests of the active keyword.

Checking the uncertainty leaves the molecular dynamics run unchanged. The frames in active.xyz
hold the energy and forces of the main potential alone.
"""
import shutil
import subprocess

import numpy as np
import pytest
from ase.build import bulk
from ase.io import read
from calorine.calculators import GPUNEP
from calorine.gpumd import write_xyz

from conftest import MODELS_DIR, REPO_ROOT, TOLERANCES, approx_tol

pytestmark = pytest.mark.fast

MODEL_PATH = MODELS_DIR / 'nep_C.txt'

# Largest difference between the forces in active.xyz and a single-point evaluation of the frame,
# in eV/Angstrom. The forces in active.xyz are written with eight decimals.
ACTIVE_FORCE_TOLERANCE = 2e-4


def _write_scaled_model(model_path, path):
    """Writes to path a copy of a NEP4 model with every parameter scaled by 1.05, a second model
    for the same species."""
    lines = model_path.read_text().splitlines()
    header_length = 6  # the lines from nep4 to ANN
    parameters = [f'{float(line) * 1.05:.7e}' for line in lines[header_length:]]
    path.write_text('\n'.join(lines[:header_length] + parameters) + '\n')


def _run_add_force(directory, gpumd_command, with_active):
    directory.mkdir()
    atoms = bulk('C', 'diamond', a=3.567, cubic=True).repeat(2)
    atoms.rattle(0.05, seed=1)
    groups = [list(range(0, len(atoms), 2)), list(range(1, len(atoms), 2))]
    write_xyz(str(directory / 'model.xyz'), atoms, groupings=[groups])
    shutil.copy(MODEL_PATH, directory)
    _write_scaled_model(MODEL_PATH, directory / 'nep_C_scaled.txt')
    run_in = [
        f'potential {MODEL_PATH.name}',
        'potential nep_C_scaled.txt',
        'velocity 300 seed 1',
        'ensemble nve',
        'time_step 1',
        'add_force 0 0 0.5 0 0',
        'add_force 0 1 -0.5 0 0',
    ]
    if with_active:
        run_in.append('active 1 0 1 0 0')
    # dump_thermo follows active and writes the thermo vector after the check has run.
    run_in += ['dump_thermo 1', 'run 20']
    (directory / 'run.in').write_text('\n'.join(run_in) + '\n')
    subprocess.run([gpumd_command], cwd=directory, check=True, stdout=subprocess.DEVNULL)
    return np.loadtxt(directory / 'thermo.out')


def test_active_leaves_run_unchanged(tmp_path, gpumd_command):
    reference = _run_add_force(tmp_path / 'reference', gpumd_command, with_active=False)
    checked = _run_add_force(tmp_path / 'checked', gpumd_command, with_active=True)
    assert np.array_equal(checked, reference)

    frames = read(tmp_path / 'checked' / 'active.xyz', index=':')
    assert len(frames) == len(reference)
    frame = frames[-1]
    energy = frame.get_potential_energy()
    forces = frame.get_forces()
    frame.calc = GPUNEP(str(MODEL_PATH), command=gpumd_command)
    assert energy == approx_tol(frame.get_potential_energy(), TOLERANCES['energy'])
    assert np.max(np.abs(forces - frame.get_forces())) < ACTIVE_FORCE_TOLERANCE


def _run_temperature_nep(directory, gpumd_command, with_active):
    directory.mkdir()
    fixtures = REPO_ROOT / 'tests_regression' / 'fixtures'
    model_path = fixtures / 'potentials' / 'temperature_F_nep.txt'
    shutil.copy(model_path, directory)
    _write_scaled_model(model_path, directory / 'temperature_F_nep_scaled.txt')
    shutil.copy(fixtures / 'systems' / 'temperature_F_512.xyz', directory / 'model.xyz')
    run_in = [
        f'potential {model_path.name}',
        'potential temperature_F_nep_scaled.txt',
        'velocity 300 seed 1',
        'ensemble nvt_ber 300 300 100',
        'time_step 1',
    ]
    if with_active:
        run_in.append('active 1 0 0 0 0')
    run_in += ['dump_thermo 1', 'run 3']
    (directory / 'run.in').write_text('\n'.join(run_in) + '\n')
    subprocess.run([gpumd_command], cwd=directory, check=True, stdout=subprocess.DEVNULL)
    return np.loadtxt(directory / 'thermo.out')


def test_active_with_temperature_nep(tmp_path, gpumd_command):
    """active evaluates a temperature-dependent NEP at the temperature of the run."""
    reference = _run_temperature_nep(tmp_path / 'reference', gpumd_command, with_active=False)
    checked = _run_temperature_nep(tmp_path / 'checked', gpumd_command, with_active=True)
    assert np.array_equal(checked, reference)

    frames = read(tmp_path / 'checked' / 'active.xyz', index=':')
    energies = [frame.get_potential_energy() for frame in frames]
    # column 2 of thermo.out is the potential energy
    assert energies == approx_tol(reference[:, 2], TOLERANCES['energy'])
