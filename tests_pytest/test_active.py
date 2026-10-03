"""Tests of the active keyword.

Checking the uncertainty leaves the molecular dynamics run unchanged. The frames in active.xyz
hold the energy, virial and forces of the main potential alone. The stress adds the kinetic tensor
to the virial.
"""
import shutil
import subprocess

import numpy as np
import pytest
from ase import units
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


def _run_carbon(directory, gpumd_command, keywords):
    """Runs 20 steps of nve on a rattled diamond cell with nep_C.txt as the main potential and a
    scaled copy as the second, and returns thermo.out. The atoms alternate between groups 0 and 1
    of grouping method 0."""
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
    ]
    # dump_thermo follows the keywords and writes the thermo vector after they have run.
    run_in += keywords + ['dump_thermo 1', 'run 20']
    (directory / 'run.in').write_text('\n'.join(run_in) + '\n')
    subprocess.run([gpumd_command], cwd=directory, check=True, stdout=subprocess.DEVNULL)
    return np.loadtxt(directory / 'thermo.out')


ADD_FORCE = ['add_force 0 0 0.5 0 0', 'add_force 0 1 -0.5 0 0']


def test_active_leaves_run_unchanged(tmp_path, gpumd_command):
    reference = _run_carbon(tmp_path / 'reference', gpumd_command, ADD_FORCE)
    checked = _run_carbon(tmp_path / 'checked', gpumd_command, ADD_FORCE + ['active 1 1 1 0 0'])
    assert np.array_equal(checked, reference)

    frames = read(tmp_path / 'checked' / 'active.xyz', index=':')
    assert len(frames) == len(reference)
    frame = frames[-1]
    energy = frame.get_potential_energy()
    forces = frame.get_forces()
    stress = frame.get_stress(voigt=False)
    virial = frame.info['virial']
    velocities = frame.arrays['vel'] / units.fs  # from Å/fs to ASE units
    # calorine would write the velocities to model.xyz and add the kinetic term to the stress.
    del frame.arrays['vel']
    frame.calc = GPUNEP(str(MODEL_PATH), command=gpumd_command)
    assert energy == approx_tol(frame.get_potential_energy(), TOLERANCES['energy'])
    assert np.max(np.abs(forces - frame.get_forces())) < ACTIVE_FORCE_TOLERANCE
    volume = frame.get_volume()
    assert -virial / volume == approx_tol(frame.get_stress(voigt=False), TOLERANCES['virial'])
    # The stress is the pressure tensor, which adds the kinetic tensor to the virial.
    kinetic = np.einsum('i,ia,ib->ab', frame.get_masses(), velocities, velocities)
    assert stress == approx_tol((virial + kinetic) / volume, TOLERANCES['virial'])


def test_active_keeps_average_mode(tmp_path, gpumd_command):
    """The average of the potentials propagates the run under dump_observer average, whichever of
    the two keywords comes first."""
    observer = 'dump_observer average 1 1 0 0'
    active = 'active 1 0 0 0 0'
    reference = _run_carbon(tmp_path / 'reference', gpumd_command, [observer])
    active_first = _run_carbon(tmp_path / 'active_first', gpumd_command, [active, observer])
    active_last = _run_carbon(tmp_path / 'active_last', gpumd_command, [observer, active])
    assert np.array_equal(active_first, reference)
    assert np.array_equal(active_last, reference)
    uncertainty_first = np.loadtxt(tmp_path / 'active_first' / 'active.out')
    uncertainty_last = np.loadtxt(tmp_path / 'active_last' / 'active.out')
    assert uncertainty_last == approx_tol(uncertainty_first, TOLERANCES['force'])


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
