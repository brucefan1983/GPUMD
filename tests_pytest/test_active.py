"""Tests of the active keyword.

Checking the uncertainty leaves the molecular dynamics run unchanged.
The frames in active.xyz hold the energy, virial and forces of the main potential alone.
The stress adds the kinetic tensor to the virial.
"""
import shutil
import subprocess

import numpy as np
import pytest
from ase import units
from ase import Atoms
from ase.build import bulk
from ase.io import read
from calorine.calculators import GPUNEP
from calorine.gpumd import write_xyz

from conftest import MODELS_DIR, REPO_ROOT, TOLERANCES, approx_tol

pytestmark = pytest.mark.fast

MODEL_PATH = MODELS_DIR / 'nep_C.txt'

# Largest difference between the forces in active.xyz and a single-point evaluation of the frame,
# in eV/Angstrom.
# The forces in active.xyz are written with eight decimals.
ACTIVE_FORCE_TOLERANCE = 2e-4


def _write_scaled_model(model_path, path, factor=1.05):
    """Writes to path a copy of a NEP4 model with every parameter scaled by factor, a second model
    for the same species."""
    lines = model_path.read_text().splitlines()
    header_length = 6  # the lines from nep4 to ANN
    parameters = [f'{float(line) * factor:.7e}' for line in lines[header_length:]]
    path.write_text('\n'.join(lines[:header_length] + parameters) + '\n')


def _write_carbon_cell(directory):
    """Writes a rattled diamond cell to model.xyz, with the atoms alternating between groups 0 and
    1 of grouping method 0."""
    atoms = bulk('C', 'diamond', a=3.567, cubic=True).repeat(2)
    atoms.rattle(0.05, seed=1)
    groups = [list(range(0, len(atoms), 2)), list(range(1, len(atoms), 2))]
    write_xyz(str(directory / 'model.xyz'), atoms, groupings=[groups])


def _run_carbon(directory, gpumd_command, keywords, second_model='nep_C_scaled.txt'):
    """Runs 20 steps of nve on a rattled diamond cell with nep_C.txt as the main potential and by
    default a scaled copy as the second, and returns thermo.out."""
    directory.mkdir()
    _write_carbon_cell(directory)
    shutil.copy(MODEL_PATH, directory)
    _write_scaled_model(MODEL_PATH, directory / 'nep_C_scaled.txt')
    run_in = [
        f'potential {MODEL_PATH.name}',
        f'potential {second_model}',
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


def _assert_main_potential(frame, gpumd_command):
    """Checks the energy, forces, virial and stress of a frame of active.xyz, written with
    velocities, against a single point of the main potential."""
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


def test_active_leaves_run_unchanged(tmp_path, gpumd_command):
    reference = _run_carbon(tmp_path / 'reference', gpumd_command, ADD_FORCE)
    checked = _run_carbon(tmp_path / 'checked', gpumd_command, ADD_FORCE + ['active 1 1 1 0 0'])
    assert np.array_equal(checked, reference)
    frames = read(tmp_path / 'checked' / 'active.xyz', index=':')
    assert len(frames) == len(reference)
    _assert_main_potential(frames[-1], gpumd_command)


def _run_cluster(directory, gpumd_command, with_active):
    """Runs a non-periodic diamond cluster for 20 steps, then initializes the velocities again and
    returns thermo.out of a second run of 5 steps."""
    directory.mkdir()
    atoms = bulk('C', 'diamond', a=3.567, cubic=True).repeat(2)
    atoms.rattle(0.05, seed=1)
    atoms.set_cell([30, 30, 30])
    atoms.set_pbc(False)
    write_xyz(str(directory / 'model.xyz'), atoms)
    shutil.copy(MODEL_PATH, directory)
    _write_scaled_model(MODEL_PATH, directory / 'nep_C_scaled.txt')
    run_in = [f'potential {MODEL_PATH.name}', 'potential nep_C_scaled.txt', 'velocity 300 seed 1',
              'ensemble nve', 'time_step 1']
    if with_active:
        run_in.append('active 1 1 0 0 0')
    run_in += ['run 20', 'velocity 300 seed 2', 'ensemble nve', 'dump_thermo 1', 'run 5']
    (directory / 'run.in').write_text('\n'.join(run_in) + '\n')
    subprocess.run([gpumd_command], cwd=directory, check=True, stdout=subprocess.DEVNULL)
    return np.loadtxt(directory / 'thermo.out')


def test_active_leaves_later_run_unchanged(tmp_path, gpumd_command):
    """The velocity keyword of a later run reads the host copies of the positions and velocities,
    which active leaves unchanged."""
    reference = _run_cluster(tmp_path / 'reference', gpumd_command, with_active=False)
    checked = _run_cluster(tmp_path / 'checked', gpumd_command, with_active=True)
    assert np.array_equal(checked, reference)


def test_active_keeps_average_mode(tmp_path, gpumd_command):
    """The average of the potentials propagates the run under dump_observer average, whichever of
    the two keywords comes first, and active.xyz holds the main potential alone."""
    observer = 'dump_observer average 1 1 0 0'
    active = 'active 1 1 1 0 0'
    reference = _run_carbon(tmp_path / 'reference', gpumd_command, [observer])
    active_first = _run_carbon(tmp_path / 'active_first', gpumd_command, [active, observer])
    active_last = _run_carbon(tmp_path / 'active_last', gpumd_command, [observer, active])
    assert np.array_equal(active_first, reference)
    assert np.array_equal(active_last, reference)
    for name in ('active_first', 'active_last'):
        _assert_main_potential(read(tmp_path / name / 'active.xyz', index=-1), gpumd_command)


def test_uncertainty_is_population_standard_deviation(tmp_path, gpumd_command):
    """The uncertainty of an atom is the norm of the standard deviations of its force components
    over the M models, with the factor 1/M, at every check."""
    directory = tmp_path / 'run'
    _run_carbon(directory, gpumd_command, ['active 4 0 0 1 0'])
    frames = read(directory / 'active.xyz', index=':')
    uncertainty = np.loadtxt(directory / 'active.out')[:, 1]
    assert len(frames) == len(uncertainty) == 5
    for frame, maximum in zip(frames, uncertainty):
        forces = []
        for model in ('nep_C.txt', 'nep_C_scaled.txt'):
            atoms = frame.copy()
            atoms.calc = GPUNEP(str(directory / model), command=gpumd_command)
            forces.append(atoms.get_forces())
        expected = np.sqrt(np.var(forces, axis=0, ddof=0).sum(axis=1))
        assert frame.arrays['uncertainty'] == approx_tol(expected, TOLERANCES['force'])
        assert frame.info['uncertainty'] == approx_tol(expected.max(), TOLERANCES['force'])
        assert maximum == approx_tol(expected.max(), TOLERANCES['force'])


def test_uncertainty_of_identical_models(tmp_path, gpumd_command):
    """Two identical models give an uncertainty of zero up to rounding."""
    directory = tmp_path / 'run'
    _run_carbon(directory, gpumd_command, ['active 1 0 0 1 0'], second_model=MODEL_PATH.name)
    uncertainty = np.loadtxt(directory / 'active.out')[:, 1]
    assert len(uncertainty) == 20
    # A NaN fails both comparisons.
    assert np.all(uncertainty >= 0) and np.all(uncertainty < 1e-10)


def test_nan_uncertainty_writes_frame(tmp_path, gpumd_command):
    """A NaN uncertainty on any atom, from a model with forces that are not finite, is the maximum
    and writes the frame whatever the threshold."""
    # An isolated atom, placed first, has no neighbors and an uncertainty of zero.
    # The diamond cluster sits in a box of 30 Angstrom, beyond the cutoff of 7 Angstrom from it.
    cluster = bulk('C', 'diamond', a=3.567, cubic=True).repeat(2)
    cluster.rattle(0.05, seed=1)
    atoms = Atoms('C', positions=[[18, 18, 18]], cell=[30, 30, 30], pbc=True) + cluster
    write_xyz(str(tmp_path / 'model.xyz'), atoms)
    shutil.copy(MODEL_PATH, tmp_path)
    # The second model gives forces that are not finite for every atom with neighbors.
    _write_scaled_model(MODEL_PATH, tmp_path / 'nep_C_overflow.txt', factor=1e30)
    run_in = [f'potential {MODEL_PATH.name}', 'potential nep_C_overflow.txt', 'velocity 300 seed 1',
              'ensemble nve', 'time_step 1', 'active 1 0 0 1 1000', 'run 5']
    (tmp_path / 'run.in').write_text('\n'.join(run_in) + '\n')
    subprocess.run([gpumd_command], cwd=tmp_path, check=True, stdout=subprocess.DEVNULL)
    frames = read(tmp_path / 'active.xyz', index=':')
    assert len(frames) == 5
    for frame in frames:
        assert frame.arrays['uncertainty'][0] == 0
        assert np.all(np.isnan(frame.arrays['uncertainty'][1:]))
    assert np.all(np.isnan(np.loadtxt(tmp_path / 'active.out')[:, 1]))


@pytest.mark.parametrize('potential', ['lj_C.txt', MODEL_PATH.name])
def test_active_requires_two_potentials(tmp_path, gpumd_command, potential):
    """A committee of one potential, NEP or not, stops with an input error."""
    _write_carbon_cell(tmp_path)
    (tmp_path / 'lj_C.txt').write_text('lj 1 C\n0.002 3.4 8.0\n')
    shutil.copy(MODEL_PATH, tmp_path)
    run_in = [f'potential {potential}', 'velocity 300 seed 1', 'ensemble nve', 'time_step 1',
              'active 1 0 0 0 0', 'run 1']
    (tmp_path / 'run.in').write_text('\n'.join(run_in) + '\n')
    result = subprocess.run(
        [gpumd_command], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert 'active requires at least two potentials' in result.stdout + result.stderr


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
        'ensemble nvt_ber 300 600 100',
        'time_step 1',
    ]
    if with_active:
        run_in.append('active 1 0 0 0 0')
    run_in += ['dump_thermo 1', 'run 3']
    (directory / 'run.in').write_text('\n'.join(run_in) + '\n')
    subprocess.run([gpumd_command], cwd=directory, check=True, stdout=subprocess.DEVNULL)
    return np.loadtxt(directory / 'thermo.out')


def test_active_with_temperature_nep(tmp_path, gpumd_command):
    """active evaluates a temperature-dependent NEP at the temperature of the step, which rises by
    100 K per step."""
    reference = _run_temperature_nep(tmp_path / 'reference', gpumd_command, with_active=False)
    checked = _run_temperature_nep(tmp_path / 'checked', gpumd_command, with_active=True)
    assert np.array_equal(checked, reference)

    frames = read(tmp_path / 'checked' / 'active.xyz', index=':')
    energies = [frame.get_potential_energy() for frame in frames]
    # column 2 of thermo.out is the potential energy
    assert energies == approx_tol(reference[:, 2], TOLERANCES['energy'])
