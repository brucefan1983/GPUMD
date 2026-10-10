"""Tests of the active keyword.

Checking the uncertainty leaves the trajectory unchanged.
The frames in active.xyz hold the energy, virial and forces of the main potential alone.
The stress adds the kinetic tensor to the virial.
"""
import shutil
import subprocess

import numpy as np
import pytest
from ase import Atoms, units
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

NVE = ['velocity 300 seed 1', 'ensemble nve', 'time_step 1']
ADD_FORCE = ['add_force 0 0 0.5 0 0', 'add_force 0 1 -0.5 0 0']
# dump_thermo follows the keywords of a test and writes the thermo vector after they have run.
RUN = ['dump_thermo 1', 'run 20']


def _write_scaled_model(model_path, path, factor=1.05):
    """Writes to path a copy of a NEP4 model with every parameter scaled by factor, a second model
    for the same species."""
    lines = model_path.read_text().splitlines()
    header_length = 6  # the lines from nep4 to ANN
    parameters = [f'{float(line) * factor:.7e}' for line in lines[header_length:]]
    path.write_text('\n'.join(lines[:header_length] + parameters) + '\n')
    return path


def _rattled_diamond():
    atoms = bulk('C', 'diamond', a=3.567, cubic=True).repeat(2)
    atoms.rattle(0.05, seed=1)
    return atoms


def _run(directory, gpumd_command, atoms, potentials, keywords, check=True):
    """Runs gpumd in directory and returns the completed process.
    model.xyz is written from atoms, alternating between groups 0 and 1 of grouping method 0,
    unless atoms is None."""
    directory.mkdir(exist_ok=True)
    if atoms is not None:
        groups = [list(range(0, len(atoms), 2)), list(range(1, len(atoms), 2))]
        write_xyz(str(directory / 'model.xyz'), atoms, groupings=[groups])
    run_in = [f'potential {path}' for path in potentials] + keywords
    (directory / 'run.in').write_text('\n'.join(run_in) + '\n')
    return subprocess.run(
        [gpumd_command], cwd=directory, capture_output=True, text=True, check=check)


@pytest.fixture
def models(tmp_path):
    """nep_C.txt and a copy with every parameter scaled by 1.05."""
    return [MODEL_PATH, _write_scaled_model(MODEL_PATH, tmp_path / 'nep_C_scaled.txt')]


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


def test_active_leaves_run_unchanged(tmp_path, gpumd_command, models):
    for name, active in [('reference', []), ('checked', ['active 1 1 1 0 0'])]:
        _run(tmp_path / name, gpumd_command, _rattled_diamond(), models,
             NVE + ADD_FORCE + active + RUN)
    reference = np.loadtxt(tmp_path / 'reference' / 'thermo.out')
    assert np.array_equal(np.loadtxt(tmp_path / 'checked' / 'thermo.out'), reference)
    frames = read(tmp_path / 'checked' / 'active.xyz', index=':')
    assert len(frames) == len(reference)
    _assert_main_potential(frames[-1], gpumd_command)


def test_active_leaves_later_run_unchanged(tmp_path, gpumd_command, models):
    """The velocity keyword of a later run reads the host copies of the positions and velocities,
    which active leaves unchanged."""
    # Velocities are drawn with zero angular momentum about the center of the cluster.
    cluster = _rattled_diamond()
    cluster.set_cell([30, 30, 30])
    cluster.set_pbc(False)
    second_run = ['velocity 300 seed 2', 'ensemble nve', 'dump_thermo 1', 'run 5']
    for name, active in [('reference', []), ('checked', ['active 1 1 0 0 0'])]:
        _run(tmp_path / name, gpumd_command, cluster, models,
             NVE + active + ['run 20'] + second_run)
    assert np.array_equal(
        np.loadtxt(tmp_path / 'checked' / 'thermo.out'),
        np.loadtxt(tmp_path / 'reference' / 'thermo.out'))


def test_active_keeps_average_mode(tmp_path, gpumd_command, models):
    """The average of the potentials propagates the run under dump_observer average, whichever of
    the two keywords comes first, and active.xyz holds the main potential alone."""
    observer = 'dump_observer average 1 1 0 0'
    active = 'active 1 1 1 0 0'
    runs = {'reference': [observer], 'active_first': [active, observer],
            'active_last': [observer, active]}
    for name, keywords in runs.items():
        _run(tmp_path / name, gpumd_command, _rattled_diamond(), models, NVE + keywords + RUN)
    reference = np.loadtxt(tmp_path / 'reference' / 'thermo.out')
    for name in ('active_first', 'active_last'):
        assert np.array_equal(np.loadtxt(tmp_path / name / 'thermo.out'), reference)
        _assert_main_potential(read(tmp_path / name / 'active.xyz', index=-1), gpumd_command)


def test_uncertainty_is_population_standard_deviation(tmp_path, gpumd_command, models):
    """The uncertainty of an atom is the norm of the standard deviations of its force components
    over the M models, with the factor 1/M, at every check."""
    _run(tmp_path, gpumd_command, _rattled_diamond(), models, NVE + ['active 4 0 0 1 0'] + RUN)
    frames = read(tmp_path / 'active.xyz', index=':')
    uncertainty = np.loadtxt(tmp_path / 'active.out', ndmin=2)[:, 1]
    assert len(frames) == len(uncertainty) == 5
    for frame, maximum in zip(frames, uncertainty):
        forces = []
        for model in models:
            atoms = frame.copy()
            atoms.calc = GPUNEP(str(model), command=gpumd_command)
            forces.append(atoms.get_forces())
        expected = np.sqrt(np.var(forces, axis=0, ddof=0).sum(axis=1))
        assert frame.arrays['uncertainty'] == approx_tol(expected, TOLERANCES['force'])
        assert frame.info['uncertainty'] == approx_tol(expected.max(), TOLERANCES['force'])
        assert maximum == approx_tol(expected.max(), TOLERANCES['force'])


def test_uncertainty_of_identical_models(tmp_path, gpumd_command):
    """Two identical models give an uncertainty of zero up to rounding."""
    _run(tmp_path, gpumd_command, _rattled_diamond(), [MODEL_PATH, MODEL_PATH],
         NVE + ['active 1 0 0 1 0'] + RUN)
    uncertainty = np.loadtxt(tmp_path / 'active.out', ndmin=2)[:, 1]
    assert len(uncertainty) == 20
    # A NaN fails both comparisons.
    assert np.all(uncertainty >= 0) and np.all(uncertainty < 1e-10)


def test_nan_uncertainty_writes_frame(tmp_path, gpumd_command):
    """A NaN uncertainty on any atom, from a model with forces that are not finite, is the maximum
    and writes the frame whatever the threshold."""
    # An isolated atom, placed first, has no neighbors and an uncertainty of zero.
    # The diamond cluster sits in a box of 30 Angstrom, beyond the cutoff of 7 Angstrom from it.
    atoms = Atoms('C', positions=[[18, 18, 18]], cell=[30, 30, 30], pbc=True) + _rattled_diamond()
    # The second model gives forces that are not finite for every atom with neighbors.
    overflow = _write_scaled_model(MODEL_PATH, tmp_path / 'nep_C_overflow.txt', factor=1e30)
    _run(tmp_path, gpumd_command, atoms, [MODEL_PATH, overflow],
         NVE + ['active 1 0 0 1 1000', 'run 5'])
    frames = read(tmp_path / 'active.xyz', index=':')
    assert len(frames) == 5
    for frame in frames:
        assert frame.arrays['uncertainty'][0] == 0
        assert np.all(np.isnan(frame.arrays['uncertainty'][1:]))
    assert np.all(np.isnan(np.loadtxt(tmp_path / 'active.out', ndmin=2)[:, 1]))


@pytest.mark.parametrize('potential', ['lj', 'nep'])
def test_active_requires_two_potentials(tmp_path, gpumd_command, potential):
    """A committee of one potential, NEP or not, stops with an input error."""
    lj = tmp_path / 'lj_C.txt'
    lj.write_text('lj 1 C\n0.002 3.4 8.0\n')
    result = _run(tmp_path, gpumd_command, _rattled_diamond(),
                  [lj if potential == 'lj' else MODEL_PATH],
                  NVE + ['active 1 0 0 0 0', 'run 1'], check=False)
    assert result.returncode != 0
    assert 'active requires at least two potentials' in result.stdout + result.stderr


def test_active_with_temperature_nep(tmp_path, gpumd_command):
    """active evaluates a temperature-dependent NEP at the temperature of the step, which rises by
    100 K per step."""
    fixtures = REPO_ROOT / 'tests_regression' / 'fixtures'
    model = fixtures / 'potentials' / 'temperature_F_nep.txt'
    potentials = [model, _write_scaled_model(model, tmp_path / 'temperature_F_nep_scaled.txt')]
    keywords = ['velocity 300 seed 1', 'ensemble nvt_ber 300 600 100', 'time_step 1']
    for name, active in [('reference', []), ('checked', ['active 1 0 0 0 0'])]:
        (tmp_path / name).mkdir()
        shutil.copy(fixtures / 'systems' / 'temperature_F_512.xyz', tmp_path / name / 'model.xyz')
        _run(tmp_path / name, gpumd_command, None, potentials,
             keywords + active + ['dump_thermo 1', 'run 3'])
    reference = np.loadtxt(tmp_path / 'reference' / 'thermo.out')
    assert np.array_equal(np.loadtxt(tmp_path / 'checked' / 'thermo.out'), reference)
    frames = read(tmp_path / 'checked' / 'active.xyz', index=':')
    energies = [frame.get_potential_energy() for frame in frames]
    # column 2 of thermo.out is the potential energy
    assert energies == approx_tol(reference[:, 2], TOLERANCES['energy'])
