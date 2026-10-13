"""Tests of dump_observer in observe mode.

Each observer file holds the energy, virial and forces of its potential alone, evaluated at the
written positions in the box of the step. The stress adds the kinetic tensor to the virial.
Writing the observers leaves the molecular dynamics run unchanged.
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

# Largest difference between the forces of an observer and a single-point evaluation of its frame,
# in eV/Angstrom. The difference is 2.1e-5 in the box of the step. It is 1.7e-3 (2x2x2) and 3e-2
# (6x6x6) with npt_ber in the box of the previous step. Forces taken from the run before the
# barostat rescales the box differ by 1.8e-3 at 1500 K.
OBSERVER_FORCE_TOLERANCE = 2e-4


def _write_diamond_cell(directory, repeat, groupings=None):
    """Writes a rattled cubic diamond cell to model.xyz. npt_ber with three target pressures
    requires an orthogonal cell."""
    atoms = bulk('C', 'diamond', a=3.567, cubic=True).repeat(repeat)
    atoms.rattle(0.05, seed=1)
    write_xyz(str(directory / 'model.xyz'), atoms, groupings=groupings)


# The radial cutoff of 7 Å puts the threshold of the small-box path at a thickness of 20 Å.
# The 2x2x2 cell takes the small-box path with two images per direction.
# The 6x6x6 cell takes the large-box path.
@pytest.mark.parametrize('repeat', [2, 6], ids=['small_box', 'large_box'])
@pytest.mark.parametrize(
    'ensemble', ['nve', 'npt_ber 1500 1500 100 0 0 0 500 500 500 200'], ids=['nve', 'npt_ber']
)
def test_observer_forces_match_single_point(tmp_path, gpumd_command, repeat, ensemble):
    _write_diamond_cell(tmp_path, repeat)
    (tmp_path / 'run.in').write_text(
        f'potential {MODEL_PATH}\n'
        f'potential {MODEL_PATH}\n'
        'velocity 1500 seed 7\n'
        f'ensemble {ensemble}\n'
        'time_step 1\n'
        'dump_observer observe 5 5 0 1\n'
        'run 5\n'
    )
    result = subprocess.run(
        [gpumd_command], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr

    frame = read(tmp_path / 'observer1.xyz', index=-1)
    observer_forces = frame.get_forces()
    frame.calc = GPUNEP(
        str(MODEL_PATH), command=gpumd_command, directory=str(tmp_path / 'single_point')
    )
    assert np.max(np.abs(observer_forces - frame.get_forces())) < OBSERVER_FORCE_TOLERANCE


# add_force adds forces after the force evaluation of the step. npt_ber rescales the box and the
# positions after it. pimd propagates beads whose forces differ from those at the written positions.

CASES = {
    'add_force': [
        'velocity 300 seed 1',
        'ensemble nve',
        'time_step 1',
        'add_force 0 0 0.5 0 0',
        'add_force 0 1 -0.5 0 0',
    ],
    'npt_ber': [
        'velocity 1500 seed 7',
        'ensemble npt_ber 1500 1500 100 0 0 0 500 500 500 200',
        'time_step 1',
    ],
    'pimd': [
        'velocity 300 seed 1',
        'ensemble pimd 4 300 300 100',
        'time_step 0.5',
    ],
}


def _write_scaled_model(path):
    """Writes to path a copy of nep_C.txt with every parameter scaled by 1.05, a second model for
    the same species."""
    lines = MODEL_PATH.read_text().splitlines()
    header_length = 6  # the lines from nep4 to ANN
    parameters = [f'{float(line) * 1.05:.7e}' for line in lines[header_length:]]
    path.write_text('\n'.join(lines[:header_length] + parameters) + '\n')


def _run(directory, gpumd_command, case, number_of_potentials, with_observer):
    directory.mkdir()
    number_of_atoms = 8 * 2**3  # eight atoms in the cubic cell, repeated twice in each direction
    groups = [list(range(0, number_of_atoms, 2)), list(range(1, number_of_atoms, 2))]
    _write_diamond_cell(directory, 2, groupings=[groups])
    model_paths = [directory / MODEL_PATH.name, directory / 'nep_C_scaled.txt']
    shutil.copy(MODEL_PATH, model_paths[0])
    _write_scaled_model(model_paths[1])
    model_paths = model_paths[:number_of_potentials]
    run_in = [f'potential {path.name}' for path in model_paths] + CASES[case]
    if with_observer:
        run_in.append('dump_observer observe 1 1 1 1')
    # dump_thermo follows dump_observer and writes the thermo vector after the observers have run.
    run_in += ['dump_thermo 1', 'run 20']
    (directory / 'run.in').write_text('\n'.join(run_in) + '\n')
    subprocess.run([gpumd_command], cwd=directory, check=True, stdout=subprocess.DEVNULL)
    return np.loadtxt(directory / 'thermo.out'), model_paths


@pytest.mark.parametrize('number_of_potentials', [1, 2])
@pytest.mark.parametrize('case', list(CASES))
def test_observers_leave_run_unchanged(tmp_path, gpumd_command, case, number_of_potentials):
    reference, _ = _run(
        tmp_path / 'reference', gpumd_command, case, number_of_potentials, with_observer=False)
    observed, model_paths = _run(
        tmp_path / 'observed', gpumd_command, case, number_of_potentials, with_observer=True)
    assert np.array_equal(observed, reference)

    for index, model_path in enumerate(model_paths):
        name = 'observer' if number_of_potentials == 1 else f'observer{index}'
        frame = read(tmp_path / 'observed' / f'{name}.xyz', index=-1)
        energy = frame.get_potential_energy()
        forces = frame.get_forces()
        stress = frame.get_stress(voigt=False)
        virial = frame.info['virial']
        velocities = frame.arrays['vel'] / units.fs  # from Å/fs to ASE units
        # calorine would write the velocities to model.xyz and add the kinetic term to the stress.
        del frame.arrays['vel']
        observer_thermo = np.loadtxt(tmp_path / 'observed' / f'{name}.out')

        frame.calc = GPUNEP(str(model_path), command=gpumd_command)
        assert energy == approx_tol(frame.get_potential_energy(), TOLERANCES['energy'])
        # column 2 of observer.out is the potential energy
        assert observer_thermo[-1, 2] == approx_tol(energy, TOLERANCES['energy'])
        assert np.max(np.abs(forces - frame.get_forces())) < OBSERVER_FORCE_TOLERANCE
        volume = frame.get_volume()
        assert -virial / volume == approx_tol(
            frame.get_stress(voigt=False), TOLERANCES['virial'])
        # The stress is the pressure tensor, which adds the kinetic tensor to the virial.
        kinetic = np.einsum('i,ia,ib->ab', frame.get_masses(), velocities, velocities)
        assert stress == approx_tol((virial + kinetic) / volume, TOLERANCES['virial'])


def test_observer_of_temperature_nep(tmp_path, gpumd_command):
    """The observer evaluates a temperature-dependent NEP at the temperature of the run."""
    fixtures = REPO_ROOT / 'tests_regression' / 'fixtures'
    shutil.copy(fixtures / 'potentials' / 'temperature_F_nep.txt', tmp_path)
    shutil.copy(fixtures / 'systems' / 'temperature_F_512.xyz', tmp_path / 'model.xyz')
    (tmp_path / 'run.in').write_text(
        'potential temperature_F_nep.txt\n'
        'velocity 300 seed 1\n'
        'ensemble nvt_ber 300 300 100\n'
        'time_step 1\n'
        'dump_observer observe 1 1 0 0\n'
        'dump_thermo 1\n'
        'run 3\n'
    )
    subprocess.run([gpumd_command], cwd=tmp_path, check=True, stdout=subprocess.DEVNULL)
    potential_energy = np.loadtxt(tmp_path / 'thermo.out')[:, 2]
    observer_potential_energy = np.loadtxt(tmp_path / 'observer.out')[:, 2]
    assert observer_potential_energy == approx_tol(potential_energy, TOLERANCES['energy'])
