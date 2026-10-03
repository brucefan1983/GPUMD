"""dump_observer observe: each observer holds its potential alone at the written positions, and
writing the observers leaves the molecular dynamics run unchanged.

Each case runs the same input with and without a dump_observer line and compares thermo.out.
dump_observer precedes dump_thermo, so thermo.out also shows whether the observer changed the thermo
vector of the step. add_force adds forces after the force evaluation of the step, npt_ber rescales
the box and the positions after it, and pimd propagates beads whose forces differ from those at the
written positions.
"""
import shutil
import subprocess

import numpy as np
import pytest
from ase.build import bulk
from ase.io import read
from calorine.calculators import GPUNEP
from calorine.gpumd import write_xyz

from conftest import MODELS_DIR, TOLERANCES, approx_tol

pytestmark = pytest.mark.slow

MODEL_PATH = MODELS_DIR / 'nep_C.txt'

# The forces of an observer and of a single-point run at the written positions differ by up to
# 2.1e-5 eV/Angstrom in these cases. Evaluating in the float box of the previous step under npt_ber
# gives 1.7e-3 eV/Angstrom.
OBSERVER_FORCE_TOLERANCE = dict(rtol=1e-4, atol=5e-5)

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
    """Writes nep_C.txt with every parameter scaled by 1.05, a second model for the same species."""
    lines = MODEL_PATH.read_text().splitlines()
    header_length = 6
    parameters = [f'{float(line) * 1.05:.7e}' for line in lines[header_length:]]
    path.write_text('\n'.join(lines[:header_length] + parameters) + '\n')


def _run(directory, gpumd_command, case, number_of_potentials, with_observer):
    directory.mkdir()
    structure = bulk('C', 'diamond', a=3.567, cubic=True).repeat((2, 2, 2))
    structure.rattle(stdev=0.05, seed=1)
    groups = [list(range(0, len(structure), 2)), list(range(1, len(structure), 2))]
    write_xyz(directory / 'model.xyz', structure, groupings=[groups])
    model_paths = [directory / MODEL_PATH.name, directory / 'nep_C_scaled.txt']
    shutil.copy(MODEL_PATH, model_paths[0])
    _write_scaled_model(model_paths[1])
    model_paths = model_paths[:number_of_potentials]
    run_in = [f'potential {path.name}' for path in model_paths] + CASES[case]
    if with_observer:
        run_in.append('dump_observer observe 1 1 0 1')
    run_in += ['dump_thermo 1', 'run 20']
    (directory / 'run.in').write_text('\n'.join(run_in) + '\n')
    subprocess.run([gpumd_command], cwd=directory, check=True, stdout=subprocess.DEVNULL)
    return np.loadtxt(directory / 'thermo.out'), model_paths


@pytest.mark.parametrize('number_of_potentials', [1, 2])
@pytest.mark.parametrize('case', list(CASES))
def test_dump_observer(tmp_path, gpumd_command, case, number_of_potentials):
    reference, _ = _run(tmp_path / 'reference', gpumd_command, case, number_of_potentials, False)
    observed, model_paths = _run(
        tmp_path / 'observed', gpumd_command, case, number_of_potentials, True)
    assert np.array_equal(observed, reference)

    for index, model_path in enumerate(model_paths):
        name = 'observer' if number_of_potentials == 1 else f'observer{index}'
        frame = read(tmp_path / 'observed' / f'{name}.xyz', index=-1)
        energy = frame.get_potential_energy()
        forces = frame.get_forces()
        observer_thermo = np.loadtxt(tmp_path / 'observed' / f'{name}.out')

        frame.calc = GPUNEP(str(model_path), command=gpumd_command)
        assert energy == approx_tol(frame.get_potential_energy(), TOLERANCES['energy'])
        assert observer_thermo[-1, 2] == approx_tol(energy, TOLERANCES['energy'])
        assert np.allclose(forces, frame.get_forces(), **OBSERVER_FORCE_TOLERANCE)
