"""dump_observer observe: writing the observers leaves the trajectory of the driving potential
unchanged.

Each case runs the same input with and without a dump_observer line and compares thermo.out.
add_force adds forces after the force evaluation of the step, and npt_ber rescales the box and the
positions after it. Both expose an observer that writes into the per-atom arrays of the run.
"""
import shutil
import subprocess

import numpy as np
import pytest
from ase.build import bulk
from calorine.gpumd import write_xyz

from conftest import MODELS_DIR

pytestmark = pytest.mark.slow

MODEL_PATH = MODELS_DIR / 'nep_C.txt'

CASES = {
    'add_force': [
        'velocity 300 seed 1',
        'ensemble nve',
        'time_step 1',
        'add_force 0 0 0.5 0 0',
        'add_force 0 1 -0.5 0 0',
        'dump_thermo 1',
        'run 50',
    ],
    'npt_ber': [
        'velocity 1500 seed 7',
        'ensemble npt_ber 1500 1500 100 0 0 0 500 500 500 200',
        'time_step 1',
        'dump_thermo 1',
        'run 20',
    ],
}


def _run(directory, gpumd_command, lines, number_of_potentials, with_observer):
    directory.mkdir()
    structure = bulk('C', 'diamond', a=3.567, cubic=True).repeat((2, 2, 2))
    structure.rattle(stdev=0.05, seed=1)
    groups = [list(range(0, len(structure), 2)), list(range(1, len(structure), 2))]
    write_xyz(directory / 'model.xyz', structure, groupings=[groups])
    shutil.copy(MODEL_PATH, directory / MODEL_PATH.name)
    run_in = [f'potential {MODEL_PATH.name}'] * number_of_potentials
    run_in += lines[:-1]
    if with_observer:
        run_in.append('dump_observer observe 1 1 0 1')
    run_in.append(lines[-1])
    (directory / 'run.in').write_text('\n'.join(run_in) + '\n')
    subprocess.run(
        [gpumd_command], cwd=directory, check=True, stdout=subprocess.DEVNULL)
    return np.loadtxt(directory / 'thermo.out')


@pytest.mark.parametrize('number_of_potentials', [1, 2])
@pytest.mark.parametrize('case', list(CASES))
def test_dump_observer_leaves_trajectory_unchanged(
        tmp_path, gpumd_command, case, number_of_potentials):
    reference = _run(
        tmp_path / 'reference', gpumd_command, CASES[case], number_of_potentials, False)
    observed = _run(
        tmp_path / 'observed', gpumd_command, CASES[case], number_of_potentials, True)
    assert np.array_equal(observed, reference)

    # observer0 holds the state of the step, which thermo.out holds as well.
    observer0_name = 'observer.out' if number_of_potentials == 1 else 'observer0.out'
    observer0 = np.loadtxt(tmp_path / 'observed' / observer0_name)
    assert np.array_equal(observer0, observed)
