"""Tests of the forces that dump_observer writes in observe mode.

The observers are evaluated at the end of each step, after the barostat has changed the box.
Their forces then match a single-point evaluation of the frame they are written with.
"""
import subprocess

import numpy as np
import pytest
from ase.build import bulk
from ase.io import read
from calorine.calculators import GPUNEP
from calorine.gpumd import write_xyz

from conftest import MODELS_DIR

pytestmark = pytest.mark.fast

MODEL_PATH = MODELS_DIR / 'nep_C.txt'


# The radial cutoff of 7 Å puts the threshold of the small-box path at a thickness of 20 Å.
# The 2x2x2 cell takes the small-box path with two images per direction.
# The 6x6x6 cell takes the large-box path.
@pytest.mark.parametrize('repeat', [2, 6], ids=['small_box', 'large_box'])
@pytest.mark.parametrize(
    'ensemble', ['nve', 'npt_ber 1500 1500 100 0 0 0 500 500 500 200'], ids=['nve', 'npt_ber']
)
def test_observer_forces_match_single_point(tmp_path, gpumd_command, repeat, ensemble):
    atoms = bulk('C', 'diamond', a=3.567, cubic=True).repeat(repeat)
    atoms.rattle(0.05, seed=1)
    write_xyz(str(tmp_path / 'model.xyz'), atoms)
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
    # the largest difference is 2e-5 eV/Å in the box of the step, 1.7e-3 eV/Å (2x2x2) and
    # 3e-2 eV/Å (6x6x6) with npt_ber in the box of the previous step
    assert np.max(np.abs(observer_forces - frame.get_forces())) < 2e-4
