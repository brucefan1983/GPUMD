"""Tests of the order of the *_train.out files that nep writes.

A training run with several batches reorders the structures before it cuts them into batches, and
every *_train.out file still lists them in the order of train.xyz, which is how a reader pairs a row
with its structure. Training writes the files every 1000 generations.
"""
import shutil
import subprocess

import numpy as np
import pytest
from ase.io import read

from conftest import TRAINING_DIR

pytestmark = pytest.mark.fast

KEYWORDS = {
    'type': '3 Ba Ti O',
    'cutoff': '6 4',
    'n_max': '4 4',
    'basis_size': '8 8',
    'l_max': '4 0 0',
    'neuron': '10',
    'population': '10',
    'generation': '1000',
    'output_interval': '500',
    'seed': '1',
}


@pytest.mark.parametrize('batch', ['2', '3'])
def test_train_outputs_of_a_training_run_follow_train_xyz(tmp_path, nep_command, batch):
    shutil.copy(TRAINING_DIR / 'train.xyz', tmp_path / 'train.xyz')
    keywords = dict(KEYWORDS, batch=batch)
    (tmp_path / 'nep.in').write_text(''.join(f'{key} {value}\n'
                                             for key, value in keywords.items()))
    result = subprocess.run(
        [nep_command], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr

    frames = read(tmp_path / 'train.xyz', index=':')
    energies = np.loadtxt(tmp_path / 'energy_train.out', ndmin=2)
    reference = [frame.get_potential_energy() / len(frame) for frame in frames]
    assert np.allclose(energies[:, 1], reference, atol=1e-4)
    forces = np.loadtxt(tmp_path / 'force_train.out', ndmin=2)
    assert np.allclose(forces[:, 3:6], np.vstack([frame.arrays['force'] for frame in frames]),
                       atol=1e-4)
