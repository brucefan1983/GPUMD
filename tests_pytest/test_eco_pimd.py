"""Focused integration test for Eco-PIMD input and frequency optimisation."""

import re
import shutil
import subprocess

import numpy as np
import pytest

from conftest import MODELS_DIR, STRUCTURES_DIR

pytestmark = pytest.mark.fast


def test_eco_pimd_frequency_fit(tmp_path, gpumd_command):
    shutil.copy(STRUCTURES_DIR / 'C-nat16-rattled.xyz', tmp_path / 'model.xyz')
    shutil.copy(MODELS_DIR / 'nep_C.txt', tmp_path / 'nep.txt')
    (tmp_path / 'run.in').write_text(
        '\n'.join([
            'potential nep.txt',
            'velocity 300 seed 42',
            'time_step 0.5',
            'ensemble pimd 8 300 300 100 eco 3500',
            'dump_thermo 1',
            'run 1',
            '',
        ])
    )

    completed = subprocess.run(
        [gpumd_command],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(
        r'RMSE\(Trotter\)=([0-9.eE+-]+), RMSE\(Eco\)=([0-9.eE+-]+)',
        completed.stdout,
    )
    assert match is not None
    rmse_trotter, rmse_eco = map(float, match.groups())
    assert np.isfinite(rmse_trotter)
    assert np.isfinite(rmse_eco)
    assert rmse_eco < rmse_trotter
    assert np.all(np.isfinite(np.loadtxt(tmp_path / 'thermo.out')))


@pytest.mark.parametrize('ensemble', ['rpmd', 'trpmd'])
def test_eco_suffix_is_rejected_for_dynamics(tmp_path, gpumd_command, ensemble):
    shutil.copy(STRUCTURES_DIR / 'C-nat16-rattled.xyz', tmp_path / 'model.xyz')
    shutil.copy(MODELS_DIR / 'nep_C.txt', tmp_path / 'nep.txt')
    (tmp_path / 'run.in').write_text(
        '\n'.join([
            'potential nep.txt',
            'velocity 300 seed 42',
            'time_step 0.5',
            f'ensemble {ensemble} 8 eco 3500',
            'run 1',
            '',
        ])
    )

    completed = subprocess.run(
        [gpumd_command],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert f'ensemble {ensemble} should have 1 parameter.' in output
