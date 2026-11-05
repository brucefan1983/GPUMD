"""Spin restart files preserve physical state and initialize a new run."""
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest

from spin_analytic_model import analytic_model
from validate_spin_multigpu import frame, structure, write_xyz

GPUMD = Path(os.environ.get('GPUMD_COMMAND', Path(__file__).parents[1] / 'src/gpumd')).resolve()


def run(path, ensemble, steps, dt, spin_velocity):
    extra = ' spin_velocity' if spin_velocity else ''
    (path / 'run.in').write_text(
        f'potential nep.txt\ntime_step {dt}\nensemble {ensemble}\n'
        f'dump_xyz {steps} state.xyz precision double velocity spin{extra}\n'
        f'dump_restart {steps}\nrun {steps}\n')
    result = subprocess.run([str(GPUMD)], cwd=path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    state = frame(path / 'state.xyz')
    assert all(np.isfinite(value).all() for value in state.values())
    return state


@pytest.mark.parametrize('mode', ['tspin', 'sib', 'glsd'])
@pytest.mark.parametrize('kind', ['nvt', 'npt'])
def test_restart_preserves_physical_state(tmp_path, mode, kind):
    original = tmp_path / 'original'
    original.mkdir()
    (original / 'nep.txt').write_text(analytic_model(False))
    write_xyz(original / 'model.xyz', *structure((4, 4, 4)))
    prefix = (f'nvt_{mode} 300 350 100' if kind == 'nvt' else
              f'npt_{mode} temp 300 350 iso 0 .01 tperiod 100 pperiod 1000')
    options = {'tspin': 'mass_factor 1', 'sib': 'alpha .01', 'glsd': 'damping .01'}[mode]
    state = run(original, f'{prefix} {options} seed 731', 3, .1, mode == 'tspin')
    checkpoint = frame(original / 'restart.xyz')
    keys = ['pos', 'vel', 'spin'] + (['spin_vel'] if mode == 'tspin' else [])
    for key in keys:
        np.testing.assert_allclose(checkpoint[key], state[key], rtol=0, atol=1e-14)
    header = (original / 'restart.xyz').read_text().splitlines()[1]
    assert not any(key in header for key in ('nhc_state=', 'mttk_state=', 'spin_glsd_', 'Time='))
    if mode != 'tspin':
        assert 'spin_vel:' not in header
    def cell(path):
        return np.fromstring(re.search(r'Lattice="([^"]+)"', path.read_text())[1], sep=' ')
    np.testing.assert_allclose(cell(original / 'restart.xyz'), cell(original / 'state.xyz'),
                               rtol=0, atol=1e-12)

    resumed = tmp_path / 'resumed'
    resumed.mkdir()
    shutil.copyfile(original / 'nep.txt', resumed / 'nep.txt')
    shutil.copyfile(original / 'restart.xyz', resumed / 'model.xyz')
    # A restart starts a new bath/protocol, so new targets, timestep and seed
    # are allowed without hidden configuration or clock restoration.
    run(resumed, f'{prefix.replace("300 350", "250 300")} {options} seed 913',
        2, .2, mode == 'tspin')
    resumed_header = (resumed / 'state.xyz').read_text().splitlines()[1]
    assert float(re.search(r'Time=([^\s]+)', resumed_header)[1]) == pytest.approx(.4)
