"""Spin lattice checkpoint ownership, run-block lifetime and constrained restart."""
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest

from validate_spin_fire_runtime import analytic_model
from validate_spin_multigpu import structure, write_xyz
from validate_mttk_restart import frames

GPUMD = Path(os.environ.get('GPUMD_COMMAND', Path(__file__).parents[1] / 'src/gpumd')).resolve()


def prepare(path, source=None):
    path.mkdir()
    (path / 'nep.txt').write_text(analytic_model(False))
    if source is not None:
        shutil.copyfile(source, path / 'model.xyz')
    else:
        pos, spin, cell, velocity = structure((4, 4, 4))
        write_xyz(path / 'model.xyz', pos, spin, cell, velocity)
    return path


def run(path, commands):
    (path / 'run.in').write_text('potential nep.txt\ntime_step 0.1\n' + commands)
    result = subprocess.run([str(GPUMD)], cwd=path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def state(path, kind):
    header = (path / 'restart.xyz').read_text().splitlines()[1]
    payload = re.search(kind + r'_state=([^\s]+)', header)
    return np.array(payload[1].split(','), float) if payload else None


@pytest.mark.parametrize('mode', ['glsd', 'sib', 'tspin'])
@pytest.mark.parametrize('kind', ['nhc', 'mttk'])
def test_spin_lattice_checkpoint_consumed_once(tmp_path, mode, kind):
    prefix = 'nvt_' + mode + ' 300 350 100' if kind == 'nhc' else (
        'npt_' + mode + ' temp 300 350 iso 0 .01 tperiod 100 pperiod 1000')
    options = {'glsd': ' damping .01', 'sib': ' alpha .01', 'tspin': ' mass_factor 1'}[mode]
    ensemble = prefix + options + ' seed 731'
    original = prepare(tmp_path / 'original')
    run(original, f'ensemble {ensemble}\ndump_restart 6\nrun 10\n')
    assert np.array_equal(state(original, kind)[:3], [1, 6, 10])
    resumed = prepare(tmp_path / 'resumed', original / 'restart.xyz')
    log = run(resumed, f'ensemble {ensemble}\ndump_restart 4\nrun 4\n'
              f'ensemble {ensemble}\ndump_restart 3\nrun 3\n')
    # A new run block creates a fresh lattice protocol, without consuming the old
    # endpoint again or resetting the execution clock to the loaded checkpoint.
    assert log.count('Restored ') == 1
    assert np.array_equal(state(resumed, kind)[:3], [1, 3, 3])
    header = (resumed / 'restart.xyz').read_text().splitlines()[1]
    clock = float(re.search(kind + r'_time=([^\s]+)', header)[1])
    assert abs(clock - 13 * .1 / 10.18051) < 1e-12
    switched = prepare(tmp_path / 'switched', original / 'restart.xyz')
    other = 'nvt_glsd 300 350 100 damping .01 seed 731' if kind == 'mttk' else (
        'npt_glsd temp 300 350 iso 0 .01 damping .01 seed 731')
    log = run(switched, f'ensemble {other}\ndump_restart 3\nrun 3\n')
    assert state(switched, kind) is None
    assert 'Discarding ' in log and 'Restored ' not in log


def test_spin_nhc_fixed_moving_group_restart(tmp_path):
    original = prepare(tmp_path / 'original')
    path = original / 'model.xyz'
    lines = path.read_text().splitlines()
    for i in range(2, len(lines)):
        lines[i] = lines[i].rsplit(' ', 1)[0] + ' ' + str(min(i - 2, 2))
    path.write_text('\n'.join(lines) + '\n')
    commands = ('ensemble nvt_glsd 300 350 100 damping .01 seed 731\n'
                'fix 0\nmove 1 .001 0 0\n'
                'dump_xyz 1 state.xyz precision double velocity force spin mforce\n')
    run(original, commands + 'dump_restart 6\nrun 10\n')
    resumed = prepare(tmp_path / 'resumed', original / 'restart.xyz')
    run(resumed, commands + 'dump_restart 4\nrun 4\n')
    left, right = frames(original / 'state.xyz')[6:], frames(resumed / 'state.xyz')
    assert len(left) == len(right) == 4
    for x, y in zip(left, right):
        for key in ('pos', 'vel', 'spin', 'force', 'mforce', 'time'):
            assert np.max(np.abs(x[key] - y[key])) < 1e-9, key
