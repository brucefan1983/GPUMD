"""Spin lattice checkpoint ownership, run-block lifetime and constrained restart."""
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest

from spin_analytic_model import analytic_model
from validate_spin_multigpu import structure, write_xyz
from spin_checkpoint_helpers import frames

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


def ensemble(kind):
    return ('nvt_glsd 250 450 100' if kind == 'nhc' else
            'npt_glsd temp 250 450 iso 0 .01 tperiod 100 pperiod 1000') + ' damping .01 seed 731'


def trajectory_commands(kind, steps, interval):
    return (f'ensemble {ensemble(kind)}\n'
            'dump_xyz 1 state.xyz precision double velocity force potential virial spin mforce\n'
            f'dump_restart {interval}\nrun {steps}\n')


def compare_trajectory(left, right, analytic):
    assert len(left) == len(right) and len(left) > 0
    tolerances = dict(pos=2e-7, vel=2e-8, spin=3e-5, force=5e-5,
                      mforce=5e-6, potential=5e-6, virial=5e-5, cell=2e-7, time=1e-7)
    for x, y in zip(left, right):
        for key, atol in tolerances.items():
            np.testing.assert_allclose(x[key], y[key], rtol=0, atol=1e-9 if analytic else atol)


@pytest.mark.parametrize('kind', ['nhc', 'mttk'])
@pytest.mark.parametrize('analytic', [True, pytest.param(False, marks=pytest.mark.slow)],
                         ids=['zero-force', 'nep'])
def test_lattice_restart_trajectory(tmp_path, kind, analytic):
    whole = prepare(tmp_path / 'whole')
    # Long x axis exercises domain decomposition when run on multiple GPUs.
    pos, spin, cell, velocity = structure((24, 4, 4))
    write_xyz(whole / 'model.xyz', pos, spin, cell, velocity)
    if not analytic:
        shutil.copyfile(Path(__file__).parent / 'fixtures/nep_spin3/nep_spin_o3c2_uniform.nep',
                        whole / 'nep.txt')
    run(whole, trajectory_commands(kind, 80, 43))
    expected = frames(whole / 'state.xyz')
    resumed = prepare(tmp_path / 'resumed', whole / 'restart.xyz')
    shutil.copyfile(whole / 'nep.txt', resumed / 'nep.txt')
    run(resumed, trajectory_commands(kind, 37, 37))
    compare_trajectory(expected[43:], frames(resumed / 'state.xyz'), analytic)
    assert np.array_equal(state(resumed, kind)[:3], [1, 80, 80])
    if not analytic:
        return
    # Repeated serialization and continuation past the ramp endpoint.
    first = prepare(tmp_path / 'first', whole / 'restart.xyz')
    run(first, trajectory_commands(kind, 20, 20))
    second = prepare(tmp_path / 'second', first / 'restart.xyz')
    run(second, trajectory_commands(kind, 17, 17))
    compare_trajectory(expected[63:], frames(second / 'state.xyz'), True)
    extended = prepare(tmp_path / 'extended', resumed / 'restart.xyz')
    run(extended, trajectory_commands(kind, 10, 10))
    long = prepare(tmp_path / 'long', whole / 'restart.xyz')
    run(long, trajectory_commands(kind, 47, 47))
    compare_trajectory(frames(long / 'state.xyz')[37:], frames(extended / 'state.xyz'), True)
    # A missing bath state must change the trajectory: otherwise the restart
    # comparison could pass without exercising restoration at all.
    missing = prepare(tmp_path / 'missing', whole / 'restart.xyz')
    path = missing / 'model.xyz'
    path.write_text(re.sub(kind + r'_(?:state|time)=[^\s]+\s*', '', path.read_text()))
    run(missing, trajectory_commands(kind, 37, 37))
    control = frames(missing / 'state.xyz')[-1]
    assert (np.max(np.abs(control['vel'] - expected[-1]['vel'])) > 1e-7 or
            np.max(np.abs(control['pos'] - expected[-1]['pos'])) > 1e-5)


@pytest.mark.parametrize('kind', ['nhc', 'mttk'])
def test_lattice_restart_rejects_invalid_state(tmp_path, kind):
    original = prepare(tmp_path / 'original')
    run(original, trajectory_commands(kind, 10, 6))
    source = (original / 'restart.xyz').read_text()
    def payload(index, value):
        match = re.search(kind + r'_state=([^\s]+)', source)
        values = match[1].split(','); values[index] = str(value)
        return source[:match.start(1)] + ','.join(values) + source[match.end(1):]
    variants = {
        'version': payload(0, 2), 'counter': payload(1, 1.5), 'nan': payload(-1, 'nan'),
        'truncated': re.sub(kind + r'_state=[^\s]+', kind + '_state=1,6,10', source),
        'duplicate': source.replace(kind + '_time=', kind + '_time=0 ' + kind + '_time=', 1),
        'missing-time': re.sub(kind + r'_time=[^\s]+\s*', '', source),
        'trailing-comma': re.sub('(' + kind + r'_state=[^\s]+)', r'\1,', source),
        'conflicting': source.replace(kind + '_state=', ('nhc' if kind == 'mttk' else 'mttk') +
                                      '_state=1 ' + kind + '_state=', 1),
    }
    if kind == 'nhc':
        variants.update(mass=payload(16, 0), atom_count=payload(4, 129))
    for name, text in variants.items():
        case = prepare(tmp_path / name)
        (case / 'model.xyz').write_text(text)
        (case / 'run.in').write_text('potential nep.txt\ntime_step .1\n' + trajectory_commands(kind, 1, 1))
        result = subprocess.run([str(GPUMD)], cwd=case, capture_output=True, text=True)
        assert result.returncode and 'Input Error' in result.stdout + result.stderr, name
    commands = trajectory_commands(kind, 1, 1)
    protocols = [('dt', 'time_step .2\n' + commands),
                 ('temperature', commands.replace('250 450', '250 451')),
                 ('coupling', commands.replace('250 450 100', '250 450 101') if kind == 'nhc'
                  else commands.replace('tperiod 100', 'tperiod 101'))]
    if kind == 'mttk':
        protocols += [('pressure', commands.replace('iso 0 .01', 'iso .1 .01')),
                      ('mode', commands.replace('iso', 'aniso'))]
    for name, commands in protocols:
        case = prepare(tmp_path / name, original / 'restart.xyz')
        (case / 'run.in').write_text('potential nep.txt\ntime_step .1\n' + commands)
        result = subprocess.run([str(GPUMD)], cwd=case, capture_output=True, text=True)
        assert result.returncode and 'Input Error' in result.stdout + result.stderr, name


@pytest.mark.parametrize('kind', ['nhc', 'mttk'])
def test_ordinary_driver_ignores_spin_bath(tmp_path, kind):
    original = prepare(tmp_path / 'original')
    run(original, trajectory_commands(kind, 10, 6))
    ordinary = prepare(tmp_path / 'ordinary', original / 'restart.xyz')
    (ordinary / 'nep.txt').write_text('lj 1 Fe\n.01 2.2 4.5\n')
    mode = 'nvt_nhc 250 450 100' if kind == 'nhc' else 'npt_mttk temp 250 450 iso 0 .01 tperiod 100 pperiod 1000'
    log = run(ordinary, f'ensemble {mode}\ndump_restart 3\nrun 3\n')
    assert state(ordinary, kind) is None
    assert 'Restored ' not in log
