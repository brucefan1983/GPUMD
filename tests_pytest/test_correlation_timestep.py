"""Physical clocks for MSD, VAC integration, and DOS under an adaptive step cap."""
import shutil
import subprocess

import numpy as np
import pytest
from ase.io import read

from conftest import FIXTURES_DIR

NC = 16
VELOCITIES = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]]) / np.sqrt(3)
POSITIONS = np.array([[4, 4, 4], [10, 4, 4], [4, 10, 4], [4, 4, 10]])
OUTPUTS = {'msd': ('msd.out',), 'sdc': ('sdc.out',), 'dos': ('mvac.out', 'dos.out')}
# Correlation files use %g (six significant digits).
OUTPUT_TOL = dict(rtol=5e-6, atol=1e-10)


def _prepare(path, force_bearing=False):
    path.mkdir(parents=True, exist_ok=True)
    if force_bearing:
        original = (FIXTURES_DIR / 'structures/C-nat16-rattled.xyz').read_text().splitlines()
        rows = [original[0], original[1].replace('Properties=species:S:1:pos:R:3',
                                               'Properties=species:S:1:pos:R:3:vel:R:3')]
        rows += [row + ' ' + ' '.join(f'{v:.17g}' for v in VELOCITIES[i % 4] * 0.01)
                 for i, row in enumerate(original[2:]) if row.strip()]
        model = FIXTURES_DIR / 'models/nep_C.txt'
    else:
        rows = ['4', 'pbc="T T T" Lattice="30 0 0 0 30 0 0 0 30" '
                'Properties=species:S:1:pos:R:3:vel:R:3:group:I:1']
        rows += ['C ' + ' '.join(f'{v:.17g}' for v in [*pos, *vel]) + f' {i // 2}'
                 for i, (pos, vel) in enumerate(zip(POSITIONS, VELOCITIES))]
        model = FIXTURES_DIR / 'replica/nep.txt'  # Zero energy and forces.
    (path / 'model.xyz').write_text('\n'.join(rows) + '\n')
    shutil.copyfile(model, path / 'nep.txt')


def _commands(command, sample=1, options=''):
    names = OUTPUTS if command == 'all' else (command,)
    return ''.join(f'compute_{name} {sample} {NC}' + (' 100' if name == 'dos' else '')
                   + options + '\n' for name in names)


def _run(path, executable, body, success=True):
    (path / 'run.in').write_text('potential nep.txt\nensemble nve\n' + body)
    result = subprocess.run([executable], cwd=path, capture_output=True, text=True, timeout=120)
    (path / 'stdout.log').write_text(result.stdout)
    (path / 'stderr.log').write_text(result.stderr)
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    return result


def _oracle(path, command, dt, sample):
    lag_ps = np.arange(NC) * dt * sample / 1000
    if command in ('msd', 'all'):
        data = np.loadtxt(path / 'msd.out')
        msd = np.repeat((lag_ps * 1000)[:, None] ** 2 / 3, 3, axis=1)
        derivative = np.zeros_like(msd)
        derivative[1:] = np.diff(msd, axis=0) / (2 * dt * sample / 1000)
        np.testing.assert_allclose(data, np.column_stack((lag_ps, msd, derivative)), **OUTPUT_TOL)
    if command in ('sdc', 'all'):
        data = np.loadtxt(path / 'sdc.out')
        vac = np.full((NC, 3), 1e6 / 3)
        np.testing.assert_allclose(data, np.column_stack((lag_ps, vac, vac * lag_ps[:, None])),
                                   **OUTPUT_TOL)
    if command in ('dos', 'all'):
        np.testing.assert_allclose(np.loadtxt(path / 'mvac.out'),
                                   np.column_stack((lag_ps, np.ones((NC, 3)))), **OUTPUT_TOL)
        # Constant-velocity normalized VAC is one in each direction. Evaluate its
        # Hann-windowed cosine transform independently of the GPU accumulator.
        omega = np.arange(1, NC + 1) * 100 / NC
        window = (1 + np.cos(np.pi * np.arange(NC) / NC)) / 2
        weights = window * 2
        weights[0] = 1
        spectrum = np.cos(omega[:, None] * lag_ps) @ weights * (dt * sample / 1000) * 8
        np.testing.assert_allclose(np.loadtxt(path / 'dos.out'),
                                   np.column_stack((omega, np.repeat(spectrum[:, None], 3, axis=1))),
                                   **OUTPUT_TOL)


@pytest.mark.parametrize('command', ['msd', 'sdc', 'dos', 'all'])
@pytest.mark.parametrize('sample', [1, 2])
@pytest.mark.parametrize('fixed,adaptive', [('0.01', '1 0.01'), ('0.1', '1 0.1'), ('1', '1 2')])
def test_uniform_adaptive_matches_fixed(tmp_path, gpumd_command, command, sample, fixed, adaptive):
    for name, step in [('fixed', fixed), ('adaptive', adaptive)]:
        path = tmp_path / name
        _prepare(path)
        _run(path, gpumd_command, f'time_step {step}\n' + _commands(command, sample)
             + 'dump_xyz 1 trajectory.xyz unwrapped_position velocity precision double\nrun 64\n')
        _oracle(path, command, float(fixed), sample)
        frames = read(path / 'trajectory.xyz', index=':')
        times = np.array([a.info['Time'] for a in frames])
        np.testing.assert_allclose(times, np.arange(1, 65) * float(fixed), atol=1e-9, rtol=0)
        np.testing.assert_allclose([a.arrays['unwrapped_position'] for a in frames],
                                   POSITIONS + times[:, None, None] * VELOCITIES, atol=1e-10, rtol=0)
    for name in (OUTPUTS if command == 'all' else (command,)):
        for filename in OUTPUTS[name]:
            np.testing.assert_allclose(np.loadtxt(tmp_path / 'adaptive' / filename),
                                       np.loadtxt(tmp_path / 'fixed' / filename), **OUTPUT_TOL)


@pytest.mark.parametrize('command', ['msd', 'sdc', 'dos'])
@pytest.mark.parametrize('sample', [1, 4])
def test_changing_step_is_rejected(tmp_path, gpumd_command, command, sample):
    _prepare(tmp_path, force_bearing=True)
    result = _run(tmp_path, gpumd_command, 'time_step 1 0.0005\n' + _commands(command, sample)
                  + 'dump_xyz 1 trajectory.xyz velocity precision double\nrun 64\n', success=False)
    assert f'compute_{command} requires a constant actual time step' in result.stderr
    for filename in OUTPUTS[command]:
        assert not (tmp_path / filename).exists()


@pytest.mark.parametrize('command', ['msd', 'sdc', 'dos', 'all'])
def test_new_run_uses_its_own_interval(tmp_path, gpumd_command, command):
    _prepare(tmp_path)
    body = ''
    # Commands create a fresh action per run; adaptive limits also reset per run.
    for step in ('1 0.01', '0.1', '1 0.02'):
        body += ('ensemble nve\n' if body else '') + f'time_step {step}\n'
        body += _commands(command) + 'run 64\n'
    _run(tmp_path, gpumd_command, body)
    for name in (OUTPUTS if command == 'all' else (command,)):
        filename = OUTPUTS[name][0]
        data = np.loadtxt(tmp_path / filename).reshape(3, NC, -1)
        for block, dt in zip(data, (0.01, 0.1, 0.02)):
            np.testing.assert_allclose(block[:, 0], np.arange(NC) * dt / 1000, **OUTPUT_TOL)


def test_dos_nyquist_uses_actual_interval(tmp_path, gpumd_command):
    _prepare(tmp_path)
    # 100 fs nominal violates Nyquist for omega_max=100, but 0.01 fs actual is valid.
    _run(tmp_path, gpumd_command, 'time_step 100 0.01\n' + _commands('dos') + 'run 64\n')
    _oracle(tmp_path, 'dos', 0.01, 1)


def test_dos_invalid_fixed_nyquist_is_still_rejected(tmp_path, gpumd_command):
    _prepare(tmp_path)
    result = _run(tmp_path, gpumd_command, 'time_step 100\n' + _commands('dos') + 'run 64\n',
                  success=False)
    assert 'Velocity sampling rate < Nyquist frequency.' in result.stderr
    assert not (tmp_path / 'dos.out').exists()


@pytest.mark.parametrize('command,options', [('msd', ' all_groups 0 save_every 32'),
                                           ('sdc', ' group 0 1'), ('dos', ' group 0 -1')])
def test_grouped_correlations_use_actual_interval(tmp_path, gpumd_command, command, options):
    for name, step in [('fixed', '0.01'), ('adaptive', '1 0.01')]:
        path = tmp_path / name
        _prepare(path)
        _run(path, gpumd_command, f'time_step {step}\n' + _commands(command, 2, options) + 'run 64\n')
    filenames = list(OUTPUTS[command])
    if command == 'msd':
        filenames += ['msd_step32.out', 'msd_step64.out']
    for filename in filenames:
        a = np.loadtxt(tmp_path / 'adaptive' / filename)
        b = np.loadtxt(tmp_path / 'fixed' / filename)
        np.testing.assert_allclose(a, b, **OUTPUT_TOL)
        if filename != 'dos.out':
            np.testing.assert_allclose(a[:NC, 0], np.arange(NC) * 0.02 / 1000, **OUTPUT_TOL)


def test_variable_step_without_correlations_remains_supported(tmp_path, gpumd_command):
    _prepare(tmp_path, force_bearing=True)
    _run(tmp_path, gpumd_command, 'time_step 1 0.0005\n'
         'dump_xyz 1 trajectory.xyz velocity precision double\nrun 64\n')
    times = np.array([a.info['Time'] for a in read(tmp_path / 'trajectory.xyz', index=':')])
    assert np.ptp(np.diff(times)) > 1e-4


@pytest.mark.parametrize('command', ['msd', 'sdc', 'dos'])
def test_step_change_after_correlation_has_started_is_rejected(tmp_path, gpumd_command, command):
    _prepare(tmp_path)
    result = _run(tmp_path, gpumd_command, 'time_step 1 1.01\nadd_force 0 0 1 0 0\n'
                  + _commands(command) + 'dump_xyz 1 trajectory.xyz velocity precision double\nrun 64\n',
                  success=False)
    assert f'compute_{command} requires a constant actual time step' in result.stderr
    frames = read(tmp_path / 'trajectory.xyz', index=':')
    assert NC <= len(frames) < 64
    np.testing.assert_allclose([a.info['Time'] for a in frames], np.arange(1, len(frames) + 1),
                               rtol=0, atol=1e-8)
