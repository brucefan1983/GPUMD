"""Nonmagnetic geometric environments remain zero under every Spin driver."""
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from spin_fixtures import load_fixture, write_xyz
from validate_spin_multigpu import frame

GPUMD = Path(os.environ.get('GPUMD_COMMAND', Path(__file__).parents[1] / 'src/gpumd'))


def execute(directory, ensemble, steps, restart=None, raw_inactive=False):
    directory.mkdir()
    (directory / 'nep.txt').write_text(load_fixture('c3l2s2_fege')[0])
    if restart:
        shutil.copyfile(restart, directory / 'model.xyz')
    else:
        _, oracle = load_fixture('c3l2s2_fege')
        write_xyz(directory / 'model.xyz', oracle['cases']['noncollinear'])
        path = directory / 'model.xyz'
        lines = path.read_text().splitlines()
        lines[1] = lines[1].replace(':spin:R:3', ':vel:R:3:spin:R:3:spin_vel:R:3')
        for i in range(2, len(lines)):
            fields = lines[i].split(); lines[i] = ' '.join(fields[:4] + ['0']*3 + fields[4:] + ['0']*3)
        path.write_text('\n'.join(lines) + '\n')
    if raw_inactive:
        path = directory / 'model.xyz'
        lines = path.read_text().splitlines()
        properties = next(x for x in lines[1].split() if x.startswith('Properties='))
        fields = properties.split('=', 1)[1].split(':')
        offsets, offset = {}, 0
        for i in range(0, len(fields), 3):
            offsets[fields[i]] = offset
            offset += int(fields[i+2])
        if 'spin_vel' not in offsets:
            lines[1] = lines[1].replace(properties, properties + ':spin_vel:R:3')
            offsets['spin_vel'] = offset
            lines[2:] = [line + ' 0 0 0' for line in lines[2:]]
        for i in range(2, len(lines)):
            values = lines[i].split()
            if values[0] == 'Ge':
                for name in ('spin', 'spin_vel'):
                    start = offsets[name]
                    values[start:start+3] = ['0.4', '-0.3', '0.2']
                lines[i] = ' '.join(values)
        path.write_text('\n'.join(lines) + '\n')
    original = (directory / 'model.xyz').read_bytes()
    spin_velocity = ' spin_velocity' if 'tspin' in ensemble else ''
    (directory / 'run.in').write_text(
        'potential nep.txt\n' + ensemble + '\n'
        'add_mfield region 0 1 0 1 0 1 12 -7 31\n'
        'time_step 0.01\n'
        f'dump_xyz 1 state.xyz precision double velocity force spin mforce{spin_velocity}\n'
        f'dump_restart {steps}\nrun {steps}\n')
    result = subprocess.run([str(GPUMD)], cwd=directory, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    if restart and 'tspin' in ensemble:
        assert 'Reuse initialized spin velocities' in result.stdout
    assert (directory / 'model.xyz').read_bytes() == original
    state = frame(directory / 'state.xyz')
    rows = (directory / 'model.xyz').read_text().splitlines()[2:]
    inactive = np.array([row.split()[0] == 'Ge' for row in rows])
    assert np.any(inactive), 'test requires nonmagnetic atoms'
    for name in ('spin', 'mforce') + (('spin_vel',) if spin_velocity else ()):
        assert np.isfinite(state[name]).all()
        assert np.count_nonzero(state[name][inactive]) == 0, (name, state[name][inactive])
    assert np.max(np.abs(state['force'][inactive])) > 1e-8
    return state


@pytest.mark.parametrize('ensemble', [
    'ensemble nvt_tspin 300 300 100 lattice off seed 731',
    'ensemble nve_sib alpha 0.1 stemp 300 seed 731',
    'ensemble nve_glsd damping 0.01 stemp 300 seed 731',
])
def test_nonmagnetic_environment_and_restart(tmp_path, ensemble):
    full = execute(tmp_path / 'full', ensemble, 8)
    first = execute(tmp_path / 'first', ensemble, 4)
    resumed = execute(tmp_path / 'resumed', ensemble, 4, tmp_path / 'first/restart.xyz')
    raw = execute(tmp_path / 'raw', ensemble, 8, raw_inactive=True)
    raw_resumed = execute(tmp_path / 'raw_resumed', ensemble, 4,
                          tmp_path / 'first/restart.xyz', raw_inactive=True)
    for expected, actual in ((full, raw), (resumed, raw_resumed)):
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key], err_msg=key)
    # Restart stores coordinates/velocities at full double precision. Noise
    # counters and thermostat chains are initialized by the new ensemble.
    saved = frame(tmp_path / 'first/restart.xyz')
    for name in ('spin', 'spin_vel') if 'tspin' in ensemble else ('spin',):
        np.testing.assert_array_equal(first[name], saved[name])


def test_nonmagnetic_geometry_has_magnetic_atomic_force(tmp_path):
    from copy import deepcopy
    from spin_fixtures import load_fixture
    from spin_force_helpers import run_case
    model, oracle = load_fixture('c3l2s2_fege')
    case = deepcopy(oracle['cases']['noncollinear'])
    atom = case['types'].index('Ge')
    state = run_case(tmp_path, 'center', case, model)
    zero = deepcopy(case); zero['spins'] = [[0, 0, 0] for _ in case['types']]
    nonmagnetic = run_case(tmp_path, 'zero', zero, model)
    assert max(abs(a-b) for a,b in zip(state['force'][atom], nonmagnetic['force'][atom])) > 1e-5
    h = 0.002
    energies = []
    for direction in (-1, 1):
        moved = deepcopy(case); moved['positions'][atom][0] += direction * h
        energies.append(run_case(tmp_path, str(direction), moved, model)['energy'])
    numerical = -(energies[1]-energies[0]) / (2*h)
    assert abs(numerical-state['force'][atom][0]) < 2e-4
