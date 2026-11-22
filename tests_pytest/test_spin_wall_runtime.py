"""Public-CLI checks for longitudinal walls; requires CUDA binaries."""
import math

import pytest

from spin_analytic_model import analytic_model
from spin_force_helpers import GPUMD, run_case
from validate_spin_model import INPUT, training_xyz
from validate_spin_training import NEP, TRAIN_XYZ
from validate_spin_training_lifecycle import CONFIG, execute, metadata, predict


SPINS = [[3, 4, 0], [0, 1, 0], [0, 0, 0], [3, 4, 0]]
WALLS = 'spin_wall 2 1.5 4 3  0 0 0 0\n'
MODEL = analytic_model(False).replace('spin_type Fe Ge\n', 'spin_type Fe\n' + WALLS)
ENERGY = [1, 0.75**3, 1, 0]
MF = [[-1.8, -2.4, 0], [0, 1.5 * 0.75**2, 0], [0, 0, 0], [0, 0, 0]]


def geometry(length):
    zeros = [[0] * 3 for _ in SPINS]
    return dict(types=['Fe', 'Fe', 'Fe', 'Ge'], spins=SPINS,
                positions=[[1, 1, 1], [3, 2, 1], [2, 1, 2], [1, 2, 1]],
                cell=[length, 0, 0, 0, length, 0, 0, 0, length],
                energy=0, force=zeros, mforce=zeros, virial=[0] * 9)


@pytest.mark.skipif(not GPUMD.is_file(), reason='GPUMD executable unavailable')
@pytest.mark.parametrize('length', [8, 24])
def test_md_wall(tmp_path, length):
    got = run_case(tmp_path, 'md', geometry(length), MODEL)
    assert got['potential'] == pytest.approx(ENERGY, abs=1e-12)
    for vector, expected in zip(got['mforce'], MF):
        assert vector == pytest.approx(expected, abs=1e-12)
    assert not any(v for row in got['force'] + got['atom_virial'] for v in row)


@pytest.mark.skipif(not NEP.is_file(), reason='NEP executable unavailable')
@pytest.mark.parametrize('compile_', [0, 1])
def test_prediction_wall(tmp_path, compile_):
    config = INPUT.format(C=3, L=2, S=2).replace('type 3 Fe Bi O', 'type 2 Fe Ge')
    config += f'nep_compile {compile_}\n'
    case = tmp_path / 'predict'
    xyz = training_xyz(geometry(24))
    got = predict(case, MODEL, xyz, config)
    if compile_:
        log = (case / 'run.log').read_text()
        assert 'Spin aggregate JIT ready' in log
    assert got['energy'][0][0] == pytest.approx(sum(ENERGY) / len(SPINS), abs=1e-6)
    for vector, expected in zip(got['mforce'], MF):
        assert vector == pytest.approx(expected, abs=1e-6)
    assert not any(v for row in got['force'] + got['virial'] for v in row)

    config += 'spin_wall 0 0 0 0\n'
    control = predict(tmp_path / 'disabled', MODEL, xyz, config)
    assert not any(v for outputs in control.values() for row in outputs for v in row)


@pytest.mark.skipif(not NEP.is_file(), reason='NEP executable unavailable')
@pytest.mark.parametrize('per_element', [False, True])
def test_auto_wall_export_and_restart(tmp_path, per_element):
    xyz = TRAIN_XYZ
    lower = math.hypot(.8, .1) - .2
    upper = math.hypot(1, .1) + .5
    if per_element:
        config = CONFIG.replace('spin_type Fe', 'spin_type Fe Ge')
        config += 'spin_wall auto 0.2 1.5  2 3   0 0  auto 0.7 2\n'
        xyz = xyz.replace('0.0 0.0 0.0 9.0 9.0 9.0', '0.3 0.0 0.0 9.0 9.0 9.0')
        expected = [lower, 1.5, 2, 3, 0, 0, 1, 2]
    else:
        config = CONFIG + 'spin_wall auto 0.2 1.5  auto 0.5 2\n'
        expected = [lower, 1.5, upper, 2, 0, 0, 0, 0]
    fresh = tmp_path / 'fresh'
    execute(fresh, config, xyz, test_xyz=training_xyz(geometry(24)))
    model = (fresh / 'nep.txt').read_text()
    checkpoint = (fresh / 'nep.restart').read_text()
    exported = metadata(model)['spin_wall']
    assert list(map(float, exported)) == pytest.approx(expected)

    resumed = tmp_path / 'resumed'
    shifted = xyz.replace('1.0 0.1 0.0 0.2 -0.1 0.05', '2.0 0.1 0.0 0.2 -0.1 0.05')
    if per_element:
        shifted = shifted.replace('0.3 0.0 0.0 9.0 9.0 9.0', '0.5 0.0 0.0 9.0 9.0 9.0')
    execute(resumed, config, shifted, model, checkpoint)
    assert metadata((resumed / 'nep.txt').read_text())['spin_wall'] == exported


@pytest.mark.skipif(not NEP.is_file(), reason='NEP executable unavailable')
@pytest.mark.parametrize('walls, error', [
    ('spin_wall 0 0 auto 0.5', 'spin_wall requires a lower and an upper wall'),
    ('spin_wall 0 0 0 0  0 0 auto 0.5 2', 'only be enabled for active spin types'),
    ('spin_wall auto 0 1  0.5 1', 'lower threshold must be below'),
    ('spin_wall 0 0 0 0\nspin_wall 0 0 0 0', 'Duplicate or incomplete spin_wall'),
])
def test_invalid_wall_input(tmp_path, walls, error):
    execute(tmp_path / 'invalid', CONFIG + walls + '\n', error=error)
