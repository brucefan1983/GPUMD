"""Public-CLI checks for longitudinal walls; requires CUDA binaries."""
import math

import pytest

from spin_analytic_model import analytic_model
from spin_force_helpers import GPUMD, run_case
from validate_spin_model import INPUT, training_xyz
from validate_spin_training import NEP, TRAIN_XYZ
from validate_spin_training_lifecycle import CONFIG, execute, metadata, predict


SPINS = [[3, 4, 0], [0, 1, 0], [0, 0, 0], [3, 4, 0]]
WALLS = 'spin_wall_high 4 3 0 0\nspin_wall_low 2 1.5 0 0\n'
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

    config += 'spin_wall_high 0 0\nspin_wall_low 0 0\n'
    control = predict(tmp_path / 'disabled', MODEL, xyz, config)
    assert not any(v for outputs in control.values() for row in outputs for v in row)


@pytest.mark.skipif(not NEP.is_file(), reason='NEP executable unavailable')
def test_auto_wall_export_and_restart(tmp_path):
    config = CONFIG + 'spin_wall_high auto 0.5 2\nspin_wall_low auto 0.2 1.5\n'
    fresh = tmp_path / 'fresh'
    execute(fresh, config, test_xyz=training_xyz(geometry(24)))
    model = (fresh / 'nep.txt').read_text()
    checkpoint = (fresh / 'nep.restart').read_text()
    exported = metadata(model)
    assert list(map(float, exported['spin_wall_high'])) == pytest.approx(
        [math.hypot(1, .1) + .5, 2, 0, 0])
    assert list(map(float, exported['spin_wall_low'])) == pytest.approx(
        [math.hypot(.8, .1) - .2, 1.5, 0, 0])

    resumed = tmp_path / 'resumed'
    # The new split contains a different moment range; the resumed model keeps the saved bounds.
    shifted = TRAIN_XYZ.replace('1.0 0.1 0.0 0.2 -0.1 0.05', '2.0 0.1 0.0 0.2 -0.1 0.05')
    execute(resumed, config, shifted, model, checkpoint)
    restarted = metadata((resumed / 'nep.txt').read_text())
    for key in ('spin_wall_high', 'spin_wall_low'):
        assert restarted[key] == exported[key]
