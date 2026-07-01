"""Golden-file regression tests for a small number of realistic (non-toy) systems.

Uses nep_BaZrO3.txt (no qNEP/TNEP counterpart in this suite, per project decision -- see
conftest.py's _MODEL_FILES comment) against the file-backed bulk_BaZrO3 structure (conftest.py's
make_bulk_bazro3) as the one "realistic non-toy" system. Golden reference arrays (energy, forces)
are frozen in fixtures/golden/ and regenerated only via --update-golden; never overwritten
silently on failure.
"""
import numpy as np
import pytest

from conftest import GOLDEN_DIR, MODELS_DIR, TOLERANCES, make_bulk_bazro3, make_gpunep

pytestmark = pytest.mark.fast


def _golden_path(name):
    return GOLDEN_DIR / f'{name}.npz'


def _compare_or_update(name, energy, forces, update_golden):
    path = _golden_path(name)
    if update_golden:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, energy=energy, forces=forces)
        pytest.skip(f'Updated golden file {path}; rerun without --update-golden to verify it.')

    if not path.exists():
        pytest.fail(f'Golden file {path} does not exist; run with --update-golden to create it.')

    golden = np.load(path)
    energy_tol = TOLERANCES['energy']
    assert energy == pytest.approx(
        float(golden['energy']), rel=energy_tol['rtol'], abs=energy_tol['atol'])
    assert np.allclose(forces, golden['forces'], **TOLERANCES['force'])


def test_bulk_bazro3_regression(update_golden, gpumd_command):
    atoms = make_bulk_bazro3()
    atoms.calc = make_gpunep(MODELS_DIR / 'nep_BaZrO3.txt', gpumd_command, 'nep')
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    _compare_or_update('bulk_bazro3', energy, forces, update_golden)
