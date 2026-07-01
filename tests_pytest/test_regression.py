"""Golden-file regression tests for a small number of realistic (non-toy) systems.

Uses nep_BaZrO3.txt (no qNEP/TNEP counterpart in this suite, per project decision -- see
conftest.py's _MODEL_FILES comment) as the one "realistic non-toy" system. Golden reference
arrays (energy, forces) are frozen in fixtures/golden/ and regenerated only via --update-golden;
never overwritten silently on failure.
"""
import numpy as np
import pytest
from ase import Atoms
from calorine.calculators import GPUNEP

from conftest import GOLDEN_DIR, MODELS_DIR, TOLERANCES

pytestmark = pytest.mark.fast

# Cubic perovskite BaZrO3, representative literature lattice constant (not user-specified, since
# this fixture only serves as a "realistic non-toy" regression target rather than a physically
# validated structure like bulk_perovskite/bulk_C elsewhere in this suite).
BAZRO3_LATTICE_CONSTANT = 4.19  # Angstrom


def _make_bulk_bazro3():
    """2x2x2 cubic BaZrO3 supercell (40 atoms), rattled with a fixed seed for reproducibility so
    the golden file stays valid across reruns."""
    a = BAZRO3_LATTICE_CONSTANT
    cell = Atoms(
        symbols=['Ba', 'Zr', 'O', 'O', 'O'],
        scaled_positions=[
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ],
        cell=[a, a, a],
        pbc=True,
    )
    atoms = cell.repeat((2, 2, 2))
    atoms.rattle(stdev=0.01, seed=42)
    return atoms


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
    atoms = _make_bulk_bazro3()
    atoms.calc = GPUNEP(str(MODELS_DIR / 'nep_BaZrO3.txt'), command=gpumd_command)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    _compare_or_update('bulk_bazro3', energy, forces, update_golden)
