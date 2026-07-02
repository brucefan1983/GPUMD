"""Golden-file regression tests for a small number of full-size, realistic systems.

Uses nep_BaZrO3.txt against the file-backed bulk_BaZrO3 structure (conftest.py's
make_bulk_bazro3); a TNEP susceptibility counterpart for BaZrO3 does exist
(tnep-BaZrO3-susceptibility.txt) and is exercised directly in test_io_tnep_commands.py, just not
through this file's golden-comparison path. Golden reference arrays (energy, forces) are frozen
in fixtures/golden/ and regenerated only via --update-golden; never overwritten silently on
failure. Uses conftest.py's compare_or_update_golden, shared with test_io_tnep_commands.py.
"""
import pytest

from conftest import MODELS_DIR, TOLERANCES, compare_or_update_golden, make_bulk_bazro3, \
    make_gpunep

pytestmark = pytest.mark.fast


def test_bulk_bazro3_regression(update_golden, gpumd_command):
    atoms = make_bulk_bazro3()
    atoms.calc = make_gpunep(MODELS_DIR / 'nep_BaZrO3.txt', gpumd_command, 'nep')
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    compare_or_update_golden(
        'bulk_bazro3', {'energy': energy, 'forces': forces},
        {'energy': TOLERANCES['energy'], 'forces': TOLERANCES['force']}, update_golden)
