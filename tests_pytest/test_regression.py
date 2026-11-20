"""Golden-file regression tests for a small number of full-size, realistic systems.

Uses nep_BaZrO3.txt against the file-backed bulk_BaZrO3 structure (conftest.py's
make_bulk_bazro3); a TNEP susceptibility counterpart for BaZrO3 does exist
(tnep-BaZrO3-susceptibility.txt) and is exercised directly in test_io_tnep_commands.py, just not
through this file's golden-comparison path. Golden reference arrays (energy, forces, stress) are
frozen in fixtures/golden/ and regenerated only via --update-golden; never overwritten silently
on failure. Uses conftest.py's compare_or_update_golden, shared with test_io_tnep_commands.py.

Stress is included alongside energy/forces since GPUNEP already computes it as part of the same
calculation (calorine's GPUNEP reads energy/forces/stress together from thermo.out in one pass,
see calorine/calculators/gpunep.py) -- this is the only place in the suite TOLERANCES['virial']
is exercised.

test_bec_regression compares Born effective charges with separate frozen references
for Ewald and PPPM. It selects each kspace method explicitly, bypassing
make_gpunep's Ewald default. test_kspace_consistency.py separately compares the
two methods with each other.
"""
import pytest
from calorine.calculators import GPUNEP

from conftest import MODELS_DIR, TOLERANCES, compare_or_update_golden, make_bulk_bazro3, \
    make_bulk_perovskite, make_gpunep

pytestmark = pytest.mark.fast


def test_bulk_bazro3_regression(update_golden, gpumd_command):
    atoms = make_bulk_bazro3()
    atoms.calc = make_gpunep(MODELS_DIR / 'nep_BaZrO3.txt', gpumd_command, 'nep')
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    compare_or_update_golden(
        'bulk_bazro3', {'energy': energy, 'forces': forces, 'stress': stress},
        {'energy': TOLERANCES['energy'], 'forces': TOLERANCES['force'],
         'stress': TOLERANCES['virial']}, update_golden)


@pytest.mark.parametrize('kspace', ['ewald', 'pppm'])
def test_bec_regression(update_golden, gpumd_command, kspace):
    atoms = make_bulk_perovskite()
    calc = GPUNEP(str(MODELS_DIR / 'qnep_mode1_BaTiO3.txt'), command=gpumd_command)
    calc.single_point_parameters = calc.single_point_parameters + [('kspace', kspace)]
    atoms.calc = calc
    bec = calc.get_born_effective_charges(atoms)
    compare_or_update_golden(
        f'bec_bulk_perovskite_qnep_mode1_{kspace}', {'bec': bec}, TOLERANCES['bec'], update_golden)
