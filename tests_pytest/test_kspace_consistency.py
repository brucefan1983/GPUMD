"""GPUMD's own PPPM-vs-Ewald consistency for qNEP models.

test_invariances.py explicitly requests Ewald for qNEP models (see conftest.py's make_gpunep) so
that the reciprocal-space method itself isn't a variable in those comparisons. This file instead
evaluates the same structure with both methods directly and checks they agree within a bound wide
enough to reflect PPPM being a genuine, expected approximation to Ewald (not floating-point
noise, so this tolerance is not meant to be tightened toward TOLERANCES the way other comparisons
in this suite are) -- a *relative* approximation-quality check, complementary to
test_regression.py's golden-file regression for each method's own output against frozen
reference data (which catches a regression in either implementation individually, not just their
mutual disagreement).

Spans multiple system sizes since PPPM's mesh/k-point resolution scales with the box: the
existing bulk_water and bulk_perovskite fixtures, plus a larger supercell built by tiling
bulk_perovskite further (no new fixture file needed).
"""
import numpy as np
import pytest
from calorine.calculators import GPUNEP

from conftest import MODELS_DIR, approx_tol, make_bulk_perovskite, make_bulk_water

pytestmark = pytest.mark.fast

# Energy is extensive (grows with system size), so its PPPM-vs-Ewald gap grows too across the
# sizes covered here -- rtol alone tracks that scaling, with atol only as a floor for the
# smallest system. Force is intensive (per atom, not per system), so it does not grow the same
# way with size; atol carries most of the weight there instead.
PPPM_VS_EWALD_ENERGY_TOLERANCE = dict(rtol=1e-2, atol=1e-3)
PPPM_VS_EWALD_FORCE_TOLERANCE = dict(rtol=1e-2, atol=5e-3)


def make_bulk_perovskite_large():
    """A larger supercell than the standard bulk_perovskite fixture, built by tiling it further
    rather than adding a new fixture file, purely to vary system/box size for this file's
    PPPM-vs-Ewald comparison."""
    return make_bulk_perovskite().repeat((2, 1, 1))


_STRUCTURE_BUILDERS = {
    'bulk_water': make_bulk_water,
    'bulk_perovskite': make_bulk_perovskite,
    'bulk_perovskite_large': make_bulk_perovskite_large,
}

# No qNEP model exists for carbon (bulk_C), same gap as elsewhere in this suite, so bulk_C is
# simply not one of the structures covered here.
_MODEL_FILES = {
    ('bulk_water', 'qnep_mode1'): 'qnep_mode1_water.txt',
    ('bulk_water', 'qnep_mode2'): 'qnep_mode2_water.txt',
    ('bulk_perovskite', 'qnep_mode1'): 'qnep_mode1_BaTiO3.txt',
    ('bulk_perovskite', 'qnep_mode2'): 'qnep_mode2_BaTiO3.txt',
    ('bulk_perovskite_large', 'qnep_mode1'): 'qnep_mode1_BaTiO3.txt',
    ('bulk_perovskite_large', 'qnep_mode2'): 'qnep_mode2_BaTiO3.txt',
}


def _evaluate(structure_name, model_type, gpumd_command, kspace):
    atoms = _STRUCTURE_BUILDERS[structure_name]()
    model_path = MODELS_DIR / _MODEL_FILES[(structure_name, model_type)]
    calc = GPUNEP(str(model_path), command=gpumd_command)
    calc.single_point_parameters = calc.single_point_parameters + [('kspace', kspace)]
    atoms.calc = calc
    return atoms.get_potential_energy(), atoms.get_forces()


@pytest.mark.parametrize('structure_name', list(_STRUCTURE_BUILDERS))
@pytest.mark.parametrize('model_type', ['qnep_mode1', 'qnep_mode2'])
def test_pppm_agrees_with_ewald(structure_name, model_type, gpumd_command):
    energy_pppm, forces_pppm = _evaluate(structure_name, model_type, gpumd_command, 'pppm')
    energy_ewald, forces_ewald = _evaluate(structure_name, model_type, gpumd_command, 'ewald')

    assert energy_pppm == approx_tol(energy_ewald, PPPM_VS_EWALD_ENERGY_TOLERANCE)
    assert np.allclose(forces_pppm, forces_ewald, **PPPM_VS_EWALD_FORCE_TOLERANCE)
