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
import re
import subprocess

import numpy as np
import pytest
from calorine.calculators import GPUNEP
from calorine.gpumd import write_xyz

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


def _run_gpumd(directory, model_path, atoms, kspace_before_run, kspace_after_run, gpumd_command):
    write_xyz(str(directory / 'model.xyz'), atoms)
    lines = [f'potential {model_path}']
    lines += [f'kspace {kspace_before_run}'] if kspace_before_run else []
    lines += ['ensemble nve', 'time_step 1', 'run 1']
    lines += [f'kspace {kspace_after_run}'] if kspace_after_run else []
    (directory / 'run.in').write_text('\n'.join(lines) + '\n')
    result = subprocess.run(
        [gpumd_command], cwd=directory, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _pppm_target_spacing(directory, rc_radial, kspace_before_run, kspace_after_run, gpumd_command):
    model_text = (MODELS_DIR / 'qnep_mode1_water.txt').read_text()
    model_path = directory / 'nep.txt'
    model_path.write_text(model_text.replace('cutoff 4 4 ', f'cutoff {rc_radial} 4 ', 1))
    stdout = _run_gpumd(
        directory, model_path, make_bulk_water(), kspace_before_run, kspace_after_run,
        gpumd_command
    )
    return float(re.search(r'PPPM mesh: .*\(target spacing (\S+) A', stdout).group(1))


# The default spacing is min(1, rc_radial / 6) A.
# With alpha = pi / rc_radial, it keeps the product of alpha and the spacing at or below pi / 6.
@pytest.mark.parametrize(
    'rc_radial, kspace_before_run, kspace_after_run, expected_spacing',
    [
        (4, None, None, 4 / 6),
        (4, 'pppm', None, 4 / 6),
        (6, 'pppm', None, 1.0),
        (8, 'pppm', None, 1.0),
        (4, 'pppm 0.9', None, 0.9),
        (4, None, 'pppm 0.9', 0.9),
    ],
    ids=['absent_rc4', 'default_rc4', 'default_rc6', 'default_rc8', 'explicit', 'explicit_late'],
)
def test_pppm_spacing(
    tmp_path, gpumd_command, rc_radial, kspace_before_run, kspace_after_run, expected_spacing
):
    spacing = _pppm_target_spacing(
        tmp_path, rc_radial, kspace_before_run, kspace_after_run, gpumd_command
    )
    assert spacing == pytest.approx(expected_spacing, rel=1e-12)


# The ratio of the estimate to the root-mean-square force difference is 0.97, 1.13 and 1.20
# at spacings of 0.5, 0.7 and 1.0 A for this water box of 1701 atoms.
PPPM_FORCE_ERROR_ESTIMATE_TOLERANCE = 0.25


@pytest.mark.parametrize('spacing', [0.5, 0.7, 1.0])
def test_pppm_force_error_estimate(tmp_path, gpumd_command, spacing):
    atoms = make_bulk_water().repeat(3)
    model_path = MODELS_DIR / 'qnep_mode1_water.txt'
    stdout = _run_gpumd(tmp_path, model_path, atoms, f'pppm {spacing}', None, gpumd_command)
    estimate = float(re.search(r'PPPM force error estimate: (\S+) eV/A', stdout).group(1))

    forces = {}
    for kspace in [f'pppm {spacing}', 'ewald']:
        calc = GPUNEP(str(model_path), command=gpumd_command)
        calc.single_point_parameters = calc.single_point_parameters + [('kspace', kspace)]
        atoms.calc = calc
        forces[kspace] = atoms.get_forces()
    difference = forces[f'pppm {spacing}'] - forces['ewald']
    rms_difference = np.sqrt(np.mean(np.sum(difference**2, axis=1)))
    assert estimate == pytest.approx(rms_difference, rel=PPPM_FORCE_ERROR_ESTIMATE_TOLERANCE)
