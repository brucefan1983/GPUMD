"""Command IO smoke tests (Tier 1): basic MD setup parameters.

Confirms each keyword runs without error (exit code 0) and produces thermo.out (proof actual MD
steps executed, not just a no-op parse). Does not validate physical correctness -- see
io_helpers.py's module docstring.

No qNEP dependency: every case here works identically for plain NEP and qNEP models, so this
file runs across the full structure x model_type matrix with no skip logic of its own (beyond
the combinations conftest.py's model_path fixture already skips for missing models).
"""
from dataclasses import replace

import numpy as np
import pytest

from io_helpers import CommandIOCase, run_and_check

pytestmark = pytest.mark.fast

_POTENTIAL_SENTINEL = '__MODEL_PATH__'
_NPT_BER_PARAMS_SENTINEL = '__NPT_BER_AUTO__'


def _npt_ber_params(cell):
    """npt_ber's orthorhombic (Condition 2) and triclinic (Condition 3) parameter forms take a
    different number of values, and GPUMD requires the form to match the box's actual shape --
    see doc/gpumd/input_parameters/ensemble_standard.rst. bulk_C's cell is a triclinic
    representation of the diamond lattice; bulk_perovskite/bulk_water are orthorhombic."""
    off_diagonal = cell - np.diag(np.diag(cell))
    if np.allclose(off_diagonal, 0, atol=1e-6):
        return (300, 300, 100, 0, 0, 0, 100, 100, 100, 1000)
    return (300, 300, 100, 0, 0, 0, 0, 0, 0, 100, 100, 100, 100, 100, 100, 1000)


BASIC_SETUP_CASES = [
    CommandIOCase(name='potential'),
    CommandIOCase(name='time_step'),
    CommandIOCase(name='run'),
    CommandIOCase(name='velocity'),
    CommandIOCase(name='ensemble_nve', ensemble='nve'),
    CommandIOCase(name='ensemble_nvt_ber', ensemble='nvt_ber', ensemble_params=(300, 300, 100)),
    CommandIOCase(
        name='ensemble_npt_ber', ensemble='npt_ber', ensemble_params=_NPT_BER_PARAMS_SENTINEL),
    CommandIOCase(name='correct_velocity', run_in_lines=[('correct_velocity', 10)]),
    CommandIOCase(
        name='replicate',
        prelude_lines=[('replicate', [1, 1, 1]), ('potential', _POTENTIAL_SENTINEL)]),
    CommandIOCase(name='fix', n_groups=1, run_in_lines=[('fix', 0)]),
    CommandIOCase(
        name='add_force', n_groups=1,
        run_in_lines=[('add_force', [0, 0, 0.01, 0, 0])]),
    CommandIOCase(
        name='add_spring', n_groups=1,
        run_in_lines=[('add_spring', ['ghost_com', 0, 0, 0, 0, 0, 'couple', 1.0, 0, 0, 0, 0])],
        expected_output_files=['spring_force_0.out']),
    CommandIOCase(name='change_box', run_in_lines=[('change_box', 0.01)]),
    CommandIOCase(
        name='deform', ensemble='npt_ber', ensemble_params=_NPT_BER_PARAMS_SENTINEL,
        run_in_lines=[('deform', [1e-5, 0, 0, 1, 0, 0])]),
    CommandIOCase(name='dftd3', run_in_lines=[('dftd3', ['pbe', 12, 6])]),
    CommandIOCase(name='kspace', run_in_lines=[('kspace', 'ewald')]),
    CommandIOCase(
        name='move', ensemble='nvt_ber', ensemble_params=(300, 300, 100), n_groups=2,
        # GPUMD refuses a moving group with no fixed group ("It is not allowed to have moving
        # group but no fixed group"), so this pairs `move` on group 1 with `fix` on group 0.
        run_in_lines=[('fix', 0), ('move', [0, 1, 0.0001, 0, 0])]),
    CommandIOCase(
        name='minimize', skip_base_block=True, expect_thermo=True,
        run_in_lines=[
            ('minimize', ['sd', -1, 10]),
            ('ensemble', 'nve'),
            ('dump_thermo', 1),
            ('run', 5),
        ]),
]


def _resolve_case(case, structure, model_path):
    """Resolves per-invocation sentinels: `replicate` needs `potential` to appear right after it
    rather than being auto-prepended by GPUNEP (see CommandIOCase.prelude_lines' docstring in
    io_helpers.py), and npt_ber's parameter form depends on the current structure's box shape."""
    def resolve(value):
        return str(model_path) if value == _POTENTIAL_SENTINEL else value

    ensemble_params = case.ensemble_params
    if ensemble_params == _NPT_BER_PARAMS_SENTINEL:
        ensemble_params = _npt_ber_params(np.array(structure.cell))

    return replace(
        case,
        prelude_lines=[(k, resolve(v)) for k, v in case.prelude_lines],
        run_in_lines=[(k, resolve(v)) for k, v in case.run_in_lines],
        ensemble_params=ensemble_params,
    )


@pytest.mark.parametrize('case', BASIC_SETUP_CASES, ids=lambda c: c.name)
def test_command_io(tmp_path, structure, structure_name, model_path, model_type, gpumd_command,
                     case):
    if case.name == 'change_box' and structure_name == 'bulk_C':
        pytest.skip(
            "confirmed GPUMD hang (not a crash/error, genuinely never returns): "
            "'change_box <delta>' (isotropic form) hangs indefinitely on bulk_C's box, which "
            'has zero on-diagonal components (the primitive rhombohedral representation of '
            'diamond, with lattice vectors along face-diagonal directions). A generic sheared '
            'triclinic box with nonzero diagonal components works fine, so this is specific to '
            'the zero-diagonal case, not triclinic boxes in general -- likely a genuine GPUMD '
            'bug worth reporting/fixing upstream, not a test setup issue.')
    case = _resolve_case(case, structure, model_path)
    run_and_check(tmp_path, structure, model_path, model_type, gpumd_command, case)
