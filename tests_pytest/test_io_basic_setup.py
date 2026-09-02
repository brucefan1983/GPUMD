"""Command IO smoke tests (Tier 1): basic MD setup parameters.

Confirms each keyword runs without error (exit code 0) and produces thermo.out (proof actual MD
steps executed, not just a no-op parse). Does not validate physical correctness -- see
io_helpers.py's module docstring.

No qNEP dependency: every case here works identically for plain NEP and qNEP models, so this
file runs across the full structure x model_type matrix, with skips only for the combinations
conftest.py's model_path fixture already skips for missing models, plus one deliberate,
evidenced skip below (change_box/bulk_C, a confirmed GPUMD hang) -- not a blanket "no skip
logic" file.
"""
from dataclasses import replace

import numpy as np
import pytest

from io_helpers import BASE_N_STEPS, CommandIOCase, run_and_check

pytestmark = pytest.mark.fast

_POTENTIAL_SENTINEL = '__MODEL_PATH__'
_NPT_BER_PARAMS_SENTINEL = '__NPT_BER_AUTO__'

# add_spring writes one row every `output_stride_` steps (src/measure/add_spring.cuh), counting
# from step 0, so a BASE_N_STEPS run writes the multiples of the stride below BASE_N_STEPS.
_SPRING_OUTPUT_STRIDE = 100
_SPRING_COLUMNS = ('step', 'mode', 'Fx', 'Fy', 'Fz', 'Ftotal', 'energy')


def _check_spring_output(path):
    """Checks the spring force file: a comment header naming the seven columns, then one row of
    seven fields per output stride. The companion `.restart` file, which ghost_com saves at the
    end of a run, is covered by the existence check in run_and_check and not parsed here."""
    if path.suffix == '.restart':
        return

    lines = [line for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert lines, f'{path.name} is empty'

    header = lines[0]
    assert header.startswith('#'), f'{path.name} lacks a comment header: {header!r}'
    for column in _SPRING_COLUMNS:
        assert column in header, f'{path.name} header lacks the {column!r} column: {header!r}'

    rows = [line.split() for line in lines[1:]]
    assert [int(row[0]) for row in rows] == list(range(0, BASE_N_STEPS, _SPRING_OUTPUT_STRIDE)), \
        f'{path.name} does not hold one row per output stride: {rows}'
    for row in rows:
        assert len(row) == len(_SPRING_COLUMNS), f'{path.name} row has {len(row)} fields: {row}'
        for field in row:
            float(field)  # raises ValueError if the run wrote a field that is not a number


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
        # The file names carry the group method, the group ID, and the spring ID, all 0 here:
        # method 0 group 0 is the single group n_groups=1 writes, and spring ID 0 is the lowest
        # unused ID in a fresh run directory. See doc/gpumd/input_parameters/add_spring.rst.
        expected_output_files=['spring_gm0_g0_s0.out', 'spring_gm0_g0_s0.restart'],
        parse_check=_check_spring_output),
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
        pytest.skip("confirmed GPUMD hang on bulk_C's zero-diagonal box (change_box <delta> "
                    'never returns); other triclinic boxes work fine -- likely a genuine '
                    'upstream bug.')
    case = _resolve_case(case, structure, model_path)
    run_and_check(tmp_path, structure, model_path, model_type, gpumd_command, case)
