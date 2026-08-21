"""Command IO smoke tests (Tier 1): dump_* commands.

Confirms each keyword runs without error (exit code 0) and produces a parseable output file of
the expected shape -- not physical correctness. See io_helpers.py's module docstring.

dump_dipole (and its rank-2 counterpart dump_polarizability) are covered in
test_io_tnep_commands.py, not here -- both strictly require a genuine TNEP model (not a qNEP
charge model, contrary to this suite's original Tier 1 scope) as the second `potential`, which
doesn't fit this file's calculator/model_type fixture pattern the way the rest of these dump_*
commands do.
"""
import re

import numpy as np
import pytest
from ase.build import bulk
from ase.cell import Cell
from ase.io import read
from calorine.gpumd import read_xyz

from conftest import MODELS_DIR
from io_helpers import BASE_N_STEPS, CommandIOCase, run_and_check, run_command_io_case
from test_parsing import read_thermo_out

pytestmark = pytest.mark.fast

# The AMBER convention specifies forces in kcal/mol/A while GPUMD works in eV/A, so the file
# carries them converted. Repeated here rather than imported, so that a change to the factor in
# dump_netcdf.cu has to be made deliberately in both places.
EV_TO_KCAL_PER_MOL = 23.06054783061903


def _check_thermo_format(path):
    data = read_thermo_out(path)
    assert data.shape[1] == 18


def _check_xyz_multi_frame(path, natoms):
    frames = read(path, index=':')
    assert len(frames) >= 1
    assert len(frames[0]) == natoms


def _check_natoms_single_frame(path, natoms):
    atoms = read_xyz(str(path))
    assert len(atoms) == natoms


def _check_columnar(path, ncols, natoms):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.shape[1] == ncols
    assert data.shape[0] % natoms == 0
    assert data.shape[0] > 0


def _significant_digits(token):
    """Counts the significant digits of a number as gpumd formatted it, ignoring the sign, the
    decimal point, any exponent, and leading zeros."""
    mantissa = token.split('e')[0].split('E')[0]
    return len(mantissa.replace('-', '').replace('.', '').lstrip('0'))


def _max_atom_row_digits(path, natoms):
    """Largest number of significant digits over every per-atom column of the first frame. The
    species column is skipped, and so is the comment line."""
    lines = path.read_text().splitlines()
    return max(
        _significant_digits(token)
        for line in lines[2:2 + natoms] for token in line.split()[1:])


def _comment_line(path):
    return path.read_text().splitlines()[1]


def _comment_fields(path):
    """Parses the extended XYZ comment line into {key: [value, ...]}, accepting both the bare
    (energy=-1.5) and the quoted (virial="1 2 3 ...") spellings."""
    fields = {}
    for match in re.finditer(r'(\w+)=(?:"([^"]*)"|(\S+))', _comment_line(path)):
        raw = match.group(2) if match.group(2) is not None else match.group(3)
        fields[match.group(1)] = raw.split()
    return fields


def _quoted_field_text(path, key):
    """The raw text inside the quotes of a `key="..."` comment-line field, needed where the exact
    spacing matters and splitting on whitespace would hide it."""
    match = re.search(rf'{key}="([^"]*)"', _comment_line(path))
    assert match is not None, f'{key} is missing from the comment line'
    return match.group(1)


def _check_double_precision(path, natoms):
    """`precision double` must write enough significant digits to recover a double exactly, which
    the default `single` setting (nine digits) does not. Counting digits in the raw text is the
    only way to see this, since reading through ase would hide it."""
    _check_xyz_multi_frame(path, natoms)
    digits = _max_atom_row_digits(path, natoms)
    assert digits > 9, f'expected more than 9 significant digits, found {digits}'


def _build_case(name, natoms):
    """Builds the CommandIOCase for `name` freshly per test invocation, since several
    parse_check callbacks need natoms, which is only known once the structure fixture runs."""
    cases = {
        'dump_thermo': CommandIOCase(
            name='dump_thermo', run_in_lines=[('dump_thermo', 1)],
            expected_output_files=['thermo.out'], parse_check=_check_thermo_format),
        'dump_restart': CommandIOCase(
            name='dump_restart', run_in_lines=[('dump_restart', 1)],
            expected_output_files=['restart.xyz'],
            parse_check=lambda p: _check_natoms_single_frame(p, natoms)),
        'dump_xyz': CommandIOCase(
            name='dump_xyz', run_in_lines=[('dump_xyz', [1, 'dump_xyz_test.xyz'])],
            expected_output_files=['dump_xyz_test.xyz'],
            parse_check=lambda p: _check_xyz_multi_frame(p, natoms)),
        'dump_xyz_group': CommandIOCase(
            name='dump_xyz_group',
            run_in_lines=[('dump_xyz', [1, 'group.xyz', 'group', 0, 1])],
            expected_output_files=['group.xyz'], n_groups=2,
            parse_check=lambda p: _check_xyz_multi_frame(p, natoms - natoms // 2)),
        'dump_xyz_precision': CommandIOCase(
            name='dump_xyz_precision',
            run_in_lines=[('dump_xyz', [1, 'precise.xyz', 'precision', 'double', 'force'])],
            expected_output_files=['precise.xyz'],
            parse_check=lambda p: _check_double_precision(p, natoms)),
    }
    return cases[name]


DUMP_COMMAND_CASE_NAMES = [
    'dump_thermo', 'dump_restart', 'dump_xyz', 'dump_xyz_group', 'dump_xyz_precision',
]


@pytest.mark.parametrize('case_name', DUMP_COMMAND_CASE_NAMES)
def test_command_io(tmp_path, structure, model_path, model_type, gpumd_command, case_name):
    case = _build_case(case_name, len(structure))
    run_and_check(tmp_path, structure, model_path, model_type, gpumd_command, case)


def test_dump_xyz_comment_line_is_well_formed(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """Lattice, virial, and stress go through Dump_XYZ::print_tensor, which loops over the nine
    values reusing the precision format string and skips that format's leading space for the first
    value only. Nothing else pins that down, so a later edit to the format could corrupt the cell
    or the stress while every other test still passed."""
    case = CommandIOCase(
        name='dump_xyz_comment_line',
        run_in_lines=[('dump_xyz', [1, 'comment.xyz', 'virial'])],
        expected_output_files=['comment.xyz'])
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    assert result.returncode == 0, result.stdout
    path = tmp_path / 'comment.xyz'

    fields = _comment_fields(path)
    for key in ('Lattice', 'virial', 'stress'):
        assert len(fields[key]) == 9, f'{key} should carry nine values, got {fields[key]}'
        # no leading space directly after the opening quote, which is what skipping the first
        # value's leading space is for
        assert not _quoted_field_text(path, key).startswith(' '), f'{key} has a leading space'

    frame = read(path, index=0)
    np.testing.assert_allclose(frame.cell.array, structure.cell.array, atol=1e-6)

    # energy on the comment line is the potential energy, matching thermo.out's U column
    potential_energy = read_thermo_out(tmp_path / 'thermo.out')[0, 2]
    assert np.isfinite(float(fields['energy'][0]))
    np.testing.assert_allclose(float(fields['energy'][0]), potential_energy, rtol=1e-6)

    for key in ('virial', 'stress'):
        tensor = np.array(fields[key], dtype=float).reshape(3, 3)
        np.testing.assert_allclose(tensor, tensor.T, atol=0, rtol=0,
                                   err_msg=f'{key} should be written symmetrically')
    assert frame.get_stress().shape == (6,)


def test_dump_xyz_precision_reaches_the_comment_line(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """`precision` must apply to the comment-line tensors too, not only the per-atom columns.
    Both files come from one run, so they describe the same trajectory and are directly
    comparable."""
    case = CommandIOCase(
        name='dump_xyz_comment_precision',
        run_in_lines=[
            ('dump_xyz', [1, 'single.xyz', 'virial']),
            ('dump_xyz', [1, 'double.xyz', 'virial', 'precision', 'double']),
        ],
        expected_output_files=['single.xyz', 'double.xyz'])
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    assert result.returncode == 0, result.stdout

    single = _comment_fields(tmp_path / 'single.xyz')
    double = _comment_fields(tmp_path / 'double.xyz')
    for key in ('energy', 'virial', 'stress'):
        single_digits = max(_significant_digits(token) for token in single[key])
        double_digits = max(_significant_digits(token) for token in double[key])
        assert single_digits <= 9, f'{key} under the default should hold at most 9 digits'
        assert double_digits > single_digits, (
            f'precision double did not widen {key}: {single_digits} vs {double_digits} digits')


def test_dump_xyz_default_precision_is_single(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """The default is single precision, and asking for it explicitly changes nothing. Without this
    the default could be flipped in dump_xyz.cuh and the suite would stay green."""
    case = CommandIOCase(
        name='dump_xyz_default_precision',
        run_in_lines=[
            ('dump_xyz', [1, 'default.xyz', 'force']),
            ('dump_xyz', [1, 'explicit.xyz', 'force', 'precision', 'single']),
        ],
        expected_output_files=['default.xyz', 'explicit.xyz'])
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    assert result.returncode == 0, result.stdout

    default_text = (tmp_path / 'default.xyz').read_text()
    assert default_text == (tmp_path / 'explicit.xyz').read_text(), (
        'an explicit `precision single` should be identical to the default')
    digits = _max_atom_row_digits(tmp_path / 'default.xyz', len(structure))
    assert digits <= 9, f'the default should hold at most 9 significant digits, found {digits}'


def test_dump_xyz_precision_is_independent_of_magnitude(tmp_path, gpumd_command):
    """The reason the precision option exists. Output used to be a fixed %.8f, which spends its
    digits on absolute scale, so a force of order 1e-5 kept only about four significant digits and
    anything below 1e-8 was written as zero. The %g formats now used keep the same relative
    precision at any magnitude.

    An unrattled diamond cell is used because its forces are small (order 1e-5), which is exactly
    where a fixed-decimal format loses out. The double-precision file is the reference: under the
    old format the single-precision relative error here would have been about 1e-4."""
    atoms = bulk('C', 'diamond', 3.57, cubic=True)
    case = CommandIOCase(
        name='dump_xyz_small_forces',
        run_in_lines=[
            ('time_step', 0),
            ('ensemble', 'nve'),
            ('dump_xyz', [1, 'single.xyz', 'force']),
            ('dump_xyz', [1, 'double.xyz', 'force', 'precision', 'double']),
            ('run', 1),
        ],
        expected_output_files=['single.xyz', 'double.xyz'],
        skip_base_block=True)
    result = run_command_io_case(
        tmp_path, atoms, MODELS_DIR / 'nep_C.txt', 'nep', gpumd_command, case)
    assert result.returncode == 0, result.stdout

    single = read(tmp_path / 'single.xyz', index=0).get_forces()
    double = read(tmp_path / 'double.xyz', index=0).get_forces()
    assert np.max(np.abs(double)) < 1e-3, (
        'this check is only meaningful on a near-force-free configuration, but the forces are '
        f'{np.max(np.abs(double)):.3e}')
    assert np.all(single != 0.0), 'small forces must not be flattened to zero'
    relative_error = np.max(np.abs(single - double) / np.abs(double))
    assert relative_error < 1e-8, (
        f'relative error {relative_error:.3e} is too large for nine significant digits')


def test_dump_xyz_per_atom_virial_is_consistent_with_the_total(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """The per-atom virial columns must add up to the total virial on the comment line.

    Note what this can and cannot catch. The comment-line total is produced by summing the very
    same per-atom array the columns come from, so a contribution missing from that array is
    missing from both sides and cancels. What this does pin down is the nine-component column
    ordering, the row-major reshape, and the fact that the comment line keeps only the six
    independent components. The reciprocal-space guard is the PPPM-versus-Ewald test below."""
    case = CommandIOCase(
        name='dump_xyz_virial_total',
        run_in_lines=[
            ('dump_xyz', [1, 'virial.xyz', 'virial', 'precision', 'double', 'group', 0, 0]),
        ],
        expected_output_files=['virial.xyz'], n_groups=1)
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    assert result.returncode == 0, result.stdout
    path = tmp_path / 'virial.xyz'

    per_atom = read(path, index=0).arrays['virial'].reshape(-1, 3, 3).sum(axis=0)
    total = np.array(_comment_fields(path)['virial'], dtype=float).reshape(3, 3)
    # the comment line symmetrizes, keeping only six independent components, so compare those
    for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
        np.testing.assert_allclose(
            per_atom[i, j], total[i, j], rtol=1e-6, atol=1e-8,
            err_msg=f'per-atom virial component ({i}, {j}) does not sum to the total')


def _dumped_per_atom_virial(directory, structure, model_path, model_type, gpumd_command, kspace):
    """Runs one qNEP single point with the given reciprocal-space method and returns the per-atom
    virial as an (natoms, 9) array."""
    directory.mkdir()
    case = CommandIOCase(
        name=f'dump_xyz_virial_{kspace}',
        prelude_lines=[('kspace', kspace)],
        run_in_lines=[('dump_xyz', [1, 'virial.xyz', 'virial', 'precision', 'double'])],
        expected_output_files=['virial.xyz'])
    result = run_command_io_case(
        directory, structure, model_path, model_type, gpumd_command, case)
    assert result.returncode == 0, f'kspace {kspace} failed:\n{result.stdout}'
    return read(directory / 'virial.xyz', index=0).arrays['virial']


def test_dump_xyz_per_atom_virial_agrees_between_pppm_and_ewald(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """Regression guard for check_need_peratom_virial in utilities/read_file.cu, which decides
    whether per-atom virials are accumulated at all by scanning the tokens of every dump_xyz line
    for `virial`. The reworked syntax moved every token on that line and put `group` and
    `precision` tokens beside them.

    That flag gates only the PPPM reciprocal-space contribution (force/pppm.cu), so the two
    methods have to be compared against each other: with the scan broken, the PPPM virial silently
    loses its k-space part while Ewald keeps it. Tolerances match test_kspace_consistency.py, where
    the same two methods are compared on energies and forces."""
    if model_type == 'nep':
        pytest.skip('a charge model is needed for there to be a reciprocal-space contribution')

    pppm = _dumped_per_atom_virial(
        tmp_path / 'pppm', structure, model_path, model_type, gpumd_command, 'pppm')
    ewald = _dumped_per_atom_virial(
        tmp_path / 'ewald', structure, model_path, model_type, gpumd_command, 'ewald')

    assert pppm.shape == ewald.shape == (len(structure), 9)
    assert np.max(np.abs(ewald)) > 0, 'the reference virial is identically zero, nothing to compare'
    np.testing.assert_allclose(pppm, ewald, rtol=1e-2, atol=5e-3)


def test_dump_xyz_column_order_does_not_follow_argument_order(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """The per-atom columns are emitted in an order fixed by the code, not in the order the
    quantities were requested, so reversing the keywords must produce an identical file."""
    quantities = ['mass', 'velocity', 'force', 'potential', 'virial']
    case = CommandIOCase(
        name='dump_xyz_column_order',
        run_in_lines=[
            ('dump_xyz', [1, 'forward.xyz'] + quantities),
            ('dump_xyz', [1, 'reversed.xyz'] + quantities[::-1]),
        ],
        expected_output_files=['forward.xyz', 'reversed.xyz'])
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    assert result.returncode == 0, result.stdout

    forward = tmp_path / 'forward.xyz'
    assert _comment_fields(forward)['Properties'] == \
        _comment_fields(tmp_path / 'reversed.xyz')['Properties']
    assert forward.read_text() == (tmp_path / 'reversed.xyz').read_text()
    assert _comment_fields(forward)['Properties'][0] == (
        'species:S:1:pos:R:3:mass:R:1:vel:R:3:forces:R:3:energy_atom:R:1:virial:R:9')


def test_dump_xyz_group_labels_match_the_groupings(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """`group_labels` writes one integer column per grouping method."""
    case = CommandIOCase(
        name='dump_xyz_group_labels',
        run_in_lines=[('dump_xyz', [1, 'labels.xyz', 'group_labels'])],
        expected_output_files=['labels.xyz'], n_groups=2)
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    assert result.returncode == 0, result.stdout
    path = tmp_path / 'labels.xyz'

    assert _comment_fields(path)['Properties'][0].endswith(':group:I:1')
    labels = read(path, index=0).arrays['group']
    half = len(structure) // 2
    expected = np.array([0] * half + [1] * (len(structure) - half))
    np.testing.assert_array_equal(labels.reshape(-1), expected)


def test_dump_xyz_writes_one_file_per_frame_for_a_starred_name(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """A file name ending in a star writes one file per frame, with the star replaced by the step
    number. This replaced the `separated` flag of the removed dump_exyz."""
    case = CommandIOCase(
        name='dump_xyz_separated',
        run_in_lines=[('dump_xyz', [1, 'frame.xyz*'])],
        expected_output_files=[])
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    assert result.returncode == 0, result.stdout

    produced = sorted(tmp_path.glob('frame.xyz*'))
    assert len(produced) == BASE_N_STEPS, (
        f'expected {BASE_N_STEPS} per-frame files, got {[p.name for p in produced]}')
    assert {p.name for p in produced} == {f'frame.xyz{step}' for step in range(1, BASE_N_STEPS + 1)}
    for path in produced:
        # the format has to be named, since ase cannot infer it from the step-numbered suffix
        frames = read(path, index=':', format='extxyz')
        assert len(frames) == 1
        assert len(frames[0]) == len(structure)


def test_deposition_generates_a_usable_dump_xyz_line(tmp_path, gpumd_command):
    """main_gpumd/deposition.cu builds a dump_xyz line as a runtime string, one per subrun, and
    appends `group_labels` when the model carries groupings. Nothing else covers it, so a stale
    argument order or quantity name there would break every deposition run while the rest of the
    suite stayed green.

    A slab with vacuum along z is built here rather than reusing the shared structure fixtures,
    since deposition needs somewhere to deposit into."""
    slab = bulk('C', 'diamond', 3.57, cubic=True).repeat((2, 2, 3))
    slab.center(vacuum=6, axis=2)
    case = CommandIOCase(
        name='deposition',
        run_in_lines=[('deposit', [5, 2, 20, 'atom', 0, 2, -0.05])],
        expected_output_files=[], n_groups=2)
    result = run_command_io_case(
        tmp_path, slab, MODELS_DIR / 'nep_C.txt', 'nep', gpumd_command, case)
    assert result.returncode == 0, result.stdout

    # the generated line, recoverable because deposition rewrites run.in in place for each subrun
    dump_lines = [line for line in (tmp_path / 'run.in').read_text().splitlines()
                  if line.startswith('dump_xyz')]
    assert dump_lines, 'deposition did not generate a dump_xyz line'
    for line in dump_lines:
        tokens = line.split()
        assert tokens[1].isdigit(), f'the interval should come first now, got {line!r}'
        assert tokens[2].startswith('deposited_')
        assert 'group_labels' in tokens, f'expected the renamed quantity in {line!r}'
        assert 'group' not in tokens, f'{line!r} still uses the old quantity name'

    produced = sorted(tmp_path.glob('deposited_*.xyz'))
    assert produced, 'deposition produced no per-subrun dump files'
    atom_counts = []
    for path in produced:
        frame = read(path, index=0)
        assert 'group' in frame.arrays, f'{path.name} is missing the group-label column'
        assert 'vel' in frame.arrays, f'{path.name} is missing the velocity columns'
        atom_counts.append(len(frame))
    assert atom_counts == sorted(atom_counts) and atom_counts[-1] > atom_counts[0], (
        f'atom count should grow as atoms are deposited, got {atom_counts}')


INVALID_DUMP_XYZ_ARGUMENTS = [
    ([1, 'f.xyz', 'forcee'], 'Unrecognized argument'),
    ([1, 'f.xyz', 'group', 0, 0, 'group', 0, 0], 'more than once'),
    ([1, 'f.xyz', 'precision', 'double', 'precision', 'single'], 'more than once'),
    ([1, 'f.xyz', 'force', 'force'], 'more than once'),
    ([1, 'f.xyz', 'precision', 'triple'], 'Invalid precision'),
    # a bare `group` is how the quantity now called `group_labels` used to be spelled
    ([1, 'f.xyz', 'velocity', 'group'], 'group_labels'),
    ([1, 'f.xyz', 'group', 0], 'group_labels'),
    # the old syntax used -1 to mean the whole system, which is now spelled by omitting the option
    ([1, 'f.xyz', 'group', -1, 0], 'Grouping method'),
    ([1], 'at least 2 parameters'),
    ([0, 'f.xyz'], 'dump interval'),
]


@pytest.mark.parametrize('args, expected_message', INVALID_DUMP_XYZ_ARGUMENTS)
def test_invalid_dump_xyz_arguments_are_rejected(
        tmp_path, structure, model_path, model_type, gpumd_command, args, expected_message):
    """Every one of these used to be accepted, silently ignored, or reported with a message that
    did not say what was wrong. The asserted substring matters as much as the exit code, since a
    test that passes because gpumd died for an unrelated reason is worse than no test."""
    case = CommandIOCase(
        name='dump_xyz_invalid', run_in_lines=[('dump_xyz', args)],
        expected_output_files=[], n_groups=1)
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    assert result.returncode != 0, f'dump_xyz {args} unexpectedly succeeded'
    assert expected_message in result.stdout + result.stderr, (
        f'dump_xyz {args} did not report {expected_message!r}\nstdout:\n{result.stdout}')


def test_dump_xyz_group_labels_without_a_grouping_method_is_rejected(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """With no grouping method the Properties field would read group:I:0, which ase cannot parse
    ("need at least one array to concatenate"), so gpumd must refuse rather than write a file that
    looks fine and is unreadable."""
    case = CommandIOCase(
        name='dump_xyz_group_labels_no_groups',
        run_in_lines=[('dump_xyz', [1, 'labels.xyz', 'group_labels'])],
        expected_output_files=[], n_groups=0)
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    assert result.returncode != 0, 'group_labels without a grouping method should be refused'
    assert 'grouping method' in result.stdout + result.stderr, result.stdout


REMOVED_COMMANDS = [
    ('dump_position', 1),
    ('dump_velocity', 1),
    ('dump_force', 1),
    ('dump_exyz', [1, 1, 1, 1]),
    # the positional form, leading with <grouping_method> <group_id>
    ('dump_xyz', [-1, 0, 1, 'dump.xyz']),
]


@pytest.mark.parametrize('keyword, args', REMOVED_COMMANDS)
def test_removed_commands_are_rejected(
        tmp_path, structure, model_path, model_type, gpumd_command, keyword, args):
    """The removed keywords, and the old dump_xyz argument order, must fail with a message naming
    dump_xyz rather than being silently ignored or hitting a generic parse error."""
    case = CommandIOCase(
        name=keyword, run_in_lines=[(keyword, args)], expected_output_files=[])
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    assert result.returncode != 0, f'{keyword}: gpumd unexpectedly succeeded'
    assert 'dump_xyz' in result.stdout + result.stderr, (
        f'{keyword}: error message does not point at dump_xyz\nstdout:\n{result.stdout}')


def _skip_without_netcdf(output):
    """Every dump_netcdf case has to tolerate a gpumd built without -DUSE_NETCDF, where the
    keyword is refused in run.cu before DUMP_NETCDF::parse is ever reached. The rejection tests
    need this as much as the passing ones, since without it they would pass for the wrong
    reason -- a non-zero exit that says nothing about the arguments under test."""
    if 'dump_netcdf is available only when USE_NETCDF flag is set' in output:
        pytest.skip('gpumd binary was built without USE_NETCDF')


def _check_netcdf_result(result):
    output = result.stdout + result.stderr
    _skip_without_netcdf(output)
    assert result.returncode == 0, (
        f'dump_netcdf: gpumd exited {result.returncode}\nstdout:\n{result.stdout}\n'
        f'stderr:\n{result.stderr}')


def test_dump_netcdf_appends_across_run_commands(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """Two dump_netcdf commands writing the same file in one execution extend it, since the
    keyword does not propagate and each run needs its own command."""
    netcdf4 = pytest.importorskip('netCDF4')
    case = CommandIOCase(
        name='dump_netcdf',
        run_in_lines=[
            ('dump_netcdf', [1, 'sed.nc', 'velocity']),
            ('run', BASE_N_STEPS),
            ('dump_netcdf', [1, 'sed.nc', 'velocity']),
        ],
        expected_output_files=['sed.nc'],
    )
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    _check_netcdf_result(result)

    with netcdf4.Dataset(tmp_path / 'sed.nc') as dataset:
        assert dataset.data_model == 'NETCDF3_64BIT_OFFSET'
        assert len(dataset.dimensions['frame']) == 2 * BASE_N_STEPS
        assert len(dataset.dimensions['atom']) == len(structure)
        assert dataset.variables['coordinates'].dtype == np.dtype('float32')
        assert dataset.variables['velocities'].dtype == np.dtype('float32')
        assert dataset.variables['type'].dimensions == ('frame', 'atom')
        assert dataset.variables['type'].shape == (2 * BASE_N_STEPS, len(structure))
        assert dataset.getncattr('gpumd_compression_level') == -1


@pytest.mark.filterwarnings('ignore:.*already contains files from an earlier calculation')
def test_dump_netcdf_appends_across_executions(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """Rerunning gpumd in a directory that already holds the file extends it rather than replacing
    it, as dump_xyz does. The frames written the first time have to survive untouched, which is
    what separates appending from starting over."""
    netcdf4 = pytest.importorskip('netCDF4')
    case = CommandIOCase(
        name='dump_netcdf_rerun',
        run_in_lines=[('dump_netcdf', [1, 'rerun.nc', 'velocity', 'precision', 'double'])],
        expected_output_files=['rerun.nc'])

    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    _check_netcdf_result(result)
    with netcdf4.Dataset(tmp_path / 'rerun.nc') as dataset:
        assert len(dataset.dimensions['frame']) == BASE_N_STEPS
        first_run = np.array(dataset.variables['coordinates'][:])

    # the same case again in the same directory, which rewrites run.in and model.xyz and leaves
    # the trajectory in place
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    _check_netcdf_result(result)
    with netcdf4.Dataset(tmp_path / 'rerun.nc') as dataset:
        assert len(dataset.dimensions['frame']) == 2 * BASE_N_STEPS, (
            'the second execution did not append to the existing file')
        both_runs = np.array(dataset.variables['coordinates'][:])
    np.testing.assert_array_equal(
        both_runs[:BASE_N_STEPS], first_run,
        err_msg='appending disturbed the frames written by the first execution')


@pytest.mark.filterwarnings('ignore:.*already contains files from an earlier calculation')
def test_dump_netcdf_appending_with_changed_quantities_is_rejected(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """A NetCDF layout is fixed when the file is created, so a rerun that asks for a different set
    of quantities cannot be appended. It has to stop with a message naming the file, since silently
    replacing the trajectory would lose the earlier one."""
    netcdf4 = pytest.importorskip('netCDF4')
    first = CommandIOCase(
        name='dump_netcdf_layout_first',
        run_in_lines=[('dump_netcdf', [1, 'layout.nc', 'velocity'])],
        expected_output_files=['layout.nc'])
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, first)
    _check_netcdf_result(result)

    second = CommandIOCase(
        name='dump_netcdf_layout_second',
        run_in_lines=[('dump_netcdf', [1, 'layout.nc', 'velocity', 'force'])],
        expected_output_files=[])
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, second)
    output = result.stdout + result.stderr
    _skip_without_netcdf(output)
    assert result.returncode != 0, 'appending with a different quantity set should be refused'
    assert 'layout.nc' in output, output
    assert 'set of quantities' in output, output

    # the trajectory that was already there must still be intact
    with netcdf4.Dataset(tmp_path / 'layout.nc') as dataset:
        assert len(dataset.dimensions['frame']) == BASE_N_STEPS
        assert 'forces' not in dataset.variables


def test_dump_netcdf_existing_file_that_is_not_netcdf_is_rejected(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """Appending means opening whatever is already there, so a file of that name that is not a
    NetCDF file at all has to be reported clearly rather than through a bare NetCDF error."""
    output_path = tmp_path / 'junk.nc'
    case = CommandIOCase(
        name='dump_netcdf_not_netcdf',
        run_in_lines=[('dump_netcdf', [1, 'junk.nc', 'velocity'])],
        expected_output_files=[])
    # written after the case is built but before the run, since run_command_io_case only writes
    # run.in and model.xyz
    output_path.write_bytes(b'an existing file that is not a NetCDF file')
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    output = result.stdout + result.stderr
    _skip_without_netcdf(output)
    assert result.returncode != 0, 'a non-NetCDF file of that name should be refused'
    assert 'junk.nc' in output, output
    assert output_path.read_bytes() == b'an existing file that is not a NetCDF file', (
        'the file that could not be opened was modified')


def test_dump_netcdf_appending_to_a_file_without_recorded_quantities_is_rejected(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """The quantities a file holds are compared through the gpumd_quantities attribute, and a
    file that does not carry it cannot be compared against at all. Such a file still opens, and
    still has every dimension and variable the append path looks up, so without an explicit check
    the run dies on a bare "NetCDF: Attribute not found" naming neither the file nor the remedy.

    The check has to sit at the comparison rather than where the file is opened, so that it does
    not preempt the more specific checks before it. The second half of this test is what pins that
    down: the same file with a different atom count must still report the atom count."""
    netcdf4 = pytest.importorskip('netCDF4')
    output_path = tmp_path / 'unrecorded.nc'
    case = CommandIOCase(
        name='dump_netcdf_unrecorded_quantities',
        run_in_lines=[('dump_netcdf', [1, 'unrecorded.nc', 'velocity'])],
        expected_output_files=['unrecorded.nc'])
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    _check_netcdf_result(result)

    # everything else about the file still matches, so the run reaches the quantity comparison
    with netcdf4.Dataset(output_path, 'a') as dataset:
        dataset.delncattr('gpumd_quantities')
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    output = result.stdout + result.stderr
    _skip_without_netcdf(output)
    assert result.returncode != 0, 'a file without recorded quantities should be refused'
    assert 'unrecorded.nc' in output, output
    assert 'does not record which quantities' in output, output

    # the same file read by a run with a different number of atoms: the earlier, more specific
    # check has to win, since claiming the quantities are unrecorded would describe the wrong
    # difference
    bigger = structure.repeat((2, 1, 1))
    result = run_command_io_case(
        tmp_path, bigger, model_path, model_type, gpumd_command, case)
    output = result.stdout + result.stderr
    _skip_without_netcdf(output)
    assert result.returncode != 0, 'a differing atom count should be refused'
    assert 'different number of atoms' in output, output


def test_dump_netcdf_without_velocity(
        tmp_path, structure, model_path, model_type, gpumd_command):
    netcdf4 = pytest.importorskip('netCDF4')
    case = CommandIOCase(
        name='dump_netcdf_positions',
        run_in_lines=[('dump_netcdf', [1, 'positions.nc'])],
        expected_output_files=['positions.nc'],
    )
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    _check_netcdf_result(result)

    with netcdf4.Dataset(tmp_path / 'positions.nc') as dataset:
        assert dataset.variables['coordinates'].dtype == np.dtype('float32')
        assert 'velocities' not in dataset.variables


def test_dump_netcdf_group_double_deflate(
        tmp_path, structure, model_path, model_type, gpumd_command):
    netcdf4 = pytest.importorskip('netCDF4')
    half = len(structure) // 2
    expected_group_size = len(structure) - half
    case = CommandIOCase(
        name='dump_netcdf_group',
        run_in_lines=[('dump_netcdf', [
            1, 'group.nc', 'group', 0, 1, 'velocity',
            'precision', 'double', 'compression', 'deflate', 1,
        ])],
        expected_output_files=['group.nc'],
        n_groups=2,
    )
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    _check_netcdf_result(result)

    with netcdf4.Dataset(tmp_path / 'group.nc') as dataset:
        coordinates = dataset.variables['coordinates']
        velocities = dataset.variables['velocities']
        assert dataset.data_model == 'NETCDF4'
        assert len(dataset.dimensions['frame']) == BASE_N_STEPS
        assert len(dataset.dimensions['atom']) == expected_group_size
        assert coordinates.dtype == np.dtype('float64')
        assert velocities.dtype == np.dtype('float64')
        assert coordinates.filters()['zlib']
        assert coordinates.filters()['complevel'] == 1
        assert velocities.filters()['zlib']
        assert dataset.getncattr('gpumd_grouping_method') == 0
        assert dataset.getncattr('gpumd_group_id') == 1


def test_dump_netcdf_multiple_groups_in_one_run(
        tmp_path, structure, model_path, model_type, gpumd_command):
    netcdf4 = pytest.importorskip('netCDF4')
    half = len(structure) // 2
    group_sizes = (half, len(structure) - half)
    filenames = ('group_0.nc', 'group_1.nc')
    intervals = (1, 2)
    case = CommandIOCase(
        name='dump_netcdf_multiple_groups',
        run_in_lines=[
            ('dump_netcdf', [intervals[0], filenames[0], 'group', 0, 0, 'velocity']),
            ('dump_netcdf', [intervals[1], filenames[1], 'group', 0, 1, 'velocity']),
        ],
        expected_output_files=list(filenames),
        n_groups=2,
    )
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    _check_netcdf_result(result)

    trajectories = []
    for group_id, (filename, group_size, interval) in enumerate(
            zip(filenames, group_sizes, intervals)):
        with netcdf4.Dataset(tmp_path / filename) as dataset:
            assert len(dataset.dimensions['frame']) == BASE_N_STEPS // interval
            assert len(dataset.dimensions['atom']) == group_size
            assert dataset.getncattr('gpumd_grouping_method') == 0
            assert dataset.getncattr('gpumd_group_id') == group_id
            assert 'velocities' in dataset.variables
            trajectories.append(dataset.variables['time'][:])

    np.testing.assert_allclose(
        trajectories[0][intervals[1] - 1::intervals[1]], trajectories[1])


def test_dump_netcdf_group_agrees_with_the_whole_system(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """Grouped output is gathered on the device, so that only the group's values cross to the host,
    while whole-system output is copied as it stands. That makes them two code paths through the
    same packing, and this is what pins them together.

    Both files come from one run, so they describe the same trajectory: the group file has to equal
    the rows of the whole-system file belonging to that group, exactly rather than approximately,
    since it is the same arithmetic on the same values. A gather that mis-indexes, or a stride
    confusion between the system size and the group size, shows up here as wrong values while
    every file still parses."""
    netcdf4 = pytest.importorskip('netCDF4')
    half = len(structure) // 2
    quantities = ['velocity', 'force', 'virial']
    case = CommandIOCase(
        name='dump_netcdf_group_vs_whole',
        run_in_lines=[
            # group 1, not group 0: group 0 holds the first half of the atoms, so a gather that
            # ignored the index list entirely would still produce the right answer for it
            ('dump_netcdf', [1, 'group.nc', 'group', 0, 1] + quantities
             + ['precision', 'double']),
            ('dump_netcdf', [1, 'whole.nc'] + quantities + ['precision', 'double']),
        ],
        expected_output_files=['group.nc', 'whole.nc'], n_groups=2)
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    _check_netcdf_result(result)

    with netcdf4.Dataset(tmp_path / 'group.nc') as group, \
            netcdf4.Dataset(tmp_path / 'whole.nc') as whole:
        # group 1 of the two-group model holds the atoms from `half` onwards
        expected_size = len(structure) - half
        assert len(group.dimensions['atom']) == expected_size
        assert len(whole.dimensions['atom']) == len(structure)
        for name in ('type', 'coordinates', 'velocities', 'forces', 'virial'):
            gathered = np.array(group.variables[name][:])
            complete = np.array(whole.variables[name][:])
            assert gathered.shape[1] == expected_size, name
            np.testing.assert_array_equal(
                gathered, complete[:, half:],
                err_msg=f'{name} gathered for the group differs from the whole-system rows')


def test_dump_netcdf_rejects_duplicate_filename_in_one_run(
        tmp_path, structure, model_path, model_type, gpumd_command):
    pytest.importorskip('netCDF4')
    case = CommandIOCase(
        name='dump_netcdf_duplicate_filename',
        run_in_lines=[
            ('dump_netcdf', [1, 'duplicate.nc', 'group', 0, 0, 'velocity']),
            ('dump_netcdf', [1, 'duplicate.nc', 'group', 0, 1, 'velocity']),
        ],
        n_groups=2,
    )
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    output = result.stdout + result.stderr
    _skip_without_netcdf(output)
    assert result.returncode != 0
    assert 'dump_netcdf filenames must be unique within one run.' in output


def test_dump_netcdf_rotates_general_cell(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """AMBER NetCDF stores only cell lengths and angles, so a general GPUMD cell is written in a
    restricted orientation and everything directional has to be rotated with it. Positions,
    velocities, forces and unwrapped positions follow as vectors, the per-atom virial and the BEC
    as rank-2 tensors.

    The dump_xyz file written by the same run is the reference, so the two describe the same
    trajectory and the rotation is the only thing between them. Getting the tensor rule wrong --
    rotating one-sidedly, or transposing -- is silent otherwise, since the values stay the same
    size and the file still reads back."""
    netcdf4 = pytest.importorskip('netCDF4')
    rotated_structure = structure.copy()
    general_cell = rotated_structure.cell.array.copy()
    general_cell[0] += 0.05 * general_cell[2]
    rotated_structure.set_cell(general_cell, scale_atoms=True)
    rotated_structure.rotate(17.0, 'y', center=(0.0, 0.0, 0.0), rotate_cell=True)

    quantities = ['velocity', 'force', 'unwrapped_position', 'virial']
    if model_type != 'nep':
        quantities.append('bec')
    case = CommandIOCase(
        name='dump_netcdf_general_cell',
        run_in_lines=[
            ('dump_xyz', [1, 'reference.xyz'] + quantities + ['precision', 'double']),
            ('dump_netcdf', [1, 'general-cell.nc'] + quantities + ['precision', 'double']),
        ],
        expected_output_files=['reference.xyz', 'general-cell.nc'],
    )
    result = run_command_io_case(
        tmp_path, rotated_structure, model_path, model_type, gpumd_command, case)
    _check_netcdf_result(result)

    reference = read(tmp_path / 'reference.xyz', index=0)
    netcdf_values = {}
    with netcdf4.Dataset(tmp_path / 'general-cell.nc') as dataset:
        lengths = dataset.variables['cell_lengths'][0]
        angles = dataset.variables['cell_angles'][0]
        netcdf_cell = Cell.fromcellpar(np.concatenate((lengths, angles))).array
        for name in ('coordinates', 'velocities', 'forces', 'unwrapped_coordinates', 'virial'):
            netcdf_values[name] = np.array(dataset.variables[name][0])
        if 'bec' in quantities:
            netcdf_values['bec'] = np.array(dataset.variables['bec'][0])

    reference_cell = reference.cell.array
    # v_netcdf = v_reference @ rotation_transpose, so the rotation itself is its transpose
    rotation_transpose = np.linalg.solve(reference_cell, netcdf_cell)
    rotation = rotation_transpose.T
    assert not np.allclose(reference_cell, netcdf_cell)
    assert not np.allclose(angles, (90.0, 90.0, 90.0))
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-10)

    vectors = [
        ('coordinates', reference.positions, 1.0, 1.0e-6),
        # dump_xyz writes velocities in A/fs and NetCDF in A/ps
        ('velocities', reference.arrays['vel'], 1000.0, 1.0e-5),
        # dump_xyz writes forces in eV/A, the AMBER convention specifies kcal/mol/A
        ('forces', reference.get_forces(), EV_TO_KCAL_PER_MOL, 1.0e-6),
        ('unwrapped_coordinates', reference.arrays['unwrapped_position'], 1.0, 1.0e-6),
    ]
    for name, reference_value, scale, tolerance in vectors:
        np.testing.assert_allclose(
            netcdf_values[name], reference_value @ rotation_transpose * scale, atol=tolerance,
            err_msg=f'{name} is not the reference rotated into the NetCDF cell')

    for name in ('virial', 'bec'):
        if name not in netcdf_values:
            continue
        reference_tensor = reference.arrays[name].reshape(-1, 3, 3)
        expected = np.einsum('ij,ajk,lk->ail', rotation, reference_tensor, rotation)
        assert not np.allclose(netcdf_values[name], reference_tensor, atol=1.0e-6), (
            f'{name} is unchanged by the rotation, so this comparison proves nothing')
        np.testing.assert_allclose(
            netcdf_values[name], expected, atol=1.0e-6,
            err_msg=f'{name} does not follow the cell as R T R^T')


# quantity keyword -> (variable, dimensions, units), as the documentation tabulates them. `bec`
# is left out because it needs a charge model, and is covered on its own below.
NETCDF_QUANTITY_LAYOUT = {
    'velocity': ('velocities', ('frame', 'atom', 'spatial'), 'angstrom/picosecond'),
    'force': ('forces', ('frame', 'atom', 'spatial'), 'kilocalorie/mole/angstrom'),
    'unwrapped_position': ('unwrapped_coordinates', ('frame', 'atom', 'spatial'), 'angstrom'),
    'potential': ('potential_energy', ('frame', 'atom'), 'eV'),
    'charge': ('charge', ('frame', 'atom'), 'e'),
    'virial': ('virial', ('frame', 'atom', 'spatial', 'spatial'), 'eV'),
    'mass': ('mass', ('atom',), 'amu'),
    'group_labels': ('group_labels', ('atom', 'grouping_method'), None),
}


def test_dump_netcdf_quantity_layout(tmp_path, structure, model_path, model_type, gpumd_command):
    """Each quantity has to land in the variable, with the dimensions and units, that the
    documentation promises, since that layout is the whole interface to the file. The mass and
    the group labels are constant over a run and carry no frame dimension."""
    netcdf4 = pytest.importorskip('netCDF4')
    quantities = list(NETCDF_QUANTITY_LAYOUT)
    case = CommandIOCase(
        name='dump_netcdf_quantities',
        run_in_lines=[
            ('dump_netcdf', [1, 'all.nc'] + quantities + ['precision', 'double']),
            # the same set in the opposite order, to pin down that neither the file layout nor
            # the recorded quantity list follows the order the arguments came in
            ('dump_netcdf', [1, 'reversed.nc'] + quantities[::-1] + ['precision', 'double']),
        ],
        expected_output_files=['all.nc', 'reversed.nc'], n_groups=2)
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    _check_netcdf_result(result)

    with netcdf4.Dataset(tmp_path / 'all.nc') as dataset:
        for quantity, (name, dimensions, units) in NETCDF_QUANTITY_LAYOUT.items():
            assert name in dataset.variables, f'{quantity} did not produce a {name} variable'
            variable = dataset.variables[name]
            assert variable.dimensions == dimensions, quantity
            expected_dtype = np.dtype('int32') if units is None else np.dtype('float64')
            assert variable.dtype == expected_dtype, quantity
            if units is None:
                assert not hasattr(variable, 'units'), quantity
            else:
                assert variable.units == units, quantity
        assert len(dataset.dimensions['grouping_method']) == 1
        assert len(dataset.dimensions['atom']) == len(structure)
        recorded = dataset.getncattr('gpumd_quantities')
        assert sorted(recorded.split()) == sorted(quantities)

    with netcdf4.Dataset(tmp_path / 'reversed.nc') as dataset:
        assert dataset.getncattr('gpumd_quantities') == recorded


def test_dump_netcdf_writes_only_the_requested_quantities(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """A quantity that was not asked for must not be in the file at all."""
    netcdf4 = pytest.importorskip('netCDF4')
    case = CommandIOCase(
        name='dump_netcdf_one_quantity',
        run_in_lines=[('dump_netcdf', [1, 'force.nc', 'force'])],
        expected_output_files=['force.nc'])
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    _check_netcdf_result(result)

    with netcdf4.Dataset(tmp_path / 'force.nc') as dataset:
        assert 'forces' in dataset.variables
        # the positions and types are unconditional, everything else follows the arguments
        assert set(dataset.variables) == {
            'spatial', 'cell_spatial', 'cell_angular', 'time', 'cell_lengths', 'cell_angles',
            'type', 'coordinates', 'forces'}
        assert 'grouping_method' not in dataset.dimensions
        assert dataset.getncattr('gpumd_quantities') == 'force'


def test_dump_netcdf_is_readable_by_mdanalysis(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """The file declares Conventions=AMBER, so readers that believe it are entitled to AMBER units
    for the variables AMBER defines. MDAnalysis enforces that in NCDFReader.__init__ and raises
    rather than converting, so a single mislabelled variable makes the whole trajectory
    unopenable, positions included -- not merely the quantity in question unreadable.

    `force` is the variable that has to be converted, since GPUMD works in eV/A and the convention
    specifies kcal/mol/A. This asks for the AMBER-defined quantities together, so relabelling any
    of them is caught here."""
    mda = pytest.importorskip('MDAnalysis')
    pytest.importorskip('netCDF4')
    case = CommandIOCase(
        name='dump_netcdf_mdanalysis',
        run_in_lines=[('dump_netcdf', [1, 'amber.nc', 'velocity', 'force'])],
        expected_output_files=['amber.nc'])
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    _check_netcdf_result(result)

    universe = mda.Universe(str(tmp_path / 'amber.nc'), format='NCDF')
    assert len(universe.atoms) == len(structure)
    assert len(universe.trajectory) == BASE_N_STEPS
    frame = universe.trajectory[0]
    assert frame.has_forces, 'MDAnalysis did not expose the forces'
    assert np.all(np.isfinite(frame.forces))
    assert np.all(np.isfinite(frame.positions))
    # kcal/(mol*Angstrom) is MDAnalysis's own base force unit, so it passes the values through
    # unchanged; the numeric value of the conversion is pinned by the general-cell test above
    assert universe.trajectory.units['force'] == 'kcal/(mol*Angstrom)'
    assert np.max(np.abs(frame.forces)) > 0


def test_dump_netcdf_bec(tmp_path, structure, model_path, model_type, gpumd_command):
    """BEC is a rank-2 tensor per atom and needs a charge model, so it is checked apart from the
    other quantities."""
    netcdf4 = pytest.importorskip('netCDF4')
    if model_type == 'nep':
        pytest.skip('BEC requires a qNEP charge model')
    case = CommandIOCase(
        name='dump_netcdf_bec',
        run_in_lines=[('dump_netcdf', [1, 'bec.nc', 'bec', 'precision', 'double'])],
        expected_output_files=['bec.nc'])
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    _check_netcdf_result(result)

    with netcdf4.Dataset(tmp_path / 'bec.nc') as dataset:
        variable = dataset.variables['bec']
        assert variable.dimensions == ('frame', 'atom', 'spatial', 'spatial')
        assert variable.units == 'e'
        assert variable.shape == (BASE_N_STEPS, len(structure), 3, 3)
        assert np.all(np.isfinite(np.array(variable[0])))


def test_dump_netcdf_bec_without_a_charge_model_is_rejected(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """BEC is only defined for a qNEP model trained against it, so asking for it otherwise has to
    be refused rather than writing zeros."""
    if model_type != 'nep':
        pytest.skip('this is about the models that cannot produce BEC')
    case = CommandIOCase(
        name='dump_netcdf_bec_no_charge_model',
        run_in_lines=[('dump_netcdf', [1, 'bec.nc', 'bec'])],
        expected_output_files=[])
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    output = result.stdout + result.stderr
    _skip_without_netcdf(output)
    assert result.returncode != 0, 'bec without a charge model should be refused'
    assert 'BEC' in output, output


def test_dump_netcdf_group_labels_without_a_grouping_method_is_rejected(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """With no grouping method the grouping_method dimension would have length zero, which
    NC_UNLIMITED makes indistinguishable from an unlimited dimension, the same trap the
    empty-group check guards against."""
    case = CommandIOCase(
        name='dump_netcdf_group_labels_no_groups',
        run_in_lines=[('dump_netcdf', [1, 'labels.nc', 'group_labels'])],
        expected_output_files=[], n_groups=0)
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    output = result.stdout + result.stderr
    _skip_without_netcdf(output)
    assert result.returncode != 0, 'group_labels without a grouping method should be refused'
    assert 'grouping method' in output, output


def _netcdf_per_atom_virial(directory, structure, model_path, model_type, gpumd_command, kspace):
    """Runs one qNEP single point with the given reciprocal-space method and returns the per-atom
    virial from the NetCDF file as an (natoms, 9) array."""
    netcdf4 = pytest.importorskip('netCDF4')
    directory.mkdir()
    case = CommandIOCase(
        name=f'dump_netcdf_virial_{kspace}',
        prelude_lines=[('kspace', kspace)],
        run_in_lines=[('dump_netcdf', [1, 'virial.nc', 'virial', 'precision', 'double'])],
        expected_output_files=['virial.nc'])
    result = run_command_io_case(
        directory, structure, model_path, model_type, gpumd_command, case)
    _check_netcdf_result(result)
    with netcdf4.Dataset(directory / 'virial.nc') as dataset:
        return np.array(dataset.variables['virial'][0]).reshape(len(structure), 9)


def test_dump_netcdf_per_atom_virial_agrees_between_pppm_and_ewald(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """The dump_netcdf counterpart of the dump_xyz test above, guarding the same scan in
    check_need_peratom_virial in utilities/read_file.cu, which now has to recognize dump_netcdf
    lines as well as dump_xyz ones.

    The flag it sets gates only the PPPM reciprocal-space contribution, so the two methods have to
    be compared against each other: with dump_netcdf missing from the scan, the PPPM virial
    silently loses its k-space part while Ewald keeps it, and the file still looks perfectly
    well-formed."""
    if model_type == 'nep':
        pytest.skip('a charge model is needed for there to be a reciprocal-space contribution')

    pppm = _netcdf_per_atom_virial(
        tmp_path / 'pppm', structure, model_path, model_type, gpumd_command, 'pppm')
    ewald = _netcdf_per_atom_virial(
        tmp_path / 'ewald', structure, model_path, model_type, gpumd_command, 'ewald')

    assert pppm.shape == ewald.shape == (len(structure), 9)
    assert np.max(np.abs(ewald)) > 0, 'the reference virial is identically zero, nothing to compare'
    np.testing.assert_allclose(pppm, ewald, rtol=1e-2, atol=5e-3)


INVALID_DUMP_NETCDF_ARGUMENTS = [
    ([1, 'f.nc', 'velocities'], 'Unrecognized argument'),
    ([1, 'f.nc', 'force', 'force'], 'more than once'),
    ([1, 'f.nc', 'group', 0, 0, 'group', 0, 0], 'more than once'),
    ([1, 'f.nc', 'velocity', 'velocity'], 'more than once'),
    ([1, 'f.nc', 'precision', 'double', 'precision', 'single'], 'more than once'),
    ([1, 'f.nc', 'compression', 'none', 'compression', 'none'], 'more than once'),
    ([1, 'f.nc', 'precision', 'triple'], 'Invalid precision'),
    # the old syntax used -1 to mean the whole system, which is now spelled by omitting the option
    ([1, 'f.nc', 'group', -1, 0], 'Grouping method'),
    ([1, 'f.nc', 'group', 0], 'Not enough arguments'),
    ([1, 'f.nc', 'compression', 'deflate'], 'deflate level is required'),
    ([1, 'f.nc', 'compression', 'deflate', 10], 'between 0 and 9'),
    ([1, 'f.nc', 'compression', 'gzip'], "'none' or 'deflate"),
    ([1], 'at least 2 parameters'),
    ([0, 'f.nc'], 'dump interval'),
]


@pytest.mark.parametrize('args, expected_message', INVALID_DUMP_NETCDF_ARGUMENTS)
def test_invalid_dump_netcdf_arguments_are_rejected(
        tmp_path, structure, model_path, model_type, gpumd_command, args, expected_message):
    """The dump_xyz rejection table applied to dump_netcdf. As there, the asserted substring
    matters as much as the exit code, since a test that passes because gpumd died for an unrelated
    reason is worse than no test."""
    case = CommandIOCase(
        name='dump_netcdf_invalid', run_in_lines=[('dump_netcdf', args)],
        expected_output_files=[], n_groups=1)
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    output = result.stdout + result.stderr
    _skip_without_netcdf(output)
    assert result.returncode != 0, f'dump_netcdf {args} unexpectedly succeeded'
    assert expected_message in output, (
        f'dump_netcdf {args} did not report {expected_message!r}\nstdout:\n{result.stdout}')


def test_dump_netcdf_old_positional_form_is_rejected(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """An input file written for the old positional form must be told what to write instead.

    Without the check this is not silent, but it is misleading: removing it and rerunning this
    case reports "dump interval should > 0", because the old grouping method is read as the
    interval. A positive grouping method fares no better, as the old group id then becomes the
    file name and the old interval an unrecognized argument. The point of the check is the
    message, so the message is what is asserted."""
    case = CommandIOCase(
        name='dump_netcdf_old_form',
        run_in_lines=[('dump_netcdf', [-1, 0, 1, 1, 'movie.nc'])],
        expected_output_files=[])
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    output = result.stdout + result.stderr
    _skip_without_netcdf(output)
    assert result.returncode != 0, 'the old positional form should be refused'
    assert 'no longer takes' in output, output
    assert 'dump_netcdf <interval> <filename>' in output, output


def test_dump_netcdf_rejects_an_empty_group(
        tmp_path, structure, model_path, model_type, gpumd_command):
    """An empty group must be refused rather than written, and this is the case where leaving the
    check out would be silent rather than loud. NC_UNLIMITED is 0, so defining the atom dimension
    with a length of zero does not fail. Removing the check and rerunning this case confirms both
    halves of that: with `compression deflate 1` gpumd exits 0 and writes a NETCDF4 file whose
    `atom` dimension is a second unlimited dimension of length zero and whose coordinates have
    shape (frames, 0, 3), and without compression the run gets as far as the first frame before
    dying with "NC_UNLIMITED size already in use", because the classic format allows only one
    unlimited dimension. parse_group checks the bounds but not the size, so dump_netcdf checks
    after calling it.

    The grouping puts every atom in group 1, which leaves group 0 of the same grouping method
    empty while keeping it in bounds -- GPUMD takes the number of groups from the largest label
    it sees."""
    case = CommandIOCase(
        name='dump_netcdf_empty_group',
        run_in_lines=[('dump_netcdf', [1, 'empty.nc', 'group', 0, 0])],
        expected_output_files=[],
        groupings=[[[], list(range(len(structure)))]])
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    output = result.stdout + result.stderr
    _skip_without_netcdf(output)
    assert result.returncode != 0, 'an empty group should be refused'
    assert 'cannot output an empty group' in output, output
    assert not (tmp_path / 'empty.nc').exists(), 'a file was written for an empty group'


def test_dump_observer(tmp_path, structure, model_path, model_type, gpumd_command):
    """dump_observer needs 2+ NEP potentials with identical species/order; reusing the same
    model file for both is a valid, if physically redundant, way to satisfy that for a smoke
    test. Uses 'average' mode (single observer.out/observer.xyz) rather than 'observe' mode
    (per-potential observerN.out/xyz) since both exercise the same underlying output-writing
    code path and 'average' needs fewer expected files."""
    case = CommandIOCase(
        name='dump_observer',
        run_in_lines=[
            ('potential', str(model_path)),
            ('potential', str(model_path)),
            ('dump_observer', ['average', 1, 1, 1, 1]),
        ],
        expected_output_files=['observer.out'],
        parse_check=_check_thermo_format,
    )
    run_and_check(tmp_path, structure, model_path, model_type, gpumd_command, case)
