"""Checks of the energy-difference loss of nep, activated by lambda_d > 0 and ediff.in.

A line of ediff.in combines the total energies of named structures with coefficients, and the
combination has to be balanced in the number of atoms of each type. The four structures of the
training fixture share one composition. Unless a test says otherwise, the first two form train.xyz
and the last two test.xyz. One generation per output_interval yields one row of loss.out per run.
"""
import math
import re
import subprocess

import numpy as np
import pytest

from conftest import TRAINING_DIR

pytestmark = pytest.mark.fast

KEYWORDS = {
    'type': '3 Ba Ti O',
    'cutoff': '6 4',
    'n_max': '4 4',
    'basis_size': '8 8',
    'l_max': '4 0 0',
    'neuron': '10',
    'batch': '1000',
    'generation': '10',
    'output_interval': '10',
}

MASTER_NEP_COLUMNS = (
    'generation total L1 L2 rmse_energy_train rmse_force_train rmse_virial_train'
    ' rmse_energy_test rmse_force_test rmse_virial_test'
).split()


def read_frames(path):
    """The frames of an extended XYZ file, each as its list of lines."""
    lines = path.read_text().splitlines()
    frames = []
    start = 0
    while start < len(lines):
        num_atoms = int(lines[start])
        frames.append(lines[start : start + num_atoms + 2])
        start += num_atoms + 2
    return frames


def write_frames(path, frames, names):
    """Write the frames with name=<name> appended to each comment line."""
    text = ''
    for frame, name in zip(frames, names):
        text += '\n'.join([frame[0], f'{frame[1]} name={name}'] + frame[2:]) + '\n'
    path.write_text(text)


def total_reference_energy(frame):
    for token in frame[1].split():
        if token.lower().startswith('energy='):
            return float(token.split('=')[1])
    raise ValueError('no energy in comment line')


def with_atoms(frame, atom_lines, energy):
    """The frame with its atoms replaced and the given total reference energy."""
    comment = re.sub(r'energy=\S+', f'energy={energy:.8f}', frame[1])
    return [str(len(atom_lines)), comment] + atom_lines


def write_nep_in(directory, overrides):
    keywords = dict(KEYWORDS)
    keywords.update(overrides or {})
    text = ''.join(f'{key} {value}\n' for key, value in keywords.items() if value is not None)
    (directory / 'nep.in').write_text(text)


def setup_directory(directory, ediff_lines, overrides=None):
    """Write train.xyz with the structures S0 and S1, test.xyz with S2 and S3, nep.in, and ediff.in
    unless ediff_lines is None. Returns the frames."""
    frames = read_frames(TRAINING_DIR / 'train.xyz')
    assert len(frames) == 4
    write_frames(directory / 'train.xyz', frames[:2], ['S0', 'S1'])
    write_frames(directory / 'test.xyz', frames[2:], ['S2', 'S3'])
    write_nep_in(directory, overrides)
    if ediff_lines is not None:
        (directory / 'ediff.in').write_text('# combination [weight]\n' + ediff_lines)
    return frames


def run_nep(directory, nep_command):
    return subprocess.run(
        [nep_command], cwd=directory, capture_output=True, text=True, check=False
    )


def read_loss_out(directory):
    """The column names of loss.out and its rows as lists of floats."""
    columns = None
    rows = []
    for line in (directory / 'loss.out').read_text().splitlines():
        if line.startswith('# columns'):
            columns = line.split()[2:]
        elif not line.startswith('#') and line.strip():
            rows.append([float(value) for value in line.split()])
    return columns, rows


def read_total_energies(path, num_atoms):
    """Total predicted energies from the per-atom energies in an energy_*.out file."""
    energies_per_atom = [float(line.split()[0]) for line in path.read_text().splitlines()]
    return [e * n for e, n in zip(energies_per_atom, num_atoms)]


@pytest.mark.parametrize('overrides', [{}, {'charge_mode': '1', 'zbl': '1.5'}], ids=['nep', 'qnep'])
def test_ediff_columns_are_appended(tmp_path, nep_command, overrides):
    """The two ediff columns follow all other columns, and every row has one value per column."""
    setup_directory(tmp_path, 's0 - s1\ns2 - s3\n', {'lambda_d': '1', **overrides})
    result = run_nep(tmp_path, nep_command)
    assert result.returncode == 0, result.stdout + result.stderr

    columns, rows = read_loss_out(tmp_path)
    assert columns[-2:] == ['rmse_ediff_train', 'rmse_ediff_test']
    assert len(rows) == 1
    assert all(len(row) == len(columns) for row in rows)


def test_ediff_rmse_matches_predicted_energies(tmp_path, nep_command):
    """The test column equals the weighted error of the predicted energy difference in
    energy_test.out, which the report writes from the same evaluation. The report writes no
    energy_train.out, so the train column is only checked to be finite."""
    frames = setup_directory(tmp_path, 'S0 - S1\nS2 - S3 4\n', {'lambda_d': '1'})
    result = run_nep(tmp_path, nep_command)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = 'ediff.in: 2 combinations, 1 in train.xyz, 1 in test.xyz (0 in both), 0 skipped.'
    assert summary in result.stdout

    columns, rows = read_loss_out(tmp_path)
    values = dict(zip(columns, rows[-1]))
    assert math.isfinite(values['rmse_ediff_train'])

    energies = read_total_energies(tmp_path / 'energy_test.out', [40, 40])
    predicted = energies[0] - energies[1]
    reference = total_reference_energy(frames[2]) - total_reference_energy(frames[3])
    # energy_test.out carries six significant digits, about 1e-4 eV per atom, and the weight of 4
    # doubles the error; the residual measured with this setup is 2.5e-3 eV at weight 1
    expected = 2 * abs(predicted - reference)
    assert values['rmse_ediff_test'] == pytest.approx(expected, abs=1e-2)


def test_formation_energy_of_unequal_sizes(tmp_path, nep_command):
    """A vacancy against the perfect cell and half an O2 molecule is balanced in composition, so
    any per-atom energy offset cancels. The train column, computed before nep removes that offset
    from the model, then equals the test column on the same structures, computed after it."""
    frames = read_frames(TRAINING_DIR / 'train.xyz')
    bulk = frames[0]
    oxygen_lines = [line for line in bulk[2:] if line.split()[0] == 'O']
    vacancy = with_atoms(bulk, bulk[2:-1], total_reference_energy(bulk) * 39 / 40)
    oxygen_molecule = with_atoms(bulk, oxygen_lines[:2], -9.0)
    assert bulk[-1].split()[0] == 'O'
    structures = [bulk, vacancy, oxygen_molecule]
    names = ['bulk', 'vac', 'O2']
    write_frames(tmp_path / 'train.xyz', structures + [frames[1]], names + ['other'])
    write_frames(tmp_path / 'test.xyz', structures, names)
    write_nep_in(tmp_path, {'lambda_d': '1'})
    (tmp_path / 'ediff.in').write_text('vac + 0.5*O2 - bulk\n')
    result = run_nep(tmp_path, nep_command)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = 'ediff.in: 1 combinations, 1 in train.xyz, 1 in test.xyz (1 in both), 0 skipped.'
    assert summary in result.stdout

    columns, rows = read_loss_out(tmp_path)
    values = dict(zip(columns, rows[-1]))
    energies = read_total_energies(tmp_path / 'energy_test.out', [40, 39, 2])
    predicted = energies[1] + 0.5 * energies[2] - energies[0]
    reference = sum(
        c * total_reference_energy(s) for c, s in zip([-1, 1, 0.5], structures)
    )
    assert values['rmse_ediff_test'] == pytest.approx(abs(predicted - reference), abs=1e-2)
    assert values['rmse_ediff_train'] == pytest.approx(values['rmse_ediff_test'], abs=1e-2)


def test_layout_without_lambda_d_is_unchanged(tmp_path, nep_command):
    """Without lambda_d an ediff.in is ignored, and loss.out has the columns of a run without it."""
    setup_directory(tmp_path, 's0 - s1\n')
    result = run_nep(tmp_path, nep_command)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ediff.in is ignored because lambda_d = 0.' in result.stdout

    columns, rows = read_loss_out(tmp_path)
    assert columns == MASTER_NEP_COLUMNS
    assert all(len(row) == len(MASTER_NEP_COLUMNS) for row in rows)


INPUT_ERRORS = {
    'missing ediff.in': (None, {}, 'lambda_d > 0 requires the file ediff.in.'),
    'no train combination': (
        's0 - unknown\ns2 - s3\n',
        {},
        'No combination in ediff.in has all its structures in train.xyz',
    ),
    'dipole model': ('s0 - s1\n', {'model_type': '1'}, 'lambda_d is only supported'),
    'nan lambda_d': ('s0 - s1\n', {'lambda_d': 'nan'}, 'should be a finite number >= 0'),
    'one structure': (
        's0 - s1\ns0\n',
        {},
        'ediff.in line 3: a combination needs at least two structures',
    ),
    'pair syntax': ('s0 s1\n', {}, "ediff.in line 2: expected + or - before 's1'"),
    'extra field': ('s0 - s1 1 2\n', {}, "ediff.in line 2: expected + or - before '1'"),
    'missing term': ('s0 -\n', {}, 'ediff.in line 2: expected a structure after -'),
    'invalid coefficient': ('s0 - x*s1\n', {}, "ediff.in line 2: invalid coefficient 'x'"),
    'zero coefficient': ('s0 - 0*s1\n', {}, "ediff.in line 2: invalid coefficient '0'"),
    'invalid weight': ('s0 - s1 0\n', {}, "ediff.in line 2: invalid weight '0'"),
    'nan weight': ('s0 - s1 nan\n', {}, "ediff.in line 2: invalid weight 'nan'"),
    'weight below float range': ('s0 - s1 1e-50\n', {}, "invalid weight '1e-50'"),
    'repeated name': ('s0 - S0\n', {}, 'ediff.in line 2: s0 occurs more than once'),
}


@pytest.mark.parametrize(
    'ediff_lines, overrides, expected_text', list(INPUT_ERRORS.values()), ids=list(INPUT_ERRORS)
)
def test_input_errors(tmp_path, nep_command, ediff_lines, overrides, expected_text):
    setup_directory(tmp_path, ediff_lines, {'lambda_d': '1', **overrides})
    result = run_nep(tmp_path, nep_command)
    assert result.returncode != 0
    assert expected_text in result.stderr


def test_test_combination_enters_only_the_test_column(tmp_path, nep_command):
    """A weight of 1e8 for the combination of test.xyz makes its error dominate any RMSE it
    enters, so that it has to raise the test column far above the train column. A combination
    split across the two files is skipped, and stdout names only its line."""
    setup_directory(tmp_path, 's0 - s1\ns2 - s3 1e8\ns0 - s2\n', {'lambda_d': '1'})
    result = run_nep(tmp_path, nep_command)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = 'ediff.in: 3 combinations, 1 in train.xyz, 1 in test.xyz (0 in both), 1 skipped.'
    assert summary in result.stdout
    assert 'e.g. line 4.' in result.stdout
    assert re.search(r'\bs[0-3]\b', result.stdout) is None

    columns, rows = read_loss_out(tmp_path)
    values = dict(zip(columns, rows[-1]))
    assert values['rmse_ediff_test'] > 100 * values['rmse_ediff_train']


def remove_last_atom(frame):
    return [str(int(frame[0]) - 1), frame[1]] + frame[2:-1]


def swap_first_species(frame, species):
    """The frame with the species of its first atom replaced."""
    tokens = frame[2].split()
    return frame[:2] + [' '.join([species] + tokens[1:])] + frame[3:]


@pytest.mark.parametrize(
    'modify, xyz_filename',
    [
        (remove_last_atom, 'train.xyz'),
        (lambda frame: swap_first_species(frame, 'Ti'), 'train.xyz'),
        (remove_last_atom, 'test.xyz'),
    ],
    ids=['fewer atoms', 'swapped species', 'test combination'],
)
def test_unbalanced_combination_is_an_input_error(tmp_path, nep_command, modify, xyz_filename):
    """A combination needs the same number of atoms of each type on both sides, since a uniform or
    per-type energy offset would otherwise enter the term."""
    frames = setup_directory(tmp_path, 's0 - s1\ns2 - s3\n', {'lambda_d': '1'})
    if xyz_filename == 'train.xyz':
        write_frames(tmp_path / 'train.xyz', [frames[0], modify(frames[1])], ['S0', 'S1'])
        expected_text = 'ediff.in line 2: the combination is not balanced in train.xyz'
    else:
        write_frames(tmp_path / 'test.xyz', [frames[2], modify(frames[3])], ['S2', 'S3'])
        expected_text = 'ediff.in line 3: the combination is not balanced in test.xyz'
    result = run_nep(tmp_path, nep_command)
    assert result.returncode != 0
    assert expected_text in result.stderr


@pytest.mark.parametrize(
    'train_names, test_names, expected_text',
    [
        (['"S 0"', 'S1'], ['S2', 'S3'], 'train.xyz line 2: the name "s 0" contains whitespace'),
        (['S0', 'S1'], ['S2', '"S 3"'], 'test.xyz line 44: the name "s 3" contains whitespace'),
        (['S0', 's0'], ['S2', 'S3'], 'the name s0 occurs on more than one structure of train.xyz'),
        (['S0', 'S1'], ['S2', 'S2'], 'the name s2 occurs on more than one structure of test.xyz'),
    ],
    ids=['whitespace in train.xyz', 'whitespace in test.xyz', 'repeated in train', 'repeated in test'],
)
def test_invalid_names_are_input_errors(
    tmp_path, nep_command, train_names, test_names, expected_text
):
    """ediff.in separates its fields by whitespace, so a name with whitespace cannot be referred to,
    and a repeated name would make a reference ambiguous."""
    frames = setup_directory(tmp_path, 's0 - s1\n', {'lambda_d': '1'})
    write_frames(tmp_path / 'train.xyz', frames[:2], train_names)
    write_frames(tmp_path / 'test.xyz', frames[2:], test_names)
    result = run_nep(tmp_path, nep_command)
    assert result.returncode != 0
    assert expected_text in result.stderr


def test_names_are_not_read_without_lambda_d(tmp_path, nep_command):
    """Without the energy-difference loss, names with whitespace and repeated names are ignored."""
    frames = setup_directory(tmp_path, None)
    write_frames(tmp_path / 'train.xyz', frames[:2], ['"S 0"', '"S 0"'])
    result = run_nep(tmp_path, nep_command)
    assert result.returncode == 0, result.stdout + result.stderr


def test_quoted_name_without_whitespace(tmp_path, nep_command):
    frames = setup_directory(tmp_path, 's0 - s1\n', {'lambda_d': '1'})
    write_frames(tmp_path / 'train.xyz', frames[:2], ['"S0"', 'S1'])
    result = run_nep(tmp_path, nep_command)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ediff.in: 1 combinations, 1 in train.xyz' in result.stdout


def test_combined_structures_share_a_batch(tmp_path, nep_command):
    """nep deals the structures, sorted by energy per atom, round-robin into the batches, which
    would put the two structures of each pair, adjacent in energy, into different batches. The
    structures of each combination are kept in one batch instead."""
    frames = setup_directory(tmp_path, None, {'lambda_d': '1', 'batch': '2'})
    energies_per_atom = [-5.0, -4.999, -4.5, -4.499]
    frames = [with_energy(frame, 40 * e) for frame, e in zip(frames, energies_per_atom)]
    write_frames(tmp_path / 'train.xyz', frames, ['S0', 'S1', 'S2', 'S3'])
    (tmp_path / 'test.xyz').unlink()
    (tmp_path / 'ediff.in').write_text('s0 - s1\ns2 - s3\n')
    result = run_nep(tmp_path, nep_command)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Number of batches = 2' in result.stdout
    assert 'span' not in result.stdout


def test_batches_left_empty_are_dropped(tmp_path, nep_command):
    """Three structures linked by combinations form one group, which fills one of three batches."""
    frames = setup_directory(tmp_path, None, {'lambda_d': '1', 'batch': '1'})
    write_frames(tmp_path / 'train.xyz', frames[:3], ['S0', 'S1', 'S2'])
    (tmp_path / 'test.xyz').unlink()
    (tmp_path / 'ediff.in').write_text('s0 - s1\ns1 - s2\n')
    result = run_nep(tmp_path, nep_command)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Number of batches reduced to 1' in result.stdout
    assert 'Warning: ediff.in links 3 structures into one group' in result.stdout


def with_energy(frame, energy):
    return [frame[0], re.sub(r'energy=\S+', f'energy={energy:.10f}', frame[1])] + frame[2:]


def make_supercell(frame, n):
    """The frame repeated n times along each cell vector."""
    match = re.search(r'Lattice="([^"]+)"', frame[1])
    cell = np.array(match.group(1).split(), float).reshape(3, 3)
    atom_lines = []
    for shift in (i * cell[0] + j * cell[1] + k * cell[2] for i in range(n) for j in range(n) for k in range(n)):
        for line in frame[2:]:
            tokens = line.split()
            position = np.array(tokens[1:4], float) + shift
            atom_lines.append(' '.join([tokens[0]] + [f'{x:.8f}' for x in position] + tokens[4:]))
    lattice = ' '.join(f'{x:.10f}' for x in (n * cell).ravel())
    return [str(len(atom_lines)), frame[1].replace(match.group(1), lattice)] + atom_lines


def test_combination_of_large_structures_is_precise(tmp_path, nep_command):
    """A 5000-atom supercell against 125 copies of its 40-atom cell, at -158 eV per atom, the
    scale of absolute plane-wave energies. The combination vanishes for the reference energies
    and, up to the float precision of the per-atom energies, for the predicted ones. Summing in
    single precision would leave 1.2e-2 and 3.8e-2 eV in two runs, while the sums in double leave
    1.1e-3 to 1.9e-3 eV over three runs."""
    frames = read_frames(TRAINING_DIR / 'train.xyz')
    energy_per_atom = -158.123457
    small = with_energy(frames[0], 40 * energy_per_atom)
    big = with_energy(make_supercell(frames[0], 5), 5000 * energy_per_atom)
    other = with_energy(frames[1], 40 * energy_per_atom + 0.3)
    write_frames(tmp_path / 'train.xyz', [small, big, other], ['small', 'big', 'other'])
    write_frames(tmp_path / 'test.xyz', [small, big], ['small', 'big'])
    write_nep_in(tmp_path, {'lambda_d': '1'})
    (tmp_path / 'ediff.in').write_text('big - 125*small\n')
    result = run_nep(tmp_path, nep_command)
    assert result.returncode == 0, result.stdout + result.stderr

    columns, rows = read_loss_out(tmp_path)
    values = dict(zip(columns, rows[-1]))
    assert values['rmse_ediff_test'] < 5e-3


@pytest.mark.parametrize(
    'coefficient, is_accepted', [('1/3', True), ('0.3333333', False)], ids=['fraction', 'decimal']
)
def test_fraction_coefficients_balance_exactly(tmp_path, nep_command, coefficient, is_accepted):
    """A third written as a decimal leaves an offset of about 1e-7 per atom in the combination,
    which the balance check rejects, while the fraction 1/3 balances."""
    frames = setup_directory(tmp_path, None, {'lambda_d': '1'})
    write_frames(tmp_path / 'train.xyz', frames, ['S0', 'S1', 'S2', 'S3'])
    (tmp_path / 'test.xyz').unlink()
    terms = ' - '.join(f'{coefficient}*s{i}' for i in (1, 2, 3))
    (tmp_path / 'ediff.in').write_text(f's0 - {terms}\n')
    result = run_nep(tmp_path, nep_command)
    if is_accepted:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0
        assert 'the combination is not balanced in train.xyz' in result.stderr


def test_total_loss_matches_columns_with_two_batches(tmp_path, nep_command):
    """Each train column of loss.out belongs to the batch of the reported generation, so the total
    loss is their weighted sum. The two pairs lie in different batches and, through a weight of
    100, differ by about a factor of ten, so a train column pooled over both batches would not
    add up."""
    frames = setup_directory(tmp_path, None, {'lambda_d': '3', 'batch': '2'})
    names = ['S0', 'S1', 'S2', 'S3']
    write_frames(tmp_path / 'train.xyz', frames, names)
    write_frames(tmp_path / 'test.xyz', frames, names)
    (tmp_path / 'ediff.in').write_text('s0 - s1\ns2 - s3 100\n')
    result = run_nep(tmp_path, nep_command)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Number of batches = 2' in result.stdout

    columns, rows = read_loss_out(tmp_path)
    v = dict(zip(columns, rows[-1]))
    parts = (
        v['L1'] + v['L2'] + v['rmse_energy_train'] + v['rmse_force_train']
        + 0.1 * v['rmse_virial_train'] + 3 * v['rmse_ediff_train']
    )
    assert v['total'] == pytest.approx(parts, rel=1e-3, abs=1e-4)

    energies = read_total_energies(tmp_path / 'energy_test.out', [40] * 4)
    targets = [total_reference_energy(frame) for frame in frames]
    errors = [
        abs(energies[0] - energies[1] - targets[0] + targets[1]),
        10 * abs(energies[2] - energies[3] - targets[2] + targets[3]),
    ]
    # energy_test.out leaves up to 4e-3 eV per pair, which the weight of 100 scales tenfold
    tolerances = [5e-3, 5e-2]
    assert any(abs(v['rmse_ediff_train'] - e) < t for e, t in zip(errors, tolerances))
