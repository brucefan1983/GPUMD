"""Checks of the energy-difference loss of nep, activated by lambda_d > 0 and ediff.in.

The four structures of the training fixture have 40 atoms each. The first two form train.xyz and
the last two test.xyz, so that every pair has equal sizes and the per-atom energy shift that nep
applies to the elite cancels in each difference. One generation per output_interval yields one
row of loss.out per run.
"""
import math
import subprocess

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


def setup_directory(directory, ediff_lines, overrides=None):
    """Write train.xyz with the structures S0 and S1, test.xyz with S2 and S3, nep.in, and ediff.in
    unless ediff_lines is None. Returns the frames."""
    frames = read_frames(TRAINING_DIR / 'train.xyz')
    assert len(frames) == 4
    write_frames(directory / 'train.xyz', frames[:2], ['S0', 'S1'])
    write_frames(directory / 'test.xyz', frames[2:], ['S2', 'S3'])
    keywords = dict(KEYWORDS)
    keywords.update(overrides or {})
    text = ''.join(f'{key} {value}\n' for key, value in keywords.items() if value is not None)
    (directory / 'nep.in').write_text(text)
    if ediff_lines is not None:
        (directory / 'ediff.in').write_text('# name_a name_b [ref_eV] [weight]\n' + ediff_lines)
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


@pytest.mark.parametrize('overrides', [{}, {'charge_mode': '1', 'zbl': '1.5'}], ids=['nep', 'qnep'])
def test_ediff_columns_are_appended(tmp_path, nep_command, overrides):
    """The two ediff columns follow all other columns, and every row has one value per column."""
    setup_directory(tmp_path, 's0 s1\ns2 s3\n', {'lambda_d': '1', **overrides})
    result = run_nep(tmp_path, nep_command)
    assert result.returncode == 0, result.stdout + result.stderr

    columns, rows = read_loss_out(tmp_path)
    assert columns[-2:] == ['rmse_ediff_train', 'rmse_ediff_test']
    assert len(rows) == 1
    assert all(len(row) == len(columns) for row in rows)


def test_ediff_rmse_matches_predicted_energies(tmp_path, nep_command):
    """The test column equals the error of the predicted energy difference in energy_test.out,
    which the report writes from the same evaluation. The report writes no energy_train.out, so
    the train column is only checked to be finite."""
    frames = setup_directory(tmp_path, 'S0 S1\nS2 S3\n', {'lambda_d': '1'})
    result = run_nep(tmp_path, nep_command)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Number of energy difference pairs = 1 in train.xyz and 1 in test.xyz.' in result.stdout

    columns, rows = read_loss_out(tmp_path)
    values = dict(zip(columns, rows[-1]))
    assert math.isfinite(values['rmse_ediff_train'])

    energies_per_atom = [
        float(line.split()[0]) for line in (tmp_path / 'energy_test.out').read_text().splitlines()
    ]
    num_atoms = 40
    predicted = num_atoms * (energies_per_atom[0] - energies_per_atom[1])
    reference = total_reference_energy(frames[2]) - total_reference_energy(frames[3])
    # energy_test.out carries six significant digits, about 1e-4 eV per atom
    assert values['rmse_ediff_test'] == pytest.approx(abs(predicted - reference), abs=5e-3)


def test_layout_without_lambda_d_is_unchanged(tmp_path, nep_command):
    """Without lambda_d an ediff.in is ignored, and loss.out has the columns of a run without it."""
    setup_directory(tmp_path, 's0 s1\n')
    result = run_nep(tmp_path, nep_command)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ediff.in is ignored because lambda_d = 0.' in result.stdout

    columns, rows = read_loss_out(tmp_path)
    assert columns == MASTER_NEP_COLUMNS
    assert all(len(row) == len(MASTER_NEP_COLUMNS) for row in rows)


@pytest.mark.parametrize(
    'ediff_lines, overrides, expected_text',
    [
        (None, {'lambda_d': '1'}, 'lambda_d > 0 requires the file ediff.in.'),
        ('s0 unknown\ns2 s3\n', {'lambda_d': '1'}, 'No pair in ediff.in has both structures'),
        ('s0 s1\n', {'lambda_d': '1', 'model_type': '1'}, 'lambda_d is only supported'),
    ],
    ids=['missing ediff.in', 'no train pair', 'dipole model'],
)
def test_input_errors(tmp_path, nep_command, ediff_lines, overrides, expected_text):
    setup_directory(tmp_path, ediff_lines, overrides)
    result = run_nep(tmp_path, nep_command)
    assert result.returncode != 0
    assert expected_text in result.stderr


def test_test_pair_enters_only_the_test_column(tmp_path, nep_command):
    """A reference of 1000 eV for the pair of test.xyz makes its error dominate any RMSE it enters,
    so that it has to raise the test column and leave the train column at the scale of the pair of
    train.xyz. A pair split across the two files is skipped."""
    setup_directory(tmp_path, 's0 s1\ns2 s3 1000\ns0 s2\n', {'lambda_d': '1'})
    result = run_nep(tmp_path, nep_command)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Warning: skipping pair s0 s2 of ediff.in' in result.stdout

    columns, rows = read_loss_out(tmp_path)
    values = dict(zip(columns, rows[-1]))
    assert values['rmse_ediff_test'] > 900
    assert values['rmse_ediff_train'] < 100
