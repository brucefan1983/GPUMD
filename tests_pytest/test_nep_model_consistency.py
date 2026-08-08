"""Consistency checks between nep.in and a nep.txt/nep.restart pair already in the working
directory.

A normal training run writes nep.txt, so a second invocation of `nep` in the same directory always
finds one, and a resume additionally finds the nep.restart written every output_interval
generations. If the model hyperparameters implied by nep.in differ from the ones recorded in those
files, the run used to continue on a differently shaped model without saying anything: reading fewer
nep.restart rows than the file holds and reporting plausible-looking losses. Any keyword with a
default can cause this, which is what these tests cover.

nep.txt counts as an input when there is a nep.restart to resume from, or when predicting, and a
mismatch is then an error. Without a nep.restart it is a stale output about to be overwritten, and a
mismatch is only a warning, so that editing nep.in and retraining in place keeps working.

Every mismatch is reported while nep.in is being read, before the training set is touched, so these
runs are short. The base models cover a header with a zbl line and one without (six header lines
instead of seven, the case that used to abort the header parse outright) as well as both qNEP charge
modes.
"""
import re
import shutil
import subprocess

import pytest

from conftest import TRAINING_DIR

pytestmark = pytest.mark.fast

# generation is kept tiny and output_interval is lowered to 1 so that nep.txt is written; nep.restart
# is written on the same interval, so train_once below deletes the one it produces and every test
# that wants a nep.restart writes its own
COMMON_KEYWORDS = {
    'type': '3 Ba Ti O',
    'cutoff': '6 4',
    'n_max': '8 6',
    'basis_size': '8 8',
    'l_max': '4 0 0',
    'neuron': '40',
    'generation': '2',
    'output_interval': '1',
}

BASE_MODELS = {
    'nep': {'zbl': '1.5'},
    'nep_without_zbl': {},
    'qnep_mode1': {'zbl': '1.5', 'charge_mode': '1'},
    'qnep_mode2': {'zbl': '1.5', 'charge_mode': '2'},
}

# each mutation of the base nep.in, with the text that must appear in the resulting report
MISMATCHES = {
    'basis_size omitted': ({'basis_size': None}, 'basis_size_radial'),
    'n_max omitted': ({'n_max': None}, 'n_max_radial'),
    'l_max 4-body flag changed': ({'l_max': '4 1 0'}, 'L_max_4body'),
    'neuron changed': ({'neuron': '30'}, 'neuron'),
    'angular cutoff changed': ({'cutoff': '6 3.5'}, 'cutoff (angular)'),
    'element order changed': ({'type': '3 Ba O Ti'}, 'type (elements and their order)'),
    'zbl removed': ({'zbl': None}, 'zbl'),
}


def keywords_for(base_model, **overrides):
    """The full nep.in keyword set for one base model, with overrides applied; an override value of
    None removes the keyword so that its default is used instead."""
    keywords = dict(COMMON_KEYWORDS)
    keywords.update(BASE_MODELS[base_model])
    keywords.update(overrides)
    return {key: value for key, value in keywords.items() if value is not None}


def write_nep_in(directory, keywords):
    text = ''.join(f'{key} {value}\n' for key, value in keywords.items())
    (directory / 'nep.in').write_text(text)


def run_nep(directory, nep_command, keywords=None):
    if keywords is not None:
        write_nep_in(directory, keywords)
    return subprocess.run(
        [nep_command], cwd=directory, capture_output=True, text=True, check=False
    )


def train_once(directory, nep_command, base_model):
    """Run a short training in an empty directory so that it leaves a nep.txt behind, and return the
    number of parameters that nep.in implies, which is the number of rows a matching nep.restart
    has. The nep.restart that the run itself writes is removed, so that a test only sees one if it
    wrote it with write_nep_restart below."""
    shutil.copy(TRAINING_DIR / 'train.xyz', directory / 'train.xyz')
    result = run_nep(directory, nep_command, keywords_for(base_model))
    assert result.returncode == 0, result.stdout + result.stderr
    assert (directory / 'nep.txt').exists(), result.stdout + result.stderr
    (directory / 'nep.restart').unlink(missing_ok=True)
    match = re.search(r'total number of parameters to be optimized = (\d+)', result.stdout)
    assert match is not None, result.stdout
    return int(match.group(1))


def write_nep_restart(directory, number_of_rows):
    text = ''.join('%15.7e %15.7e\n' % (0.1, 0.1) for _ in range(number_of_rows))
    (directory / 'nep.restart').write_text(text)


@pytest.mark.parametrize('base_model', list(BASE_MODELS))
def test_matching_nep_in_is_accepted(tmp_path, nep_command, base_model):
    """A nep.in that agrees with both nep.txt and nep.restart resumes without any complaint."""
    number_of_variables = train_once(tmp_path, nep_command, base_model)
    write_nep_restart(tmp_path, number_of_variables)

    result = run_nep(tmp_path, nep_command, keywords_for(base_model))
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'inconsistent' not in result.stdout


@pytest.mark.parametrize('base_model', list(BASE_MODELS))
def test_stale_nep_txt_without_restart_only_warns(tmp_path, nep_command, base_model):
    """Editing nep.in and retraining in place has to keep working, so a mismatch against a nep.txt
    that this run is about to overwrite is reported but not fatal."""
    train_once(tmp_path, nep_command, base_model)

    result = run_nep(tmp_path, nep_command, keywords_for(base_model, basis_size=None))
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'basis_size_radial' in result.stdout
    assert 'nep.in gives 6 (default), nep.txt gives 8' in result.stdout
    assert 'nep.txt will be overwritten' in result.stdout


@pytest.mark.parametrize(
    'overrides, expected_text', list(MISMATCHES.values()), ids=list(MISMATCHES)
)
def test_resume_with_mismatch_is_fatal(tmp_path, nep_command, overrides, expected_text):
    """With a nep.restart present the run is a genuine resume, where continuing on a differently
    shaped model corrupts the training, so every mismatch is an error naming the keyword."""
    number_of_variables = train_once(tmp_path, nep_command, 'nep')
    write_nep_restart(tmp_path, number_of_variables)

    result = run_nep(tmp_path, nep_command, keywords_for('nep', **overrides))
    assert result.returncode != 0
    assert expected_text in result.stdout
    assert 'remove nep.restart' in result.stdout
    assert 'nep.in is inconsistent with nep.txt.' in result.stderr


def test_zbl_added_to_a_model_without_it_is_fatal(tmp_path, nep_command):
    """The mirror image of the 'zbl removed' case above: a six-line header where nep.in asks for a
    seven-line one."""
    number_of_variables = train_once(tmp_path, nep_command, 'nep_without_zbl')
    write_nep_restart(tmp_path, number_of_variables)

    result = run_nep(tmp_path, nep_command, keywords_for('nep_without_zbl', zbl='1.5'))
    assert result.returncode != 0
    assert 'nep.in has ZBL enabled, nep.txt has it disabled' in result.stdout


@pytest.mark.parametrize(
    'base_model, charge_mode, expected_text',
    [
        ('qnep_mode2', '1', 'nep.in gives 1 (input), nep.txt gives 2'),
        ('qnep_mode1', None, 'nep.in gives 0 (default), nep.txt gives 1'),
        ('nep', '2', 'nep.in gives 2 (input), nep.txt gives 0'),
    ],
    ids=['mode 2 model read as mode 1', 'charge model read as plain', 'plain model read as qNEP'],
)
def test_charge_mode_mismatch_is_fatal(
    tmp_path, nep_command, base_model, charge_mode, expected_text
):
    """Swapping charge_mode 1 for 2, or a charge model for a plain one, changes the number of
    parameters and must not be silently accepted."""
    number_of_variables = train_once(tmp_path, nep_command, base_model)
    write_nep_restart(tmp_path, number_of_variables)

    result = run_nep(tmp_path, nep_command, keywords_for(base_model, charge_mode=charge_mode))
    assert result.returncode != 0
    assert 'charge_mode: ' + expected_text in result.stdout


def test_prediction_with_mismatched_nep_txt_is_fatal(tmp_path, nep_command):
    """In prediction mode nep.txt is an input rather than an output, so a mismatch is an error even
    though there is no nep.restart."""
    train_once(tmp_path, nep_command, 'nep')

    result = run_nep(
        tmp_path, nep_command, keywords_for('nep', basis_size=None, prediction='1')
    )
    assert result.returncode != 0
    assert 'basis_size_radial' in result.stdout
    assert 'nep.in is inconsistent with nep.txt.' in result.stderr


def test_prediction_ignores_a_stale_restart(tmp_path, nep_command):
    """nep.restart is read only when resuming a training run, so a leftover one of the wrong length
    must not stop a prediction run that agrees with nep.txt."""
    number_of_variables = train_once(tmp_path, nep_command, 'nep')
    write_nep_restart(tmp_path, number_of_variables + 96)

    result = run_nep(tmp_path, nep_command, keywords_for('nep', prediction='1'))
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'energy_train.out').exists()


@pytest.mark.parametrize('row_offset', [-96, -1, 96], ids=['96 short', '1 short', '96 long'])
def test_restart_row_count_mismatch_is_fatal(tmp_path, nep_command, row_offset):
    """nep.restart carries no header, so its row count is the only thing that can be checked. Both
    counts are reported because their difference is what identifies the keyword at fault."""
    number_of_variables = train_once(tmp_path, nep_command, 'nep')
    write_nep_restart(tmp_path, number_of_variables + row_offset)

    result = run_nep(tmp_path, nep_command, keywords_for('nep'))
    assert result.returncode != 0
    assert f'nep.restart holds {number_of_variables + row_offset} rows' in result.stdout
    assert f'nep.in implies {number_of_variables} parameters' in result.stdout
    assert f'difference of {abs(row_offset)} rows' in result.stdout
    assert 'nep.restart does not match the model implied by nep.in.' in result.stderr


def test_restart_row_count_is_checked_without_a_nep_txt(tmp_path, nep_command):
    """The row count is checked on its own merits, so a nep.restart left behind after nep.txt was
    deleted is still rejected."""
    number_of_variables = train_once(tmp_path, nep_command, 'nep')
    write_nep_restart(tmp_path, number_of_variables + 96)
    (tmp_path / 'nep.txt').unlink()

    result = run_nep(tmp_path, nep_command, keywords_for('nep'))
    assert result.returncode != 0
    assert 'nep.restart does not match the model implied by nep.in.' in result.stderr
