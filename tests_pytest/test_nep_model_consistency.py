"""Consistency checks for nep.in and existing nep.txt/nep.restart files.

Prediction and import_q_scaler validate the metadata in nep.txt. Ordinary training, including
resume, ignores nep.txt. The new restart format validates energy-reference and parameter counts;
it does not store model hyperparameters. Tests cover NEP with and without ZBL, both qNEP charge
modes, and tensor models.
"""
import re
import shutil
import subprocess

import pytest

from conftest import MODELS_DIR, STRUCTURES_DIR, TRAINING_DIR

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
    text = f'nep_restart\nenergy_reference 3 0 0 0\nparameters {number_of_rows}\n'
    text += ''.join('%15.7e %15.7e\n' % (0.1, 0.1) for _ in range(number_of_rows))
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
def test_stale_nep_txt_without_restart_is_ignored(tmp_path, nep_command, base_model):
    """Fresh training ignores the nep.txt that it is about to overwrite."""
    train_once(tmp_path, nep_command, base_model)

    result = run_nep(tmp_path, nep_command, keywords_for(base_model, basis_size=None))
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'inconsistent' not in result.stdout
    assert 'nep.txt will be overwritten' not in result.stdout


@pytest.mark.parametrize(
    'overrides, expected_text', list(MISMATCHES.values()), ids=list(MISMATCHES)
)
def test_import_q_scaler_with_mismatch_is_fatal(tmp_path, nep_command, overrides, expected_text):
    """Importing scalers makes nep.txt an input, so metadata mismatches must be reported."""
    number_of_variables = train_once(tmp_path, nep_command, 'nep')
    write_nep_restart(tmp_path, number_of_variables)

    result = run_nep(tmp_path, nep_command, keywords_for('nep', import_q_scaler='1', **overrides))
    assert result.returncode != 0
    assert expected_text in result.stdout
    assert 'nep.in is inconsistent with nep.txt.' in result.stderr


def test_zbl_added_to_a_model_without_it_is_fatal(tmp_path, nep_command):
    """Importing scalers also rejects adding ZBL to a model that does not have it."""
    number_of_variables = train_once(tmp_path, nep_command, 'nep_without_zbl')
    write_nep_restart(tmp_path, number_of_variables)

    result = run_nep(
        tmp_path, nep_command, keywords_for('nep_without_zbl', zbl='1.5', import_q_scaler='1')
    )
    assert result.returncode != 0
    assert 'nep.in has ZBL enabled, nep.txt has it disabled' in result.stdout


@pytest.mark.parametrize(
    'base_model, charge_mode, expected_text',
    [
        ('qnep_mode2', '1', 'model: nep.in implies qnep1, nep.txt holds qnep2.'),
        ('qnep_mode1', None, 'model: nep.in implies nep, nep.txt holds qnep1.'),
        ('nep', '2', 'model: nep.in implies qnep2, nep.txt holds nep.'),
    ],
    ids=['mode 2 model read as mode 1', 'charge model read as plain', 'plain model read as qNEP'],
)
def test_charge_mode_mismatch_is_fatal(
    tmp_path, nep_command, base_model, charge_mode, expected_text
):
    """Importing scalers requires the same model type, including the charge mode."""
    number_of_variables = train_once(tmp_path, nep_command, base_model)
    write_nep_restart(tmp_path, number_of_variables)

    result = run_nep(
        tmp_path, nep_command,
        keywords_for(base_model, charge_mode=charge_mode, import_q_scaler='1')
    )
    assert result.returncode != 0
    assert expected_text in result.stdout


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
    """The parameter count declared in nep.restart must match the model implied by nep.in."""
    number_of_variables = train_once(tmp_path, nep_command, 'nep')
    write_nep_restart(tmp_path, number_of_variables + row_offset)

    result = run_nep(tmp_path, nep_command, keywords_for('nep'))
    assert result.returncode != 0
    assert 'nep.restart: invalid parameter count.' in result.stderr


def test_restart_row_count_is_checked_without_a_nep_txt(tmp_path, nep_command):
    """The restart parameter count is checked even when nep.txt has been deleted."""
    number_of_variables = train_once(tmp_path, nep_command, 'nep')
    write_nep_restart(tmp_path, number_of_variables + 96)
    (tmp_path / 'nep.txt').unlink()

    result = run_nep(tmp_path, nep_command, keywords_for('nep'))
    assert result.returncode != 0
    assert 'nep.restart: invalid parameter count.' in result.stderr


# the tensor models, each with the nep.in keywords of its header, a structure, and the target
# that structure carries
TENSOR_MODELS = {
    'dipole': (
        'tnep-water-polarization.txt',
        {'type': '2 O H', 'model_type': '1', 'cutoff': '6 4', 'n_max': '6 6',
         'basis_size': '10 10', 'l_max': '4 2 1', 'neuron': '10', 'prediction': '1'},
        'water-nat63-from-md.xyz',
        'dipole="0 0 0"',
    ),
    'polarizability': (
        'tnep-BaZrO3-susceptibility.txt',
        {'type': '3 Ba Zr O', 'model_type': '2', 'cutoff': '6 4', 'n_max': '4 4',
         'basis_size': '10 10', 'l_max': '4 2 0', 'neuron': '20', 'prediction': '1'},
        'BaZrO3-nat40-rattled.xyz',
        'pol="0 0 0 0 0 0 0 0 0"',
    ),
}
DIPOLE_KEYWORDS = TENSOR_MODELS['dipole'][1]


def write_tensor_prediction_inputs(directory, name):
    """The tensor model as nep.txt and one structure with its target as train.xyz."""
    model, _, structure, target = TENSOR_MODELS[name]
    shutil.copy(MODELS_DIR / 'new_format' / model, directory / 'nep.txt')
    lines = (STRUCTURES_DIR / structure).read_text().splitlines()
    lines[1] = lines[1] + ' ' + target
    (directory / 'train.xyz').write_text('\n'.join(lines) + '\n')


@pytest.mark.parametrize('name', list(TENSOR_MODELS))
def test_prediction_of_a_tensor_model_with_matching_nep_in_runs(tmp_path, nep_command, name):
    write_tensor_prediction_inputs(tmp_path, name)

    result = run_nep(tmp_path, nep_command, TENSOR_MODELS[name][1])
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'inconsistent' not in result.stdout
    assert (tmp_path / f'{name}_train.out').exists()


def test_prediction_of_a_tensor_model_with_mismatched_nep_txt_is_fatal(tmp_path, nep_command):
    """The check of nep.in against nep.txt applies to every model type, so a dipole model read
    with another basis size stops with an input error instead of evaluating a misread model."""
    write_tensor_prediction_inputs(tmp_path, 'dipole')

    result = run_nep(tmp_path, nep_command, dict(DIPOLE_KEYWORDS, basis_size='10 8'))
    assert result.returncode != 0
    assert 'basis_size_angular' in result.stdout
    assert 'nep.in is inconsistent with nep.txt.' in result.stderr


def train_tensor_model(directory, nep_command, name):
    """Train the tensor model briefly, remove the nep.txt the run writes so that only its
    nep.restart remains, and return the nep.in keywords of the run."""
    write_tensor_prediction_inputs(directory, name)
    keywords = dict(TENSOR_MODELS[name][1], prediction='0', generation='2', output_interval='1')
    result = run_nep(directory, nep_command, keywords)
    assert result.returncode == 0, result.stdout + result.stderr
    (directory / 'nep.txt').unlink()
    return keywords


@pytest.mark.parametrize('name', list(TENSOR_MODELS))
def test_resume_of_a_tensor_model_with_matching_nep_restart_runs(tmp_path, nep_command, name):
    keywords = train_tensor_model(tmp_path, nep_command, name)

    result = run_nep(tmp_path, nep_command, keywords)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('name', list(TENSOR_MODELS))
def test_resume_of_a_tensor_model_with_mismatched_nep_restart_is_fatal(
    tmp_path, nep_command, name
):
    """The parameter count of nep.restart is checked for every model type, so a tensor model resumed
    with another basis size stops instead of reading the leading rows of a larger model."""
    keywords = train_tensor_model(tmp_path, nep_command, name)

    result = run_nep(tmp_path, nep_command, dict(keywords, basis_size='10 8'))
    assert result.returncode != 0
    assert 'nep.restart: invalid parameter count.' in result.stderr
