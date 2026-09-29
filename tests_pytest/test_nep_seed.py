"""Tests of the seed keyword in nep.in, which seeds both random generators of SNES.

A host generator draws the initial mu, and nothing when a nep.restart is present. The population of
every generation is drawn on the GPU. Both take the seed, so two runs of one seeded input write the
same model.
"""
import shutil
import subprocess

import pytest

from conftest import TRAINING_DIR

pytestmark = pytest.mark.fast

KEYWORDS = {
    'type': '3 Ba Ti O',
    'cutoff': '6 4',
    'n_max': '8 6',
    'basis_size': '8 8',
    'l_max': '4 0 0',
    'neuron': '40',
    'generation': '2',
    'output_interval': '1',
}


def run_nep(directory, nep_command, keywords):
    text = ''.join(f'{key} {value}\n' for key, value in keywords.items())
    (directory / 'nep.in').write_text(text)
    return subprocess.run(
        [nep_command], cwd=directory, capture_output=True, text=True, check=False
    )


def train(directory, nep_command, seed, restart_directory=None):
    """Train in directory with the given seed, starting from the nep.restart and nep.txt of
    restart_directory if one is given."""
    directory.mkdir()
    shutil.copy(TRAINING_DIR / 'train.xyz', directory / 'train.xyz')
    if restart_directory is not None:
        for name in ['nep.restart', 'nep.txt']:
            shutil.copy(restart_directory / name, directory / name)
    result = run_nep(directory, nep_command, {**KEYWORDS, 'seed': str(seed)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert f'(input)   random seed = {seed}.' in result.stdout
    return directory


def test_same_seed_gives_the_same_model(tmp_path, nep_command):
    first = train(tmp_path / 'first', nep_command, seed=1)
    second = train(tmp_path / 'second', nep_command, seed=1)
    for name in ['loss.out', 'nep.txt']:
        assert (first / name).read_text() == (second / name).read_text(), name


def test_seed_reaches_the_population_draws(tmp_path, nep_command):
    """Starting from a nep.restart, mu is read from the file and the host generator draws
    nothing, so the models differ only if the population draws take the seed."""
    initial = train(tmp_path / 'initial', nep_command, seed=0)
    first = train(tmp_path / 'first', nep_command, seed=1, restart_directory=initial)
    second = train(tmp_path / 'second', nep_command, seed=2, restart_directory=initial)
    assert (first / 'nep.txt').read_text() != (second / 'nep.txt').read_text()


@pytest.mark.parametrize(
    'value, expected_text',
    [('-1', 'seed should be >= 0.'), ('', 'The seed keyword must be followed by a parameter.')],
    ids=['negative', 'missing'],
)
def test_invalid_seed_is_rejected(tmp_path, nep_command, value, expected_text):
    shutil.copy(TRAINING_DIR / 'train.xyz', tmp_path / 'train.xyz')
    result = run_nep(tmp_path, nep_command, {**KEYWORDS, 'seed': value})
    assert result.returncode != 0
    assert 'Input Error' in result.stderr
    assert expected_text in result.stderr
