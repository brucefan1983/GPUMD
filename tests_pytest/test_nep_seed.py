"""Tests of the seed keyword in nep.in, which seeds both random generators of SNES.

A host generator draws the initial mu, and nothing when a nep.restart is present. The population of
every generation is drawn on the GPU. Both take the seed, so two runs of one seeded input draw the
same random numbers.
"""
import shutil
import subprocess

import numpy as np
import pytest
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

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


def train(directory, nep_command, seed, restart_directory=None, keywords=KEYWORDS):
    """Train in directory with the given seed, starting from the nep.restart and nep.txt of
    restart_directory if one is given."""
    directory.mkdir()
    shutil.copy(TRAINING_DIR / 'train.xyz', directory / 'train.xyz')
    if restart_directory is not None:
        for name in ['nep.restart', 'nep.txt']:
            shutil.copy(restart_directory / name, directory / name)
    result = run_nep(directory, nep_command, {**keywords, 'seed': str(seed)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert f'(input)   random seed = {seed}.' in result.stdout
    return directory


def test_same_seed_gives_the_same_model(tmp_path, nep_command):
    first = train(tmp_path / 'first', nep_command, seed=1)
    second = train(tmp_path / 'second', nep_command, seed=1)
    for name in ['loss.out', 'nep.txt']:
        assert (first / name).read_text() == (second / name).read_text(), name


def write_supercells(path):
    """Write 2x2x2 supercells of the first five training structures, 320 atoms each, so that
    every atom has enough neighbors for the summation order of the forces to matter. Each atom
    carries a diagonal Born effective charge by species."""
    born_charges = {'Ba': 2.7, 'Ti': 7.0, 'O': -3.2}
    supercells = []
    for structure in read(TRAINING_DIR / 'train.xyz', index=':5'):
        supercell = structure.repeat(2)
        forces = np.tile(structure.arrays['force'], (8, 1))
        energy = 8 * structure.get_potential_energy()
        supercell.calc = SinglePointCalculator(supercell, energy=energy, forces=forces)
        diagonal = [born_charges[symbol] for symbol in supercell.get_chemical_symbols()]
        supercell.arrays['bec'] = np.outer(diagonal, np.eye(3).ravel())
        supercells.append(supercell)
    write(path, supercells, format='extxyz')


MODELS = {
    'nep': {},
    'vdw': {'vdw': '1'},
    'charge_mode1': {'charge_mode': '1'},
    'charge_mode2': {'charge_mode': '2'},
    'charge_vdw': {'charge_vdw': '1'},
}


@pytest.mark.parametrize('model', list(MODELS))
@pytest.mark.parametrize('prediction', ['0', '1'], ids=['training', 'prediction'])
def test_same_seed_gives_the_same_forces(tmp_path, nep_command, model, prediction):
    """Training evaluates the radial and angular forces with the kernels that nep_compile builds at
    run time, and prediction with the generic ones. The ZBL and charge real-space forces use the
    generic kernels in both. The charge models also sum Born effective charges."""
    keywords = {**KEYWORDS, 'zbl': '2', **MODELS[model]}
    initial = train(tmp_path / 'initial', nep_command, seed=1, keywords=keywords)
    suffix = 'test' if prediction == '0' else 'train'
    names = [f'force_{suffix}.out'] + ([f'bec_{suffix}.out'] if model.startswith('charge') else [])
    outputs = []
    for run in ['first', 'second', 'third']:
        directory = tmp_path / run
        directory.mkdir()
        shutil.copy(initial / 'nep.txt', directory / 'nep.txt')
        write_supercells(directory / 'train.xyz')
        write_supercells(directory / 'test.xyz')
        result = run_nep(
            directory, nep_command, {**keywords, 'prediction': prediction, 'seed': '1'})
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append({name: (directory / name).read_text() for name in names})
    differing = [
        name for name in names if not outputs[0][name] == outputs[1][name] == outputs[2][name]]
    assert differing == []


@pytest.mark.parametrize('separation', [0.0, 0.002], ids=['coincident', 'overlapping'])
def test_seed_keeps_nan_and_huge_forces(tmp_path, nep_command, separation):
    """Two coincident atoms give NaN forces, and two atoms 0.002 Å apart give a ZBL force of
    about 4e9 eV/Å, beyond the range of the fixed-point sum. A seeded prediction writes the same
    forces as an unseeded one."""
    keywords = {**KEYWORDS, 'zbl': '2'}
    initial = train(tmp_path / 'initial', nep_command, seed=1, keywords=keywords)
    structure = read(TRAINING_DIR / 'train.xyz', index=0)
    energy = structure.get_potential_energy()
    structure.calc = None
    structure.positions[1] = structure.positions[0] + [separation, 0, 0]
    structure.calc = SinglePointCalculator(
        structure, energy=energy, forces=structure.arrays['force'])
    forces = {}
    for label, extra in [('float', {}), ('fixed', {'seed': '1'})]:
        directory = tmp_path / label
        directory.mkdir()
        shutil.copy(initial / 'nep.txt', directory / 'nep.txt')
        write(directory / 'train.xyz', [structure], format='extxyz')
        result = run_nep(directory, nep_command, {**keywords, 'prediction': '1', **extra})
        assert result.returncode == 0, result.stdout + result.stderr
        forces[label] = np.loadtxt(directory / 'force_train.out')[:, :3]
    assert np.allclose(forces['fixed'], forces['float'], rtol=1e-5, atol=1e-4, equal_nan=True)


def read_parameters(directory):
    lines = (directory / 'nep.txt').read_text().splitlines()
    return np.array([float(line) for line in lines if len(line.split()) == 1])


def test_seed_reaches_the_initial_mu(tmp_path, nep_command):
    """After one generation with the smallest sigma0 the parameters lie within a few sigma0 of the
    initial mu. Two seeds then give parameters far apart only if the initial mu takes the seed."""
    keywords = {**KEYWORDS, 'generation': '1', 'sigma0': '0.01'}
    first = train(tmp_path / 'first', nep_command, seed=1, keywords=keywords)
    second = train(tmp_path / 'second', nep_command, seed=2, keywords=keywords)
    # the largest difference on this input is 2.8, or 0.06 with the initial mu from a fixed seed
    assert np.max(np.abs(read_parameters(first) - read_parameters(second))) > 0.5


def test_seed_reaches_the_population_draws(tmp_path, nep_command):
    """Starting from a nep.restart, mu is read from the file and the host generator draws
    nothing, so the models differ only if the population draws take the seed."""
    initial = train(tmp_path / 'initial', nep_command, seed=0)
    first = train(tmp_path / 'first', nep_command, seed=1, restart_directory=initial)
    second = train(tmp_path / 'second', nep_command, seed=2, restart_directory=initial)
    assert (first / 'nep.txt').read_text() != (second / 'nep.txt').read_text()


def test_default_seed_is_reported_as_not_set(tmp_path, nep_command):
    shutil.copy(TRAINING_DIR / 'train.xyz', tmp_path / 'train.xyz')
    result = run_nep(tmp_path, nep_command, KEYWORDS)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '(default) random seed not set.' in result.stdout


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
