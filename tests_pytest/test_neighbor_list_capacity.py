"""Tests of the bound check of the neighbor lists.

An atom with more neighbors than a list holds stops gpumd with a message that names the list, its
capacity and the atom.
Each case lowers the capacity of one list through the cutoff field of the model file, or
compresses a cell until the fixed capacity of the small-box lists is exceeded.
gpumd runs under compute-sanitizer memcheck when it is on the PATH, which detects a write past a
list that the check reports only afterwards.
"""
import re
import shutil
import subprocess

import pytest
from ase.io import read
from calorine.gpumd import write_xyz

from conftest import MODELS_DIR, STRUCTURES_DIR

pytestmark = pytest.mark.fast

SANITIZER = shutil.which('compute-sanitizer')


def _write_model(source, destination, cutoff):
    """Copies a model file with the cutoff field replaced by cutoff."""
    text = (MODELS_DIR / source).read_text()
    text, count = re.subn(r'^cutoff .*$', f'cutoff {cutoff}', text, count=1, flags=re.MULTILINE)
    assert count == 1
    destination.write_text(text)
    return destination


def _run(directory, gpumd_command, atoms, model, keywords=()):
    write_xyz(str(directory / 'model.xyz'), atoms)
    run_in = [f'potential {model}', 'velocity 300 seed 1', 'ensemble nve', 'time_step 1']
    run_in += list(keywords) + ['run 1']
    (directory / 'run.in').write_text('\n'.join(run_in) + '\n')
    command = [gpumd_command]
    if SANITIZER is not None:
        command = [SANITIZER, '--tool', 'memcheck'] + command
    return subprocess.run(command, cwd=directory, capture_output=True, text=True)


def _diamond(repeat):
    atoms = read(STRUCTURES_DIR / 'C-nat16-rattled.xyz')
    atoms.calc = None
    return atoms.repeat(repeat)


def _assert_stop(result, message):
    assert result.returncode != 0
    assert 'CUDA Error' not in result.stderr
    assert re.search(message, result.stderr), result.stderr
    if SANITIZER is not None:
        assert 'ERROR SUMMARY: 0 errors' in result.stdout, result.stdout


def _assert_overflow(result, kind, capacity):
    message = rf'The {kind} neighbor list holds {capacity} neighbors per atom, but atom \d+ has'
    _assert_stop(result, message)


def test_candidate_list_of_a_large_box(tmp_path, gpumd_command):
    # max_radial 100 gives MN_radial 125, and the candidate list at rc + 1 Angstrom of skin holds
    # int(125 * 8^3 / 7^3) = 186, against about 370 neighbors of diamond within 8 Angstrom.
    model = _write_model('nep_C.txt', tmp_path / 'nep.txt', '7 4 100 71')
    result = _run(tmp_path, gpumd_command, _diamond(5), model)
    _assert_overflow(result, 'candidate', 186)


def test_angular_list_of_a_large_box(tmp_path, gpumd_command):
    # max_angular 10 gives MN_angular 13, against 46 neighbors of diamond within 4 Angstrom.
    model = _write_model('nep_C.txt', tmp_path / 'nep.txt', '7 4 358 10')
    result = _run(tmp_path, gpumd_command, _diamond(5), model)
    _assert_overflow(result, 'angular', 13)


def test_radial_list_of_a_small_box(tmp_path, gpumd_command):
    # Compressed to 0.4 of its lattice parameter, the 16-atom cell has about 4000 images within the
    # radial cutoff of 7 Angstrom, against the 2000 the small-box lists hold.
    atoms = _diamond(1)
    atoms.set_cell(atoms.cell * 0.4, scale_atoms=True)
    model = MODELS_DIR / 'nep_C.txt'
    result = _run(tmp_path, gpumd_command, atoms, model)
    _assert_overflow(result, 'radial', 2000)


def test_angular_list_of_a_qnep_model(tmp_path, gpumd_command):
    # max_angular 4 gives MN_angular 5, against about 27 neighbors in water within 4 Angstrom.
    model = _write_model('qnep_mode1_water.txt', tmp_path / 'nep.txt', '4 4 39 4')
    atoms = read(STRUCTURES_DIR / 'water-nat63-from-md.xyz').repeat((2, 2, 3))
    result = _run(tmp_path, gpumd_command, atoms, model)
    _assert_overflow(result, 'angular', 5)


def test_angular_list_of_a_dipole_model(tmp_path, gpumd_command):
    # max_angular 4 gives MN_angular 5 for the dipole model of dump_dipole, against about 27
    # neighbors in water within 4 Angstrom. The lists of the driving potential hold them all.
    dipole_model = _write_model(
        'tnep-water-polarization.txt', tmp_path / 'dipole.txt', '6 4 113 4')
    atoms = read(STRUCTURES_DIR / 'water-nat63-from-md.xyz').repeat((3, 2, 3))
    result = _run(
        tmp_path, gpumd_command, atoms, MODELS_DIR / 'nep_water.txt',
        [f'dump_dipole 1 {dipole_model}'])
    _assert_overflow(result, 'angular', 5)


def test_angular_list_of_an_extrapolation_model(tmp_path, gpumd_command):
    # max_angular 4 gives MN_angular 5 for the model of compute_extrapolation, against about 27
    # neighbors in water within 4 Angstrom. The lists of the driving potential hold them all.
    model = _write_model('nep_water.txt', tmp_path / 'extrapolation.txt', '6 4 251 4')
    # An identity active set inverse of the size of the model, one block per species.
    size = 960
    identity = '\n'.join(' '.join('1' if i == j else '0' for j in range(size)) for i in range(size))
    (tmp_path / 'active_set.asi').write_text(
        ''.join(f'{species} {size} {size}\n{identity}\n' for species in ('O', 'H')))
    atoms = read(STRUCTURES_DIR / 'water-nat63-from-md.xyz').repeat((3, 2, 3))
    result = _run(
        tmp_path, gpumd_command, atoms, MODELS_DIR / 'nep_water.txt',
        [f'compute_extrapolation nep_file {model} asi_file active_set.asi check_interval 1'])
    _assert_overflow(result, 'angular', 5)


@pytest.mark.parametrize('mc', ['mc canonical 1 2 300 300', 'mc sgc 1 2 300 300 2 Ba 0 Ti 0'])
def test_trial_list_of_monte_carlo(tmp_path, gpumd_command, mc):
    # Compressed to 0.4 of its lattice parameter, BaTiO3 has 1080 to 1126 atoms within the radial
    # cutoff of 6 Angstrom of every atom, against the 1000 atoms an MC trial holds.
    # The 16 Angstrom cell takes the small-box path of the potential, whose lists hold 2000.
    atoms = read(STRUCTURES_DIR / 'BaTiO3-nat40-rattled.xyz')
    atoms.calc = None
    atoms.set_cell(atoms.cell * 0.4, scale_atoms=True)
    model = MODELS_DIR / 'nep_BaTiO3.txt'
    result = _run(tmp_path, gpumd_command, atoms.repeat(5), model, [mc])
    message = (
        r'A Monte Carlo trial holds 1000 atoms within the radial cutoff of its trial atoms, '
        r'but the trial of atoms? \d+( and \d+)? has \d+')
    _assert_stop(result, message)
