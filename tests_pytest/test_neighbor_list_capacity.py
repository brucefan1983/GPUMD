"""Tests of the bound check of the neighbor lists.

An atom with more neighbors than a list holds stops gpumd with a message that names the list, its
capacity and the atom.
Each case lowers the capacity of one list through the cutoff field of the model file, or
compresses a cell until the fixed capacity of the small-box lists is exceeded.
"""
import re
import subprocess

import pytest
from ase.io import read
from calorine.gpumd import write_xyz

from conftest import MODELS_DIR, STRUCTURES_DIR

pytestmark = pytest.mark.fast


def _write_model(source, destination, cutoff):
    """Copies a model file with the cutoff field replaced by cutoff."""
    text = (MODELS_DIR / source).read_text()
    text, count = re.subn(r'^cutoff .*$', f'cutoff {cutoff}', text, count=1, flags=re.MULTILINE)
    assert count == 1
    destination.write_text(text)
    return destination


def _run(directory, gpumd_command, atoms, model):
    write_xyz(str(directory / 'model.xyz'), atoms)
    keywords = [f'potential {model}', 'velocity 300 seed 1', 'ensemble nve', 'time_step 1', 'run 1']
    (directory / 'run.in').write_text('\n'.join(keywords) + '\n')
    return subprocess.run([gpumd_command], cwd=directory, capture_output=True, text=True)


def _diamond(repeat):
    atoms = read(STRUCTURES_DIR / 'C-nat16-rattled.xyz')
    atoms.calc = None
    return atoms.repeat(repeat)


def _assert_overflow(result, kind, capacity):
    assert result.returncode != 0
    assert 'CUDA Error' not in result.stderr
    message = rf'The {kind} neighbor list holds {capacity} neighbors per atom, but atom \d+ has'
    assert re.search(message, result.stderr), result.stderr


def test_candidate_list_of_a_large_box(tmp_path, gpumd_command):
    # max_radial 100 gives MN_radial 125, and the candidate list at rc + 1 Angstrom of skin holds
    # int(125 * 8^3 / 7^3) = 186, against about 380 neighbors of diamond within 8 Angstrom.
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
