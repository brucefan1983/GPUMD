"""Exercise the XRD interface and check single-bead sampling."""

import os
import subprocess

import numpy as np
import pytest
from ase.build import bulk
from ase.io import read
from calorine.gpumd import write_xyz

from conftest import MODELS_DIR, make_bulk_C, make_bulk_water


UNIT_CARBON = 'C 0 0 0 0 0 0 0 0 1\n'


@pytest.mark.fast
def test_compute_xrd(tmp_path, structure, structure_name, model_path, gpumd_command):
    if structure_name == 'bulk_C':
        structure = bulk('C', 'diamond', a=3.567, cubic=True).repeat(2)
    symbols = structure.get_chemical_symbols()
    factors = ''.join(f'{element} 0 0 0 0 0 0 0 0 1 # unit scatterer\n'
                      for element in sorted(set(symbols)))
    (tmp_path / 'factors.txt').write_text('# element and nine coefficients\n\n' + factors)
    write_xyz(str(tmp_path / 'model.xyz'), structure)
    (tmp_path / 'run.in').write_text(
        f'potential {model_path}\n'
        'velocity 300 seed 7\n'
        'ensemble nve\n'
        'time_step 0\n'
        'compute_xrd 1 1.5 form_factor factors.txt spacing 0.2 0.2 0.2 two_theta 10 30 lp 0\n'
        'run 1\n'
    )
    result = subprocess.run(
        [gpumd_command], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with (tmp_path / 'xrd.txt').open() as stream:
        meta = stream.readline().split()
        assert meta[:4] == ['#', 'step', '1', 'nodes']
        assert stream.readline().split() == ['#', 'h', 'k', 'l', 'two_theta_deg', 'intensity']
        data = np.loadtxt(stream, ndmin=2)
    assert data.shape == (int(meta[4]), 5)
    assert len(data) > 0 and np.isfinite(data).all()
    np.testing.assert_array_equal(data[:, :3], np.rint(data[:, :3]))
    g = 0.2 * np.linalg.norm(data[:, :3], axis=1)
    np.testing.assert_allclose(data[:, 3], np.degrees(2 * np.arcsin(0.75 * g)))
    assert np.all((data[:, 4] >= 0) & (data[:, 4] <= len(symbols) * (1 + 1e-12)))


@pytest.mark.fast
def test_xrd_builtin_form_factors(tmp_path, gpumd_command):
    atoms = make_bulk_water()
    write_xyz(str(tmp_path / 'model.xyz'), atoms)
    (tmp_path / 'run.in').write_text(
        f'potential {MODELS_DIR / "nep_water.txt"}\n'
        'velocity 300 seed 7\n'
        'ensemble nve\n'
        'time_step 0\n'
        'compute_xrd 1 1.5 spacing 0.2 0.2 0.2 two_theta 10 65 lp 0\n'
        'run 1\n'
    )
    result = subprocess.run(
        [gpumd_command], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = np.loadtxt(tmp_path / 'xrd.txt', ndmin=2)
    assert len(data) > 0
    wavevectors = 0.2 * data[:, :3]
    s_squared = 0.25 * np.sum(wavevectors ** 2, axis=1)
    # Neutral H and O values from Waasmaier and Kirfel, doi:10.1107/S0108767394013292.
    coefficients = {
        'H': ([0.413048, 0.294953, 0.187491, 0.080701, 0.023736],
              [15.569946, 32.398468, 5.711404, 61.889874, 1.334118], 0.000049),
        'O': ([2.960427, 2.508818, 0.637853, 0.722838, 1.142756],
              [14.182259, 5.936858, 0.112726, 34.958481, 0.390240], 0.027014),
    }
    phase = np.exp(2j * np.pi * wavevectors @ atoms.positions.T)
    symbols = np.asarray(atoms.get_chemical_symbols())
    amplitude = np.zeros(len(data), dtype=complex)
    for symbol, (a, b, c) in coefficients.items():
        factor = np.exp(-np.outer(s_squared, b)) @ a + c
        amplitude += factor * phase[:, symbols == symbol].sum(axis=1)
    np.testing.assert_allclose(data[:, 4], abs(amplitude) ** 2 / len(atoms), rtol=1e-10, atol=1e-10)


@pytest.mark.fast
@pytest.mark.parametrize(
    ('option', 'coefficients', 'message'),
    [
        ('form_factor', None, 'Invalid or incomplete compute_xrd option: form_factor'),
        ('form_factor factors.txt', None, 'Cannot open compute_xrd form factor file factors.txt'),
        ('form_factor factors.txt', '# empty\n', 'No coefficients in compute_xrd form factor file'),
        ('form_factor factors.txt', UNIT_CARBON.replace('C ', 'Si '), 'Missing element C'),
        ('form_factor factors.txt', UNIT_CARBON * 2, 'factors.txt, line 2: Duplicate element C'),
        ('form_factor factors.txt', 'C 0 0\n', 'factors.txt, line 1: Expected an element and nine'),
        ('form_factor factors.txt', UNIT_CARBON.replace('1\n', 'nan\n'),
         'factors.txt, line 1: Invalid coefficient: nan'),
    ],
    ids=['missing-filename', 'missing-file', 'empty', 'missing-element', 'duplicate',
         'truncated', 'nonfinite'],
)
def test_xrd_rejects_invalid_form_factors(tmp_path, gpumd_command, option, coefficients, message):
    write_xyz(str(tmp_path / 'model.xyz'), make_bulk_C())
    if coefficients is not None:
        (tmp_path / 'factors.txt').write_text(coefficients)
    (tmp_path / 'run.in').write_text(
        f'potential {MODELS_DIR / "nep_C.txt"}\n'
        'velocity 300 seed 7\n'
        'ensemble nve\n'
        'time_step 1\n'
        f'compute_xrd 1 1.5 {option}\n'
        'run 1\n'
    )
    result = subprocess.run(
        [gpumd_command], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0 and message in output, output
    assert not (tmp_path / 'xrd.txt').exists()


@pytest.mark.fast
@pytest.mark.parametrize('alias', ['literal', 'relative', 'absolute', 'symlink', 'hardlink', 'default'])
def test_xrd_preserves_form_factor_file(tmp_path, gpumd_command, alias):
    atoms = bulk('C', 'diamond', a=3.567, cubic=True)
    write_xyz(str(tmp_path / 'model.xyz'), atoms)
    original = UNIT_CARBON.encode('ascii')
    factors = tmp_path / ('xrd.txt' if alias == 'default' else 'factors.txt')
    factors.write_bytes(original)
    source = factors.name
    output = factors.name
    if alias == 'relative':
        output = './factors.txt'
    elif alias == 'absolute':
        output = factors.resolve().as_posix()
    elif alias == 'symlink':
        output = 'alias.txt'
        try:
            (tmp_path / output).symlink_to(factors)
        except OSError as error:
            if getattr(error, 'winerror', None) == 1314:
                pytest.skip('Creating symlinks requires additional Windows permissions')
            raise
    elif alias == 'hardlink':
        output = 'alias.txt'
        os.link(factors, tmp_path / output)
    elif alias == 'default':
        source = './xrd.txt'
    output_option = '' if alias == 'default' else f' file {output}'

    (tmp_path / 'run.in').write_text(
        f'potential {MODELS_DIR / "nep_C.txt"}\n'
        'velocity 300 seed 7\n'
        'ensemble nve\n'
        'time_step 0\n'
        f'compute_xrd 1 1.5 form_factor {source}{output_option}\n'
        'run 1\n'
    )
    result = subprocess.run(
        [gpumd_command], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert factors.read_bytes() == original
    assert result.returncode != 0
    assert 'requires different form factor and output filenames' in result.stdout + result.stderr


@pytest.mark.fast
def test_xrd_samples_bead_zero(tmp_path, gpumd_command):
    atoms = bulk('C', 'diamond', a=3.567, cubic=True).repeat(2)
    selected = np.arange(0, len(atoms), 2)
    other = np.arange(1, len(atoms), 2)
    write_xyz(str(tmp_path / 'model.xyz'), atoms, groupings=[[selected.tolist(), other.tolist()]])
    (tmp_path / 'factors.txt').write_text(UNIT_CARBON)
    (tmp_path / 'run.in').write_text(
        f'potential {MODELS_DIR / "nep_C.txt"}\n'
        'velocity 300 seed 7\n'
        'ensemble pimd 4 300 300 100\n'
        'time_step 0.5\n'
        'dump_beads 30 0 0\n'
        'compute_xrd 30 1.5 form_factor factors.txt spacing 0.2 0.2 0.2 '
        'two_theta 10 60 lp 0 group 0 0\n'
        'run 30\n'
    )
    result = subprocess.run(
        [gpumd_command], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr

    data = np.loadtxt(tmp_path / 'xrd.txt', ndmin=2)
    assert data.shape[0] > 0 and data.shape[1] == 5
    wavevectors = 2 * np.pi * 0.2 * data[:, :3]
    beads = np.stack([
        read(tmp_path / f'beads_dump_{index}.xyz').positions[selected] for index in range(4)
    ])

    def intensities(positions):
        amplitudes = np.exp(1j * (wavevectors @ positions.T)).sum(axis=1)
        return abs(amplitudes) ** 2 / len(selected)

    bead_zero = intensities(beads[0])
    centroid = intensities(beads.mean(axis=0))
    assert np.max(abs(bead_zero - centroid)) > 1e-5
    # dump_beads writes positions to eight decimal places.
    np.testing.assert_allclose(data[:, 4], bead_zero, rtol=1e-7, atol=2e-7)
