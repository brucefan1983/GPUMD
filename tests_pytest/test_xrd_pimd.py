"""Exercise the XRD interface and check single-bead sampling."""

import subprocess

import numpy as np
import pytest
from ase.build import bulk
from ase.io import read
from calorine.gpumd import write_xyz

from conftest import MODELS_DIR, make_bulk_C


UNIT_CARBON = 'form_factor C 0 0 0 0 0 0 0 0 1'


@pytest.mark.fast
def test_compute_xrd(tmp_path, structure, structure_name, model_path, gpumd_command):
    if structure_name == 'bulk_C':
        structure = bulk('C', 'diamond', a=3.567, cubic=True).repeat(2)
    symbols = structure.get_chemical_symbols()
    factors = ' '.join(f'form_factor {element} 0 0 0 0 0 0 0 0 1'
                       for element in sorted(set(symbols)))
    write_xyz(str(tmp_path / 'model.xyz'), structure)
    (tmp_path / 'run.in').write_text(
        f'potential {model_path}\n'
        'velocity 300 seed 7\n'
        'ensemble nve\n'
        'time_step 0\n'
        f'compute_xrd 1 1.5 spacing 0.2 0.2 0.2 two_theta 10 30 lp 0 {factors}\n'
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
@pytest.mark.parametrize(
    ('coefficients', 'message'),
    [
        ('', 'Missing compute_xrd form_factor for element C.'),
        (f'{UNIT_CARBON} {UNIT_CARBON}', 'Repeated compute_xrd form_factor for an element.'),
        ('form_factor C 0 0 0 0 0 0 0 0 nan', 'compute_xrd form_factor requires finite coefficients.'),
    ],
    ids=['missing', 'duplicate', 'nonfinite'],
)
def test_xrd_rejects_invalid_form_factors(tmp_path, gpumd_command, coefficients, message):
    write_xyz(str(tmp_path / 'model.xyz'), make_bulk_C())
    (tmp_path / 'run.in').write_text(
        f'potential {MODELS_DIR / "nep_C.txt"}\n'
        'velocity 300 seed 7\n'
        'ensemble nve\n'
        'time_step 1\n'
        f'compute_xrd 1 1.5 {coefficients}\n'
        'run 1\n'
    )
    result = subprocess.run(
        [gpumd_command], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0 and message in output, output
    assert not (tmp_path / 'xrd.txt').exists()


@pytest.mark.fast
def test_xrd_samples_bead_zero(tmp_path, gpumd_command):
    atoms = bulk('C', 'diamond', a=3.567, cubic=True).repeat(2)
    selected = np.arange(0, len(atoms), 2)
    other = np.arange(1, len(atoms), 2)
    write_xyz(str(tmp_path / 'model.xyz'), atoms, groupings=[[selected.tolist(), other.tolist()]])
    (tmp_path / 'run.in').write_text(
        f'potential {MODELS_DIR / "nep_C.txt"}\n'
        'velocity 300 seed 7\n'
        'ensemble pimd 4 300 300 100\n'
        'time_step 0.5\n'
        'dump_beads 30 0 0\n'
        'compute_xrd 30 1.5 spacing 0.2 0.2 0.2 two_theta 10 60 lp 0 group 0 0 '
        f'{UNIT_CARBON}\n'
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
