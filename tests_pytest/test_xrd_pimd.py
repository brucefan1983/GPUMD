"""Exercise the XRD interface and check single-bead sampling."""

import subprocess

import numpy as np
import pytest
from ase.build import bulk
from ase.io import read
from calorine.gpumd import write_xyz

from conftest import MODELS_DIR, make_bulk_water


@pytest.mark.fast
def test_compute_xrd(tmp_path, structure, structure_name, model_path, gpumd_command):
    if structure_name == 'bulk_C':
        structure = bulk('C', 'diamond', a=3.567, cubic=True).repeat(2)
    write_xyz(str(tmp_path / 'model.xyz'), structure)
    (tmp_path / 'run.in').write_text(
        f'potential {model_path}\n'
        'velocity 300 seed 7\n'
        'ensemble nve\n'
        'time_step 0\n'
        'compute_xrd 1 1.5 spacing 0.2 0.2 0.2 two_theta 10 30 lp 0\n'
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
    assert np.all(data[:, 4] >= 0)
    assert np.any(data[:, 4] > 0)


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
    ('arguments', 'message'),
    [
        ('1.5 form_factor factors.txt', 'Invalid or incomplete compute_xrd option: form_factor'),
        ('0.1 two_theta 10 179', 'require sin(theta)/wavelength <= 6 A^-1'),
    ],
    ids=['removed-form-factor-option', 'outside-form-factor-range'],
)
def test_xrd_rejects_unsupported_inputs(tmp_path, gpumd_command, arguments, message):
    atoms = bulk('C', 'diamond', a=3.567, cubic=True)
    write_xyz(str(tmp_path / 'model.xyz'), atoms)
    (tmp_path / 'run.in').write_text(
        f'potential {MODELS_DIR / "nep_C.txt"}\n'
        'velocity 300 seed 7\n'
        'ensemble nve\n'
        'time_step 0\n'
        f'compute_xrd 1 {arguments}\n'
        'run 1\n'
    )
    result = subprocess.run(
        [gpumd_command], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0 and message in output, output
    assert not (tmp_path / 'xrd.txt').exists()


@pytest.mark.fast
def test_xrd_appends_across_runs(tmp_path, gpumd_command):
    atoms = bulk('C', 'diamond', a=3.567, cubic=True)
    write_xyz(str(tmp_path / 'model.xyz'), atoms)
    (tmp_path / 'run.in').write_text(
        f'potential {MODELS_DIR / "nep_C.txt"}\n'
        'velocity 300 seed 7\n'
        'ensemble nve\n'
        'time_step 0\n'
        'compute_xrd 1 1.5 two_theta 10 60 file pattern.txt\n'
        'run 1\n'
        'ensemble nve\n'
        'time_step 0\n'
        'compute_xrd 2 1.5 two_theta 10 60 file pattern.txt\n'
        'run 2\n'
    )
    result = subprocess.run(
        [gpumd_command], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / 'xrd.txt').exists()
    with (tmp_path / 'pattern.txt').open() as stream:
        blocks = []
        for step in (1, 2):
            meta = stream.readline().split()
            assert meta[:4] == ['#', 'step', str(step), 'nodes']
            nodes = int(meta[4])
            assert nodes > 0
            assert stream.readline().split() == ['#', 'h', 'k', 'l', 'two_theta_deg', 'intensity']
            block = np.loadtxt(stream, ndmin=2, max_rows=nodes)
            assert block.shape == (nodes, 5)
            blocks.append(block)
        assert stream.read() == ''
    first, second = blocks
    np.testing.assert_array_equal(first[:, :4], second[:, :4])
    np.testing.assert_allclose(first[:, 4], second[:, 4], rtol=1e-12, atol=1e-11)


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
        'compute_xrd 30 1.5 spacing 0.2 0.2 0.2 '
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

    # Neutral carbon from Waasmaier and Kirfel, doi:10.1107/S0108767394013292.
    a = [2.657506, 1.078079, 1.490909, -4.241070, 0.713791]
    b = [14.780758, 0.776775, 42.086842, -0.000294, 0.239535]
    s_squared = np.sum((wavevectors / (4 * np.pi)) ** 2, axis=1)
    factor = np.exp(-np.outer(s_squared, b)) @ a + 4.297983

    def intensities(positions):
        amplitudes = factor * np.exp(1j * (wavevectors @ positions.T)).sum(axis=1)
        return abs(amplitudes) ** 2 / len(selected)

    bead_zero = intensities(beads[0])
    centroid = intensities(beads.mean(axis=0))
    assert np.max(abs(bead_zero - centroid)) > 1e-5
    # dump_beads writes positions to eight decimal places.
    np.testing.assert_allclose(data[:, 4], bead_zero, rtol=1e-7, atol=2e-7)
