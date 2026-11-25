"""Numerical checks for compute_chunk through model.xyz, run.in and compute_chunk.out."""

from itertools import product
import subprocess

import numpy as np
import pytest


pytestmark = pytest.mark.fast


def run_chunk(tmp_path, gpumd_command, cell, points, velocities, command, steps=1, deform=''):
    (tmp_path / 'lj.txt').write_text('lj 1 Ar\n1.0 1.0 0.8\n')
    lattice = ' '.join(map(str, np.ravel(cell)))
    atoms = [f'{len(points)}\nLattice="{lattice}" pbc="T T T" '
             'Properties=species:S:1:pos:R:3:vel:R:3:mass:R:1']
    for i, (position, velocity) in enumerate(zip(points, velocities)):
        atoms.append('Ar ' + ' '.join(map(str, [*position, *velocity, 40 + i])))
    (tmp_path / 'model.xyz').write_text('\n'.join(atoms) + '\n')
    (tmp_path / 'run.in').write_text(
        f'potential lj.txt\ntime_step 1\nensemble nve\n{deform}'
        f'compute_chunk {command}\nrun {steps}\n'
    )
    result = subprocess.run([gpumd_command], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return np.loadtxt(tmp_path / 'compute_chunk.out', ndmin=2)


@pytest.mark.parametrize('properties', ['vx vy vz', 'vz temperature vx vy'])
def test_chunk_velocity_units(tmp_path, gpumd_command, properties):
    points = [[2, 1, 1], [4, 5, 5]]
    velocities = np.array([[.01, -.02, .03], [.03, -.04, -.01]])
    data = run_chunk(tmp_path, gpumd_command, np.eye(3) * 10, points, velocities,
                     f'1 2 bin/1d x lower 5 {properties}', steps=2)
    vx, vy, vz = velocities.mean(axis=0)
    ke_twice = np.dot([40, 41], (velocities ** 2).sum(axis=1)) * 10.18051 ** 2
    temperature = ke_twice / (3 * 2 * 8.617333262e-5)
    values = {'vx': vx, 'vy': vy, 'vz': vz, 'temperature': temperature}
    expected = [values[name] for name in properties.split()]
    np.testing.assert_allclose(data, [[0, 2.5, 2, *expected],
                                      [1, 7.5, 0, *np.zeros(len(expected))]], rtol=1e-9)


@pytest.mark.parametrize('tilt', [0, 5], ids=['orthogonal', 'triclinic'])
@pytest.mark.parametrize('axes', ['x', 'yx', 'zxy'])
def test_chunk_geometry(tmp_path, gpumd_command, tilt, axes):
    cell = np.array([[10, 0, 0], [tilt, 10, 0], [0, 0, 10]])
    # Every bin is occupied. The extra atom changes bin in a Cartesian x partition.
    fractional = np.array([*product([.2, .85], repeat=3), (.4, .85, .2)])
    directions = ['xyz'.index(axis) for axis in axes]
    heights = np.array([100 / np.hypot(10, tilt), 10, 10])[directions]
    spec = ' '.join(f'{axis} lower 6' for axis in axes)
    data = run_chunk(tmp_path, gpumd_command, cell, fractional @ cell, np.zeros_like(fractional),
                     f'1 1 bin/{len(axes)}d {spec} density/number density/mass')
    inferred_volumes = []
    for index, row in enumerate(data):
        upper_half = np.array([(index >> d) & 1 for d in range(len(axes))], dtype=bool)
        centers = np.where(upper_half, (6 + heights) / 2, 3)
        widths = np.where(upper_half, heights - 6, 6)
        volume = 1000 * np.prod(widths / heights)
        selected = ((fractional[:, directions] * heights >= 6) == upper_half).all(axis=1)
        count = selected.sum()
        mass = (40 + np.flatnonzero(selected)).sum()
        np.testing.assert_allclose(row[:1 + len(axes)], [index, *centers], rtol=0, atol=5.1e-7)
        np.testing.assert_allclose(row[1 + len(axes):], [count, count / volume, mass / volume],
                                   rtol=1e-9)
        inferred_volumes.append(row[-3] / row[-2])
    assert len(data) == 2 ** len(axes)
    assert sum(inferred_volumes) == pytest.approx(1000, rel=1e-9)


def test_single_triclinic_chunk_spans_whole_cell(tmp_path, gpumd_command):
    cell = [[10, 0, 0], [5, 10, 0], [0, 0, 10]]
    points = [[3, 2, 2], [8, 7, 6]]
    data = run_chunk(tmp_path, gpumd_command, cell, points, np.zeros((2, 3)),
                     '1 1 bin/2d x lower 20 y lower 20 density/number')
    count, density = data[0, -2:]
    assert count == len(points)
    assert count / density == pytest.approx(1000, rel=1e-9)


def test_chunk_deforming_box(tmp_path, gpumd_command):
    cell = np.eye(3) * 10
    points = [[2, 2, 2], [8, 8, 8]]
    data = run_chunk(tmp_path, gpumd_command, cell, points, np.zeros((2, 3)),
                     '1 2 bin/1d x lower 6 density/number', steps=4, deform='deform xx 1\n')
    expected = []
    for lengths in (np.array([11, 12]), np.array([13, 14])):
        for i, (center, width) in enumerate(((.3, .6), (.8, .4))):
            density = np.mean(1 / (lengths * 100 * width))
            expected.append([i, center * lengths.mean(), 1, density])
    np.testing.assert_allclose(data, expected, rtol=1e-9)
