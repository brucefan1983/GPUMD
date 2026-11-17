"""Check momentum removal against a mass-weighted NumPy pseudoinverse."""
import shutil
import subprocess

import numpy as np
import pytest
from ase.io import read

from conftest import FIXTURES_DIR

SHAPES = {
    'dimer': np.array([[-5., 0, 0], [5., 0, 0]]),
    'line': np.array([[-5., 0, 0], [0., 0, 0], [5., 0, 0]]),
    'bent': np.array([[-5., 0, 0], [0., 1, 0], [5., 0, 0]]),
    'tetrahedron': np.array([[3., 3, 3], [-3., -3, 3], [-3., 3, -3], [3., -3, -3]]),
}
DT_FS = 0.1


def _rotation(rotated):
    if not rotated:
        return np.eye(3)
    axis = np.array([1., 2, 3]) / np.sqrt(14)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angle = 0.73
    return np.cos(angle) * np.eye(3) + (1-np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * skew


def _velocities(positions):
    return np.array([0.03, -0.02, 0.01]) + np.cross([0.013, -0.017, 0.021], positions) + 0.002 * positions


def _project(positions, velocities, masses, labels=None):
    if labels is None:
        labels = np.zeros(len(masses), dtype=int)
    result = np.empty_like(velocities)
    for label in np.unique(labels):
        select = labels == label
        m = masses[select]
        r = positions[select] - np.average(positions[select], axis=0, weights=m)
        v = velocities[select] - np.average(velocities[select], axis=0, weights=m)
        inertia = sum(mi * (np.dot(x, x) * np.eye(3) - np.outer(x, x)) for mi, x in zip(m, r))
        angular = np.sum(m[:, None] * np.cross(r, v), axis=0)
        omega = np.linalg.pinv(inertia, rcond=1e-12) @ angular
        result[select] = v - np.cross(omega, r)
    return result


def _run(path, executable, positions, velocities, masses, labels=None, steps=1):
    path.mkdir(parents=True, exist_ok=True)
    side = max(40., 4 * np.max(np.abs(positions)) + 40)
    positions = positions + side / 2
    properties = 'species:S:1:pos:R:3:mass:R:1'
    if velocities is not None:
        properties += ':vel:R:3'
    if labels is not None:
        properties += ':group:I:1'
    rows = [str(len(masses)), f'pbc="F F F" Lattice="{side} 0 0 0 {side} 0 0 0 {side}" Properties={properties}']
    for n, (p, m) in enumerate(zip(positions, masses)):
        data = [*p, m]
        if velocities is not None:
            data.extend(velocities[n])
        line = 'C ' + ' '.join(f'{x:.17g}' for x in data)
        if labels is not None:
            line += f' {labels[n]}'
        rows.append(line)
    (path / 'model.xyz').write_text('\n'.join(rows) + '\n')
    shutil.copyfile(FIXTURES_DIR / 'replica/nep.txt', path / 'nep.txt')
    if velocities is None:
        command = 'velocity 300 seed 42\n'
    else:
        command = 'correct_velocity 10' + (' 0' if labels is not None else '') + '\n'
    (path / 'run.in').write_text('potential nep.txt\nensemble nve\ntime_step 0.1\n' + command
                                + 'dump_xyz 1 trajectory.xyz mass velocity precision double\n'
                                + f'run {steps}\n')
    process = subprocess.run([executable], cwd=path, capture_output=True, text=True, timeout=120)
    (path / 'stdout.log').write_text(process.stdout)
    (path / 'stderr.log').write_text(process.stderr)
    assert process.returncode == 0, process.stdout + process.stderr
    return positions, read(path / 'trajectory.xyz', index=':')


def _assert_momenta(positions, velocities, masses, labels=None):
    if labels is None:
        labels = np.zeros(len(masses), dtype=int)
    for label in np.unique(labels):
        select = labels == label
        m = masses[select]
        r = positions[select] - np.average(positions[select], axis=0, weights=m)
        v = velocities[select]
        linear = np.sum(m[:, None] * v, axis=0)
        angular = np.sum(m[:, None] * np.cross(r, v), axis=0)
        np.testing.assert_allclose(linear, 0, rtol=0, atol=2e-11)
        np.testing.assert_allclose(angular, 0, rtol=0, atol=2e-10)


@pytest.mark.parametrize('shape', list(SHAPES))
@pytest.mark.parametrize('rotated', [False, True])
@pytest.mark.parametrize('unequal_masses', [False, True])
def test_correction_matches_rigid_motion_projection(tmp_path, gpumd_command, shape, rotated, unequal_masses):
    r = SHAPES[shape] @ _rotation(rotated).T
    masses = np.arange(1, len(r)+1, dtype=float) * 12 if unequal_masses else np.full(len(r), 12.)
    v = _velocities(r)
    positions, frames = _run(tmp_path, gpumd_command, r, v, masses)
    expected = _project(positions, v, masses)
    actual = frames[0].arrays['vel']
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=2e-13)
    np.testing.assert_allclose(frames[0].positions, positions + DT_FS * expected, rtol=0, atol=2e-12)
    _assert_momenta(positions, actual, masses)
    assert np.sum(masses[:, None] * actual**2) <= np.sum(masses[:, None] * v**2) * (1+1e-12)


@pytest.mark.parametrize('shape', ['dimer', 'line'])
@pytest.mark.parametrize('rotated', [False, True])
@pytest.mark.parametrize('unequal_masses', [False, True])
def test_initial_velocity_removes_linear_group_rotation(tmp_path, gpumd_command, shape, rotated, unequal_masses):
    r = SHAPES[shape] @ _rotation(rotated).T
    masses = np.arange(1, len(r)+1, dtype=float) * 12 if unequal_masses else np.full(len(r), 12.)
    positions, frames = _run(tmp_path, gpumd_command, r, None, masses)
    velocities = frames[0].arrays['vel']
    assert np.isfinite(velocities).all()
    _assert_momenta(positions, velocities, masses)
    # Initialization must retain the available internal motion rather than zero all velocities.
    assert np.sum(masses[:, None] * velocities**2) > 0


@pytest.mark.parametrize('scale', [0.001, 1., 1000.])
@pytest.mark.parametrize('rotated', [False, True])
def test_linear_correction_is_independent_of_coordinate_scale(tmp_path, gpumd_command, scale, rotated):
    r = scale * SHAPES['line'] @ _rotation(rotated).T
    masses = np.array([12., 24, 36])
    v = _velocities(r)
    positions, frames = _run(tmp_path, gpumd_command, r, v, masses)
    np.testing.assert_allclose(frames[0].arrays['vel'], _project(positions, v, masses), rtol=2e-9, atol=2e-11)


@pytest.mark.parametrize('unequal_masses', [False, True])
def test_interleaved_linear_groups(tmp_path, gpumd_command, unequal_masses):
    a = SHAPES['line']
    b = a @ _rotation(True).T + [10, 10, 10]
    r = np.stack((a, b), axis=1).reshape(-1, 3)
    v = np.stack((_velocities(a), -_velocities(b)), axis=1).reshape(-1, 3)
    labels = np.tile([0, 2], 3)  # Group 1 is empty; each occupied group is non-contiguous.
    masses = np.arange(1, len(r)+1, dtype=float) * 12 if unequal_masses else np.full(len(r), 12.)
    positions, frames = _run(tmp_path, gpumd_command, r, v, masses, labels)
    np.testing.assert_allclose(frames[0].arrays['vel'], _project(positions, v, masses, labels), rtol=1e-10, atol=2e-13)
    _assert_momenta(positions, frames[0].arrays['vel'], masses, labels)


def test_single_atom_group_remains_finite(tmp_path, gpumd_command):
    r = np.vstack((SHAPES['line'], [10, 10, 10]))
    masses = np.full(len(r), 12.)
    labels = np.array([0, 0, 0, 1])
    v = _velocities(r)
    positions, frames = _run(tmp_path, gpumd_command, r, v, masses, labels)
    actual = frames[0].arrays['vel']
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, _project(positions, v, masses, labels), rtol=1e-10, atol=2e-13)


def test_repeated_corrections_preserve_internal_motion(tmp_path, gpumd_command):
    r = SHAPES['line'] @ _rotation(True).T
    masses = np.array([12., 24, 36])
    v = _velocities(r)
    positions, frames = _run(tmp_path, gpumd_command, r, v, masses, steps=21)
    expected = _project(positions, v, masses)
    for step, frame in enumerate(frames, start=1):
        np.testing.assert_allclose(frame.arrays['vel'], expected, rtol=1e-10, atol=2e-13)
        np.testing.assert_allclose(frame.positions, positions + step * DT_FS * expected, rtol=0, atol=2e-12)


def test_correction_rotates_with_the_structure(tmp_path, gpumd_command):
    r = SHAPES['line']
    v = _velocities(r)
    masses = np.array([12., 24, 36])
    _, a = _run(tmp_path / 'original', gpumd_command, r, v, masses)
    rotation = _rotation(True)
    _, b = _run(tmp_path / 'rotated', gpumd_command, r @ rotation.T, v @ rotation.T, masses)
    np.testing.assert_allclose(b[0].arrays['vel'], a[0].arrays['vel'] @ rotation.T, rtol=1e-10, atol=2e-13)
