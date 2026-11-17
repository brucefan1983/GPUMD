"""TNEP rotational consistency for dipole vectors and polarizability tensors.

The tests reuse TNEP_CASES and rotate positions, cell vectors and explicit
velocities together. Outputs are sampled after an integration step, so rotating
velocities is necessary to preserve the trajectory's rotation relationship.

For polarizability, columns xx, yy, zz, xy, yz, zx reconstruct a symmetric
matrix M; the expected rotated matrix is rot @ M @ rot.T. Each comparison uses
the same GPUMD executable and potential models on the original and rotated
systems.
"""
import numpy as np
import pytest

from conftest import MODELS_DIR
from io_helpers import CommandIOCase, run_command_io_case
from test_invariances import _rotation_matrix
from test_io_tnep_commands import TNEP_CASES

pytestmark = pytest.mark.fast

ANGLE = 37.0  # degrees; same fixed value as test_invariances.py, arbitrary but reproducible
AXIS = np.array([0.3, 0.5, 0.8113883008])  # same fixed axis as test_invariances.py
VELOCITY_SEED = 11
VELOCITY_SCALE = 0.01  # Angstrom/fs; comfortably above calorine.gpumd.write_xyz's 1e-6
# no-velocity-data threshold -- magnitude is otherwise immaterial to correctness, since the same
# array is rotated consistently for the "rotated" run rather than relied upon to be negligible.

# GPU-vs-GPU under a geometric transform: two independent gpumd subprocess runs on genuinely
# different floating-point input, the same class of noise as GPU_TRANSFORM_*_TOLERANCE
# (conftest.py) but for different physical quantities. Rotating the cell changes every coordinate
# and the neighbor list order with it, so the difference is larger than between two runs of the
# same configuration: up to 1e-5 for the polarizability, whose off-diagonal components are of
# order 1e-2 against a diagonal of order 1e2, and 1.1e-6 for the dipole. atol carries the
# near-zero components, where a relative bound alone is unstable, and sits an order of magnitude
# above that. A broken rotation relationship shows up at the scale of the components themselves,
# far above this bound.
TNEP_ROTATION_TOLERANCE = dict(rtol=1e-4, atol=1e-4)


def _fixed_velocities(n_atoms):
    return np.random.default_rng(VELOCITY_SEED).normal(scale=VELOCITY_SCALE, size=(n_atoms, 3))


def _rotate_atoms(atoms, rot):
    rotated = atoms.copy()
    center = rotated.get_center_of_mass()
    rotated.set_positions((rotated.get_positions() - center) @ rot.T + center)
    if any(rotated.pbc):
        rotated.set_cell(rotated.get_cell() @ rot.T, scale_atoms=False)
        rotated.wrap()
    rotated.set_velocities(atoms.get_velocities() @ rot.T)
    return rotated


def _run_and_read_row0(tmp_path, atoms, pes_path, tnep_path, dump_command, expected_output_file,
                       gpumd_command):
    case = CommandIOCase(
        name='tnep_rotation',
        run_in_lines=[
            ('potential', str(pes_path)),
            (dump_command, [1, str(tnep_path)]),
        ],
        expected_output_files=[expected_output_file],
    )
    result = run_command_io_case(tmp_path, atoms, pes_path, 'nep', gpumd_command, case)
    assert result.returncode == 0, (
        f'tnep rotation check: gpumd exited {result.returncode}\n'
        f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}')
    data = np.loadtxt(tmp_path / expected_output_file)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[0, 1:]


def _polarizability_matrix(cols6):
    xx, yy, zz, xy, yz, zx = cols6
    return np.array([[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]])


def _polarizability_columns(matrix):
    return np.array([
        matrix[0, 0], matrix[1, 1], matrix[2, 2], matrix[0, 1], matrix[1, 2], matrix[2, 0],
    ])


@pytest.mark.parametrize('case_name', list(TNEP_CASES))
def test_tnep_rotation_self_consistency(tmp_path, gpumd_command, case_name):
    case = TNEP_CASES[case_name]
    pes_path = MODELS_DIR / case['pes_model']
    tnep_path = MODELS_DIR / case['tnep_model']

    original = case['structure_builder']()
    original.set_velocities(_fixed_velocities(len(original)))
    rot = _rotation_matrix(ANGLE, AXIS)
    rotated = _rotate_atoms(original, rot)

    value_before = _run_and_read_row0(
        tmp_path / 'original', original, pes_path, tnep_path, case['dump_command'],
        case['expected_output_file'], gpumd_command)
    value_after = _run_and_read_row0(
        tmp_path / 'rotated', rotated, pes_path, tnep_path, case['dump_command'],
        case['expected_output_file'], gpumd_command)

    if case['dump_command'] == 'dump_dipole':
        expected_after = value_before @ rot.T
    else:
        expected_after = _polarizability_columns(
            rot @ _polarizability_matrix(value_before) @ rot.T)
    assert np.allclose(value_after, expected_after, **TNEP_ROTATION_TOLERANCE)
