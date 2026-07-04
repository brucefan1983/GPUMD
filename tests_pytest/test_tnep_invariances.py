"""TNEP rotational self-consistency: dipole rotates as a vector, polarizability transforms as a
rank-2 tensor, under the same rotation applied to the input structure.

GPUNEP-vs-GPUNEP only, consistent with this suite's numeric-validation strategy (see
conftest.py's module docstring) -- CPUNEP has no role here. Reuses test_invariances.py's
_rotation_matrix (Rodrigues' formula) and the same (positions - center_of_mass) @ rot.T + center
/ cell @ rot.T + wrap() convention as test_rotation_invariance, applied to the fixed (structure,
PES model, TNEP model) triples from test_io_tnep_commands.py's TNEP_CASES -- a new file rather
than new cases in test_invariances.py, since TNEP needs two explicit `potential` lines and (see
below) explicit velocity control that don't fit that file's swept structure/calculator/model_type
fixture matrix, the same reasoning that already justifies test_io_tnep_commands.py being separate
from test_io_dump_commands.py.

Rank-2 tensor rotation: polarizability.out's 6 columns (p_xx, p_yy, p_zz, p_xy, p_yz, p_zx,
confirmed against doc/gpumd/output_files/polarizability_out.rst) reconstruct to a symmetric 3x3
matrix M = [[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]]; the correct rotation for a rank-2 Cartesian
tensor under an orthogonal rotation is the congruence transform M_after = rot @ M_before @ rot.T
(re-derived from p_i = alpha_ij E_j transforming consistently under p' = R p, E' = R E, which
forces alpha' = R alpha R^T; cross-validated against tests/gpumd/dump_polarizability/
test_dump_polarizability.py's existing, calorine-confirmed column<->matrix mapping -- both
agree). Checked that GPUMD's dipole/polarizability sum (src/force/nep.cu) is built from
minimum-image-convention pairwise displacement vectors, not a raw absolute-position sum, so it is
not vulnerable to the periodic "polarization quantum" discontinuity -- rotating+wrapping a
periodic cell is safe here, same as for ordinary forces/virial.

Critical subtlety this file has to work around: dump_dipole/dump_polarizability only write output
*after* a full velocity-Verlet step -- there is no way to sample the pristine input structure
(`run 0` never enters the integration loop body, so no row is ever written for it). The usual
`velocity <T> seed 42` block's randomly-drawn initial velocities depend only on atom index/mass/
temperature, not position (src/main_gpumd/velocity.cu), so the same seed applied to the original
and rotated structures would add an *identical, un-rotated* v*dt drift to both -- breaking the
exact rotation relationship this test needs between their row-0 outputs. `velocity 0` is also
rejected by GPUMD (positive temperature required, src/main_gpumd/run.cu). Fix: bypass GPUMD's
velocity RNG by setting an explicit, fixed velocity array directly on the Atoms object (and its
consistently-rotated counterpart on the rotated Atoms) before building the calculator --
GPUNEP writes whatever velocities are already on `atoms` into model.xyz, and confirmed directly
in src/main_gpumd/velocity.cu (Velocity::initialize's `if (!has_velocity_in_xyz)` branch) that
GPUMD's `velocity` keyword leaves embedded velocities untouched rather than overwriting them, so
the usual `velocity 300 seed 42` line can stay in run_in_lines as an inert no-op. This makes the
rotation identity exact (not an approximation), for any velocity magnitude, as long as it is set
and rotated consistently between the two runs.
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
# different floating-point input, same class of noise as GPU_TRANSFORM_*_TOLERANCE (conftest.py)
# but for different physical quantities/units -- calibrated empirically against a live run rather
# than asserted a priori; an initial, much looser guess was tightened down to this once the real
# run-to-run noise floor was observed.
TNEP_ROTATION_TOLERANCE = dict(rtol=1e-4, atol=1e-5)


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
            ('potential', str(tnep_path)),
            (dump_command, 1),
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
