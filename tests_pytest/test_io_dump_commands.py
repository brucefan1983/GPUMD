"""Command IO smoke tests (Tier 1): dump_* commands.

Confirms each keyword runs without error (exit code 0) and produces a parseable output file of
the expected shape -- not physical correctness. See io_helpers.py's module docstring.

Known spec discrepancy, confirmed against the real gpumd binary: dump_dipole does NOT work with
qNEP (charge) models, contrary to this suite's original Tier 1 scope ("dump_dipole (qNEP models
only)"). GPUMD's own error is explicit: "dump_dipole requires the second NEP potential to be a
dipole model" -- it strictly requires a genuine TNEP dipole model (model_type 1) as the second
potential, which is a different model family from qNEP charge models (model_type charge_mode
1/2). No TNEP dipole model exists in fixtures/models/ (TNEP is explicitly deferred for this
suite -- see gpumd_pytest_suite_spec.md's "Model types and phasing" section), so dump_dipole is
skipped here entirely rather than silently reinterpreted to mean something GPUMD doesn't
actually support. Revisit once TNEP fixtures are in scope.
"""
import numpy as np
import pytest
from ase.io import read
from calorine.gpumd import read_xyz

from io_helpers import CommandIOCase, run_and_check
from test_parsing import read_thermo_out

pytestmark = pytest.mark.fast


def _check_thermo_format(path):
    data = read_thermo_out(path)
    assert data.shape[1] == 18


def _check_xyz_multi_frame(path, natoms):
    frames = read(path, index=':')
    assert len(frames) >= 1
    assert len(frames[0]) == natoms


def _check_natoms_single_frame(path, natoms):
    atoms = read_xyz(str(path))
    assert len(atoms) == natoms


def _check_columnar(path, ncols, natoms):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.shape[1] == ncols
    assert data.shape[0] % natoms == 0
    assert data.shape[0] > 0


def _build_case(name, natoms):
    """Builds the CommandIOCase for `name` freshly per test invocation, since several
    parse_check callbacks need natoms, which is only known once the structure fixture runs."""
    cases = {
        'dump_thermo': CommandIOCase(
            name='dump_thermo', run_in_lines=[('dump_thermo', 1)],
            expected_output_files=['thermo.out'], parse_check=_check_thermo_format),
        'dump_position': CommandIOCase(
            name='dump_position', run_in_lines=[('dump_position', 1)],
            expected_output_files=['movie.xyz'],
            parse_check=lambda p: _check_xyz_multi_frame(p, natoms)),
        'dump_restart': CommandIOCase(
            name='dump_restart', run_in_lines=[('dump_restart', 1)],
            expected_output_files=['restart.xyz'],
            parse_check=lambda p: _check_natoms_single_frame(p, natoms)),
        'dump_velocity': CommandIOCase(
            name='dump_velocity', run_in_lines=[('dump_velocity', 1)],
            expected_output_files=['velocity.out'],
            parse_check=lambda p: _check_columnar(p, ncols=3, natoms=natoms)),
        'dump_force': CommandIOCase(
            name='dump_force', run_in_lines=[('dump_force', 1)],
            expected_output_files=['force.out'],
            parse_check=lambda p: _check_columnar(p, ncols=3, natoms=natoms)),
        'dump_xyz': CommandIOCase(
            name='dump_xyz', run_in_lines=[('dump_xyz', [-1, 0, 1, 'dump_xyz_test.xyz'])],
            expected_output_files=['dump_xyz_test.xyz'],
            parse_check=lambda p: _check_xyz_multi_frame(p, natoms)),
        'dump_exyz': CommandIOCase(
            name='dump_exyz', run_in_lines=[('dump_exyz', [1, 1, 1, 1])],
            expected_output_files=['dump.xyz'],
            parse_check=lambda p: _check_xyz_multi_frame(p, natoms)),
    }
    return cases[name]


DUMP_COMMAND_CASE_NAMES = [
    'dump_thermo', 'dump_position', 'dump_restart', 'dump_velocity', 'dump_force', 'dump_xyz',
    'dump_exyz',
]


@pytest.mark.parametrize('case_name', DUMP_COMMAND_CASE_NAMES)
def test_command_io(tmp_path, structure, model_path, model_type, gpumd_command, case_name):
    case = _build_case(case_name, len(structure))
    run_and_check(tmp_path, structure, model_path, model_type, gpumd_command, case)


def test_dump_netcdf():
    pytest.skip(
        "the gpumd binary at the repo root was not actually compiled with -DUSE_NETCDF, "
        "despite src/makefile.loc listing it in CFLAGS -- confirmed directly: it prints "
        "'dump_netcdf is available only when USE_NETCDF flag is set.' (that exact string is "
        'present in the binary), most likely built before that flag was added or via a '
        'different invocation than makefile.loc. netCDF4 is available in this Python '
        'environment for parsing if/when the binary is rebuilt with NetCDF support.')


def test_dump_observer(tmp_path, structure, model_path, model_type, gpumd_command):
    """dump_observer needs 2+ NEP potentials with identical species/order; reusing the same
    model file for both is a valid, if physically redundant, way to satisfy that for a smoke
    test. Uses 'average' mode (single observer.out/observer.xyz) rather than 'observe' mode
    (per-potential observerN.out/xyz) since both exercise the same underlying output-writing
    code path and 'average' needs fewer expected files."""
    case = CommandIOCase(
        name='dump_observer',
        run_in_lines=[
            ('potential', str(model_path)),
            ('potential', str(model_path)),
            ('dump_observer', ['average', 1, 1, 1, 1]),
        ],
        expected_output_files=['observer.out'],
        parse_check=_check_thermo_format,
    )
    run_and_check(tmp_path, structure, model_path, model_type, gpumd_command, case)


@pytest.mark.parametrize('structure_name', ['bulk_perovskite', 'bulk_water'])
@pytest.mark.parametrize('model_type', ['qnep_mode1', 'qnep_mode2'])
def test_dump_dipole(structure_name, model_type):
    pytest.skip(
        "dump_dipole requires a genuine TNEP dipole model as its second potential ('dump_dipole "
        "requires the second NEP potential to be a dipole model', confirmed against the real "
        'gpumd binary) -- it does not accept qNEP charge models. No TNEP dipole model is in '
        'scope for this suite (TNEP is explicitly deferred). See module docstring.')
