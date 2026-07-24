"""Command IO smoke tests (Tier 1): dump_* commands.

Confirms each keyword runs without error (exit code 0) and produces a parseable output file of
the expected shape -- not physical correctness. See io_helpers.py's module docstring.

dump_dipole (and its rank-2 counterpart dump_polarizability) are covered in
test_io_tnep_commands.py, not here -- both strictly require a genuine TNEP model (not a qNEP
charge model, contrary to this suite's original Tier 1 scope) as the second `potential`, which
doesn't fit this file's calculator/model_type fixture pattern the way the rest of these dump_*
commands do.
"""
import numpy as np
import pytest
from ase.io import read
from calorine.gpumd import read_xyz

from io_helpers import BASE_N_STEPS, CommandIOCase, run_and_check, run_command_io_case
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


def _check_netcdf_result(result):
    output = result.stdout + result.stderr
    if 'dump_netcdf is available only when USE_NETCDF flag is set' in output:
        pytest.skip('gpumd binary was built without USE_NETCDF')
    assert result.returncode == 0, (
        f'dump_netcdf: gpumd exited {result.returncode}\nstdout:\n{result.stdout}\n'
        f'stderr:\n{result.stderr}')


def test_dump_netcdf_default_overwrite_and_append(
        tmp_path, structure, model_path, model_type, gpumd_command):
    netcdf4 = pytest.importorskip('netCDF4')
    output_path = tmp_path / 'sed.nc'
    output_path.write_bytes(b'an existing file that must be overwritten')
    case = CommandIOCase(
        name='dump_netcdf',
        run_in_lines=[
            ('dump_netcdf', [-1, 0, 1, 1, 'sed.nc']),
            ('run', BASE_N_STEPS),
            ('dump_netcdf', [-1, 0, 1, 1, 'sed.nc']),
        ],
        expected_output_files=['sed.nc'],
    )
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    _check_netcdf_result(result)

    with netcdf4.Dataset(output_path) as dataset:
        assert dataset.data_model == 'NETCDF3_64BIT_OFFSET'
        assert len(dataset.dimensions['frame']) == 2 * BASE_N_STEPS
        assert len(dataset.dimensions['atom']) == len(structure)
        assert dataset.variables['coordinates'].dtype == np.dtype('float32')
        assert dataset.variables['velocities'].dtype == np.dtype('float32')
        assert dataset.variables['type'].dimensions == ('frame', 'atom')
        assert dataset.variables['type'].shape == (2 * BASE_N_STEPS, len(structure))
        assert dataset.getncattr('gpumd_compression_level') == -1


def test_dump_netcdf_without_velocity(
        tmp_path, structure, model_path, model_type, gpumd_command):
    netcdf4 = pytest.importorskip('netCDF4')
    case = CommandIOCase(
        name='dump_netcdf_positions',
        run_in_lines=[('dump_netcdf', [-1, 0, 1, 0, 'positions.nc'])],
        expected_output_files=['positions.nc'],
    )
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    _check_netcdf_result(result)

    with netcdf4.Dataset(tmp_path / 'positions.nc') as dataset:
        assert dataset.variables['coordinates'].dtype == np.dtype('float32')
        assert 'velocities' not in dataset.variables


def test_dump_netcdf_group_double_deflate(
        tmp_path, structure, model_path, model_type, gpumd_command):
    netcdf4 = pytest.importorskip('netCDF4')
    half = len(structure) // 2
    expected_group_size = len(structure) - half
    case = CommandIOCase(
        name='dump_netcdf_group',
        run_in_lines=[('dump_netcdf', [
            0, 1, 1, 1, 'group.nc',
            'precision', 'double', 'compression', 'deflate', 1,
        ])],
        expected_output_files=['group.nc'],
        n_groups=2,
    )
    result = run_command_io_case(
        tmp_path, structure, model_path, model_type, gpumd_command, case)
    _check_netcdf_result(result)

    with netcdf4.Dataset(tmp_path / 'group.nc') as dataset:
        coordinates = dataset.variables['coordinates']
        velocities = dataset.variables['velocities']
        assert dataset.data_model == 'NETCDF4'
        assert len(dataset.dimensions['frame']) == BASE_N_STEPS
        assert len(dataset.dimensions['atom']) == expected_group_size
        assert coordinates.dtype == np.dtype('float64')
        assert velocities.dtype == np.dtype('float64')
        assert coordinates.filters()['zlib']
        assert coordinates.filters()['complevel'] == 1
        assert velocities.filters()['zlib']
        assert dataset.getncattr('gpumd_grouping_method') == 0
        assert dataset.getncattr('gpumd_group_id') == 1


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
