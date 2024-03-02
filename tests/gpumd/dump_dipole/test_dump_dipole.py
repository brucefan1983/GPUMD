import os
from subprocess import CalledProcessError, run

import numpy as np
import pytest
from ase.io import read
from calorine.calculators import CPUNEP, GPUNEP

suite_path = 'gpumd/dump_dipole'
repo_dir = f'{os.path.expanduser("~")}/repos/GPUMD/'
test_folder = f'{repo_dir}/tests/gpumd/dump_dipole/self_consistent/'


def run_md(params, path, repeat=1):
    gpumd_command = f'{repo_dir}/src/gpumd'
    structure = read(f'{test_folder}/model.xyz')
    structure = structure.repeat(repeat)
    calc = GPUNEP(f"{test_folder}/nep.txt", command=gpumd_command)
    structure.calc = calc
    calc.set_directory(path)
    calc.run_custom_md(params, only_prepare=True)
    run('ls', cwd=path, check=True)
    return run(gpumd_command, cwd=path, capture_output=True)


@pytest.fixture
def md_without_dip(tmp_path, request):
    path = tmp_path / 'without_dip'
    params = [
        ("potential", f"{test_folder}/nep.txt"),
        ("time_step", 1),
        ("velocity", 300),
        ("ensemble", "nve"),
        ("dump_position", 1),
        ("dump_force", 1),
        ("dump_thermo", 1),
        ("dump_velocity", 1),
        ("run", 10),
    ]
    run_md(params, path, repeat=request.param)
    return path


@pytest.fixture
def md(tmp_path, request):
    path = tmp_path / 'with_dip'
    dipole_model = f"{test_folder}/nep4_dipole.txt"
    params = [
        ("potential", f"{test_folder}/nep.txt"),
        ("potential", dipole_model),
        ("time_step", 1),
        ("velocity", 300),
        ("ensemble", "nve"),
        ("dump_dipole", 1),
        ("dump_position", 1),
        ("dump_force", 1),
        ("dump_thermo", 1),
        ("dump_velocity", 1),
        ("run", 10),
    ]
    run_md(params, path, repeat=request.param)
    return path, dipole_model

@pytest.mark.parametrize('md', [1, 2], indirect=True)
def test_dump_dipole_self_consistent(md):
    """Ensure dump_dipole writes dipoles that are consistent with the NEP executable"""
    md_path, dipole_model = md
    dipole = np.loadtxt(f'{md_path}/dipole.out')
    # Read positions, and predict dipole with dipole model
    for gpu_dipole, conf in zip(dipole[:, 1:], read(f'{md_path}/movie.xyz', ':')):
        conf.calc = CPUNEP(dipole_model)
        cpu_dipole = conf.get_dipole_moment()
        assert np.allclose(cpu_dipole, gpu_dipole, atol=1e-2, rtol=1e-6)


@pytest.mark.parametrize('md', [1], indirect=True)
def test_dump_dipole_numeric(md):
    """
    Compare with a hardcoded value from the self consistent case above, in case
    later updates breaks one of the implemenetations in either CPUNEP or GPUMD.
    """
    md_path, _ = md
    dipole = np.loadtxt(f'{md_path}/dipole.out')
    gpu_dipole = dipole[0, 1:]
    cpu_dipole = [
        4.79686227,
        0.01536603,
        0.07771485,
    ]
    assert np.allclose(cpu_dipole, gpu_dipole, atol=1e-3, rtol=1e-6)


@pytest.mark.parametrize('md, md_without_dip', [[1, 1]], indirect=True)
def test_dump_dipole_does_not_change_forces_and_virials(md, md_without_dip):
    """Ensure that all regular observables are unchanged"""
    md_path, _ = md
    md_without_dip_path = md_without_dip
    files = ('thermo.out', 'force.out', 'velocity.out')
    for file in files:
        pol_content = np.loadtxt(os.path.join(md_path, file))
        reg_content = np.loadtxt(os.path.join(md_without_dip_path, file))
        assert np.allclose(pol_content, reg_content, atol=1e-12, rtol=1e-6)


def test_dump_dipole_invalid_potential(tmp_path):
    """
    Should raise an error when the second specified potential
    is not a dipole model.
    """
    params = [
        ("potential", f"{test_folder}/nep.txt"),
        ("potential", f"{test_folder}/nep.txt"),
        ("time_step", 1),
        ("velocity", 300),
        ("ensemble", "nve"),
        ("dump_dipole", 1),
        ("run", 10),
    ]
    process = run_md(params, tmp_path)
    assert 'dump_dipole requires the second NEP potential to be a dipole model' in str(
        process.stderr
    )


def test_dump_dipole_missing_potential(tmp_path):
    """Should raise an error when only a single NEP potential is specified."""
    params = [
        ("potential", f"{test_folder}/nep.txt"),
        ("time_step", 1),
        ("velocity", 300),
        ("ensemble", "nve"),
        ("dump_dipole", 1),
        ("run", 10),
    ]
    process = run_md(params, tmp_path)
    assert 'dump_dipole requires two potentials to be specified.' in str(process.stderr)
