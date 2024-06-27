import os
from subprocess import CalledProcessError, run

import numpy as np
import pytest
from ase.io import read
from calorine.calculators import CPUNEP, GPUNEP
from ase.units import Bohr

suite_path = 'gpumd/cavity'
repo_dir = f'{os.path.expanduser("~")}/repos/GPUMD/'
test_folder = f'{repo_dir}/tests/gpumd/cavity/self_consistent/'


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
def md_without_cavity(tmp_path, request):
    path = tmp_path / 'without_cavity'
    params = [
        ("potential", f"{test_folder}/nep.txt"),
        ("time_step", 1),
        ("velocity", 300),
        ("ensemble", "nve"),
        ("dump_position", 1),
        ("dump_force", 1),
        ("dump_thermo", 1),
        ("dump_velocity", 1),
        ("run", 2),
    ]
    run_md(params, path, repeat=request.param)
    return path


@pytest.fixture
def md(tmp_path, request):
    path = tmp_path / 'with_cavity'
    dipole_model = f"{test_folder}/nep4_dipole.txt"
    params = [
        ("potential", f"{test_folder}/nep.txt"),
        ("potential", dipole_model),
        ("time_step", 1),
        ("velocity", 300),
        ("ensemble", "nve"),
        ("cavity", (1.0, 1.0, -1)),
        ("dump_position", 1),
        ("dump_force", 1),
        ("dump_thermo", 1),
        ("dump_velocity", 1),
        ("run", 2),
    ]
    run_md(params, path, repeat=request.param)
    return path, dipole_model

@pytest.mark.parametrize('md', [1], indirect=True)
def test_cavity_self_consistent(md):
    """Ensure cavity writes dipoles and dipole jacobians that are consistent with the NEP executable"""
    md_path, dipole_model = md
    jacobian = np.loadtxt(f'{md_path}/jacobian.out')
    charge = -1
    # Read positions, and predict dipole with dipole model
    for gpu_dipole, conf in zip(jacobian[:, 1:4], read(f'{md_path}/movie.xyz', ':')):
        COM = conf.get_center_of_mass()
        conf.calc = CPUNEP(dipole_model)
        cpu_dipole = conf.get_dipole_moment() * Bohr + charge * COM
        assert np.allclose(cpu_dipole, gpu_dipole, atol=1e-2, rtol=1e-6)
    for gpu_jacobian, conf in zip(jacobian[:, 4:], read(f'{md_path}/movie.xyz', ':')):
        calc = CPUNEP(dipole_model)
        conf.calc = calc
        cpu_jacobian = calc.get_dipole_gradient(displacement=0.01, method='second order central difference', charge=-1/Bohr) * Bohr # CPUNEP corrects for center of mass, but not the unit conversion from au to ASE units. 
        gj = gpu_jacobian.reshape(len(conf), 3, 3)
        assert np.allclose(cpu_jacobian, gj, atol=5e-2, rtol=1e-6)



# @pytest.mark.parametrize('md', [1], indirect=True)
# def test_cavity_numeric(md):
#     """
#     Compare with a hardcoded value from the self consistent case above, in case
#     later updates breaks one of the implemenetations in either CPUNEP or GPUMD.
#     """
#     md_path, _ = md
#     dipole = np.loadtxt(f'{md_path}/dipole.out')
#     gpu_dipole = dipole[0, 1:]
#     cpu_dipole = [
#         4.79686227,
#         0.01536603,
#         0.07771485,
#     ]
#     assert np.allclose(cpu_dipole, gpu_dipole, atol=1e-3, rtol=1e-6)
# 
# 
# @pytest.mark.parametrize('md, md_without_dip', [[1, 1]], indirect=True)
# def test_cavity_does_not_change_forces_and_virials(md, md_without_dip):
#     """Ensure that all regular observables are unchanged"""
#     md_path, _ = md
#     md_without_dip_path = md_without_dip
#     files = ('thermo.out', 'force.out', 'velocity.out')
#     for file in files:
#         pol_content = np.loadtxt(os.path.join(md_path, file))
#         reg_content = np.loadtxt(os.path.join(md_without_dip_path, file))
#         assert np.allclose(pol_content, reg_content, atol=1e-12, rtol=1e-6)
# 
# 
# def test_cavity_invalid_potential(tmp_path):
#     """
#     Should raise an error when the second specified potential
#     is not a dipole model.
#     """
#     params = [
#         ("potential", f"{test_folder}/nep.txt"),
#         ("potential", f"{test_folder}/nep.txt"),
#         ("time_step", 1),
#         ("velocity", 300),
#         ("ensemble", "nve"),
#         ("cavity", 1),
#         ("run", 10),
#     ]
#     process = run_md(params, tmp_path)
#     assert 'cavity requires the second NEP potential to be a dipole model' in str(
#         process.stderr
#     )
# 
# 
# def test_cavity_missing_potential(tmp_path):
#     """Should raise an error when only a single NEP potential is specified."""
#     params = [
#         ("potential", f"{test_folder}/nep.txt"),
#         ("time_step", 1),
#         ("velocity", 300),
#         ("ensemble", "nve"),
#         ("cavity", 1),
#         ("run", 10),
#     ]
#     process = run_md(params, tmp_path)
#     assert 'cavity requires two potentials to be specified.' in str(process.stderr)
