import os
from subprocess import run

import numpy as np
import pytest
from ase.io import read
from calorine.calculators import GPUNEP
from calorine.nep import get_polarizability

suite_path = 'gpumd/dump_polarizability'

@pytest.fixture
def md_without_pol(tmp_path):
    repo_dir = f'{os.path.expanduser("~")}/repos/GPUMD/'
    test_folder = f'{repo_dir}/tests/gpumd/dump_polarizability/self_consistent/'
    structure = read(f'{test_folder}/model.xyz')

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

    gpumd_command = f'{repo_dir}/src/gpumd > out'
    calc = GPUNEP(f"{test_folder}/nep.txt", command=gpumd_command)
    structure.calc= calc
    calc.set_directory(tmp_path)
    calc.run_custom_md(params, only_prepare=True)
    run('ls', cwd=tmp_path, check=True)
    run('/home/elindgren/repos/GPUMD/src/gpumd', cwd=tmp_path, check=True)
    return tmp_path

@pytest.fixture
def md(tmp_path):
    repo_dir = f'{os.path.expanduser("~")}/repos/GPUMD/'
    test_folder = f'{repo_dir}/tests/gpumd/dump_polarizability/self_consistent/'
    structure = read(f'{test_folder}/model.xyz')
    pol_model = f"{test_folder}/nep_pol.txt"

    params = [
        ("potential", f"{test_folder}/nep.txt"),
        ("potential", pol_model),
        ("time_step", 1),
        ("velocity", 300),
        ("ensemble", "nve"),
        ("dump_polarizability", 1),
        ("dump_position", 1),
        ("dump_force", 1),
        ("dump_thermo", 1),
        ("dump_velocity", 1),
        ("run", 10),
    ]

    gpumd_command = f'{repo_dir}/src/gpumd > out'
    calc = GPUNEP(f"{test_folder}/nep.txt", command=gpumd_command)
    structure.calc= calc
    calc.set_directory(tmp_path)
    calc.run_custom_md(params, only_prepare=True)
    run('ls', cwd=tmp_path, check=True)
    run('/home/elindgren/repos/GPUMD/src/gpumd', cwd=tmp_path, check=True)
    return tmp_path, pol_model


def test_dump_polarizability_self_consistent(md):
    """Ensure dump_polarizability writes pols that are consistent with the NEP executable"""
    md_path, pol_model = md
    pol = np.loadtxt(f'{md_path}/polarizability.out')
    assert pol.shape == (10, 7)
    assert np.allclose(pol[:, 0], np.arange(10))
    # Read positions, and predict pol with pol model
    for gpu_pol, conf in zip(pol[:, 1:], read(f'{md_path}/movie.xyz', ':')):
        cpu_pol_matrix = get_polarizability(conf, pol_model) # 3x3
        cpu_pol = np.array([
                cpu_pol_matrix[0, 0],  # xx
                cpu_pol_matrix[1, 1],  # yy
                cpu_pol_matrix[2, 2],  # zz
                cpu_pol_matrix[0, 1],  # xy
                cpu_pol_matrix[1, 2],  # yz
                cpu_pol_matrix[2, 0],  # zx
        ])
        # We can expect the diagonal elements to match to a larger extent than the off-diagonals,
        # due to the off-diagonals typically being small.
        assert np.allclose(cpu_pol[:3], gpu_pol[:3], atol=1e-6, rtol=1e-6)
        assert np.allclose(cpu_pol[3:], gpu_pol[3:], atol=1e-6, rtol=1e-1)
    

def test_dump_polarizability_numeric(md):
    """
    Compare with a hardcoded value from the self consistent case above, in case
    later updates breaks one of the implemenetations in either CPUNEP or GPUMD.
    """
    md_path, _ = md
    pol = np.loadtxt(f'{md_path}/polarizability.out')
    assert pol.shape == (10, 7)
    assert np.allclose(pol[:, 0], np.arange(10))
    # Read positions, and predict pol with pol model
    gpu_pol = pol[0, 1:]
    cpu_pol = [
            1.55883440e+03,
            1.55883598e+03,
            1.55883172e+03,
            -6.52388999e-04,
            -1.83034756e-03,
            -1.09319071e-03
    ]
    assert np.allclose(cpu_pol[:3], gpu_pol[:3], atol=1e-6, rtol=1e-6)
    assert np.allclose(cpu_pol[3:], gpu_pol[3:], atol=1e-6, rtol=1e-1)


def test_dump_polarizability_does_not_change_forces_and_virials(md, md_without_pol):
    """Ensure that all regular observables are unchanged"""
    md_path, _ = md
    md_without_pol_path = md_without_pol

    files = ('thermo.out', 'force.out', 'velocity.out')
    for file in files:
        pol_content = np.loadtxt(os.path.join(md_path, file))
        reg_content = np.loadtxt(os.path.join(md_without_pol_path, file))
        assert np.allclose(pol_content, reg_content, atol=1e-12, rtol=1e-6)


def test_dump_polarizability_invalid_potential():
    """Should raise an error when the second specified potential is not a pol model."""
    pass


def test_dump_polarizability_missing_potential():
    """Should raise an error when only a single NEP potential is specified."""
    pass
