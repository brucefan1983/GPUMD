import os
from subprocess import run

import numpy as np
import pytest
from ase.io import read
from calorine.calculators import GPUNEP
from calorine.nep import get_polarizability

suite_path = 'gpumd/dump_polarizability'

def test_dump_polarizability_self_consistent(tmp_path):
    """Ensure dump_polarizability writes pols that are consistent with the NEP executable"""
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
        ("run", 10),
    ]

    gpumd_command = f'{repo_dir}/src/gpumd > out'
    calc = GPUNEP(f"{test_folder}/nep.txt", command=gpumd_command)
    structure.calc= calc
    calc.set_directory(tmp_path)
    calc.run_custom_md(params, only_prepare=True)
    run('ls', cwd=tmp_path, check=True)
    run('/home/elindgren/repos/GPUMD/src/gpumd', cwd=tmp_path, check=True)

    pol = np.loadtxt(f'{tmp_path}/polarizability.out')
    assert pol.shape == (10, 7)
    assert np.allclose(pol[:, 0], np.arange(10))
    print(pol)
    # Read positions, and predict pol with pol model
    for gpu_pol, conf in zip(pol[:, 1:], read(f'{tmp_path}/movie.xyz', ':')):
        cpu_pol_matrix = get_polarizability(conf, pol_model) # 3x3
        cpu_pol = np.array([
                cpu_pol_matrix[0, 0],  # xx
                cpu_pol_matrix[1, 1],  # yy
                cpu_pol_matrix[2, 2],  # zz
                cpu_pol_matrix[0, 1],  # xy
                cpu_pol_matrix[1, 2],  # yz
                cpu_pol_matrix[2 ,0],  # zx
        ])
        print(gpu_pol)
        print(cpu_pol)
        print(gpu_pol / cpu_pol)

        assert np.allclose(cpu_pol, gpu_pol)


    # Numeric values from calorine
    # assert pol.shape == (3,)
    # assert np.allclose(
    #     [-0.07468218, -0.03891397, -0.11160894], delta, atol=1e-12, rtol=1e-5
    # )



def test_dump_polarizability_numeric():
    """Ensure dump_polarizability writes known correct pol values."""
    pass


def test_dump_polarizability_invalid_potential():
    """Should raise an error when the second specified potential is not a pol model."""
    pass


def test_dump_polarizability_missing_potential():
    """Should raise an error when only a single NEP potential is specified."""
    pass


def test_dump_polarizability_does_not_change_forces_and_virials():
    """Ensure that all regular observables are unchanged"""
    pass
