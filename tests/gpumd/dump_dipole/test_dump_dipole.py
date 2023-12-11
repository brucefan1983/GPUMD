import os
from subprocess import run

import numpy as np
import pytest
from ase.io import read
from calorine.calculators import CPUNEP, GPUNEP

suite_path = 'gpumd/dump_dipole'

def test_dump_dipole_self_consistent(tmp_path):
    """Ensure dump_dipole writes dipoles that are consistent with the NEP executable"""
    repo_dir = f'{os.path.expanduser("~")}/repos/GPUMD/'
    test_folder = f'{repo_dir}/tests/gpumd/dump_dipole/self_consistent/'

    structure = read(f'{test_folder}/model.xyz')

    dipole_model = f"{test_folder}/nep4_dipole.txt"
    params = [
        ("potential", f"{test_folder}/nep.txt"),
        ("potential", dipole_model), 
        ("time_step", 1),
        ("velocity", 300),
        ("ensemble", "nve"),
        ("dump_dipole", 1),
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

    dipole = np.loadtxt(f'{tmp_path}/dipole.out')
    assert dipole.shape == (10, 4)
    assert np.allclose(dipole[:,0], np.arange(10))
    print(dipole)
    # Read positions, and predict dipole with dipole model  
    for gpu_dipole, conf in zip(dipole[:, 1:], read(f'{tmp_path}/movie.xyz', ':')):
        conf.calc = CPUNEP(dipole_model)
        cpu_dipole = conf.get_dipole_moment()
        # Note that the GPU dipole is dipole/atom; multiply by n_atoms
        print(cpu_dipole)
        gpu_dipole *= len(conf)
        print(gpu_dipole)

        assert np.allclose(cpu_dipole, gpu_dipole)


    # Numeric values from calorine
    # assert dipole.shape == (3,)
    # assert np.allclose(
    #     [-0.07468218, -0.03891397, -0.11160894], delta, atol=1e-12, rtol=1e-5
    # )



def test_dump_dipole_numeric():
    """Ensure dump_dipole writes known correct dipole values."""
    pass


def test_dump_dipole_invalid_potential():
    """Should raise an error when the second specified potential is not a dipole model."""
    pass


def test_dump_dipole_missing_potential():
    """Should raise an error when only a single NEP potential is specified."""
    pass


def test_dump_dipole_does_not_change_forces_and_virials():
    """Ensure that all regular observables are unchanged"""
    pass
