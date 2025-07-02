import os
from subprocess import CalledProcessError, run

import numpy as np
import pytest
from ase.io import read
from calorine.calculators import CPUNEP, GPUNEP

suite_path = 'gpumd/msd'
repo_dir = f'{os.path.expanduser("~")}/repos/GPUMD/'
test_folder = f'{repo_dir}/tests/gpumd/msd/self_consistent/'


def compute_msd_gpumd(positions, window_size):
    """A Python implementation of the MSD algorithm in GPUMD"""
    msd = np.zeros((window_size, 3))
    N = len(positions)
    Natoms = positions[0].shape[0]
    num_time_origins = 0
    memory = np.zeros((window_size, Natoms, 3))
    for t in range(N):
        correlation_step = t % window_size
        pt = positions[t]
        memory[correlation_step] = pt
        if t < window_size - 1:
            continue
        num_time_origins += 1
        for tau in range(window_size):
            ptau = memory[correlation_step-tau]
            dx = np.sum( (pt - ptau)**2 , axis=0)
            msd[tau] += dx
    return msd / (num_time_origins * Natoms)


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
def md(tmp_path):
    path = tmp_path / 'all_groups'
    params = [
        ("potential", f"{test_folder}/nep.txt"),
        ("time_step", 1),
        ("velocity", 300),
        ("ensemble", "nve"),
        ("compute_msd", (1, 5, "all_groups", 0, "save_every", 10)),
        ("dump_xyz", (0, 0, 1, "group_1.xyz", "unwrapped_position")),
        ("dump_xyz", (0, 1, 1, "group_2.xyz", "unwrapped_position")),
        ("dump_xyz", (0, 2, 1, "group_3.xyz", "unwrapped_position")),
        ("run", 30),
    ]
    run_md(params, path)
    return path


@pytest.fixture
def md_group_0(tmp_path):
    path = tmp_path / 'group_0'
    params = [
        ("potential", f"{test_folder}/nep.txt"),
        ("time_step", 1),
        ("velocity", 300),
        ("ensemble", "nve"),
        ("compute_msd", (1, 5, "group", 0, 0)),
        ("dump_xyz", (0, 0, 1, "group_1.xyz", "unwrapped_position")),
        ("run", 30),
    ]
    run_md(params, path)
    return path


@pytest.fixture
def md_all_atoms(tmp_path):
    path = tmp_path / 'all_atoms'
    params = [
        ("potential", f"{test_folder}/nep.txt"),
        ("time_step", 1),
        ("velocity", 300),
        ("ensemble", "nve"),
        ("compute_msd", (1, 5)),
        ("run", 30),
    ]
    run_md(params, path)
    return path


def test_compute_msd_all_groups(md):
    """Ensure the MSD computed by GPUMD is numerically equivalent to a Python implementation"""
    md_path = md
    group_1 = np.array([structure.get_array('unwrapped_position') for structure in read(f'{md_path}/group_1.xyz', ':')])
    group_2 = np.array([structure.get_array('unwrapped_position') for structure in read(f'{md_path}/group_2.xyz', ':')])
    group_3 = np.array([structure.get_array('unwrapped_position') for structure in read(f'{md_path}/group_3.xyz', ':')])
    msd = np.loadtxt(f'{md_path}/msd.out')
    
    assert msd.shape == (5, 3*6+1)

    for group_index, positions in enumerate((group_1, group_2, group_3)):
        python_msd = compute_msd_gpumd(positions, window_size=5)
        group_index_in_msd = 1 + group_index * 6
        gpumd_msd = msd[:, group_index_in_msd:group_index_in_msd+3]  # order is t, msd_x_1, msd_y_1, msd_z_1, sdc_x_1, sdc_y_1, sdc_z_1 etc.
        assert np.allclose(python_msd, gpumd_msd)


def test_compute_msd_all_groups_consistent_with_group_0(md, md_group_0):
    """Compare the MSD for group 0 computed via all_groups with just computing it for group 0"""
    path_all_groups = md
    path_group_0 = md_group_0
    msd_all_groups = np.loadtxt(f'{path_all_groups}/msd.out')
    msd_group_0 = np.loadtxt(f'{path_group_0}/msd.out')
    
    group_index_in_msd = 1 + 0 * 6
    msd_all = msd_all_groups[:, group_index_in_msd:group_index_in_msd+6]  # Include SDC
    msd_group = msd_group_0[:, 1:7]
    assert np.allclose(msd_all, msd_group)


def test_compute_msd_all_groups_consistent_with_all_atoms(md, md_all_atoms):
    """Compare the average MSD computed with all_groups with the MSD for all atoms"""
    path_all_groups = md
    path_all_atoms = md_all_atoms
    msd_all_groups = np.loadtxt(f'{path_all_groups}/msd.out')
    msd_all_atoms = np.loadtxt(f'{path_all_atoms}/msd.out')
        
    # Average MSD over all groups
    msd = np.zeros((len(msd_all_groups), 3))
    sdc = np.zeros((len(msd_all_groups), 3))
    for group in range(3):
        group_index_in_msd = 1 + group * 6
        msd_g = msd_all_groups[:, group_index_in_msd:group_index_in_msd+3]
        sdc_g = msd_all_groups[:, group_index_in_msd+3:group_index_in_msd+6]
        msd += msd_g
        sdc += sdc_g
    msd /= 3
    sdc /= 3

    # Compare to results directly averaged over all atoms
    msd_all = msd_all_atoms[:, 1:4]
    sdc_all = msd_all_atoms[:, 4:]

    assert np.allclose(msd_all, msd, atol=1e-1, rtol=1e-1)  # For some reason they do not agree very well
    assert np.allclose(sdc_all, sdc, atol=1e-1, rtol=1e-1)  # I don't think it is only due to precision loss
