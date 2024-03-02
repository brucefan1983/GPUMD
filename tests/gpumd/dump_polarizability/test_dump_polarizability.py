import os
from subprocess import run

import numpy as np
import pytest
from ase.io import read
from calorine.calculators import GPUNEP
from calorine.nep import get_polarizability

suite_path = 'gpumd/dump_polarizability'
repo_dir = f'{os.path.expanduser("~")}/repos/GPUMD/'
test_folder = f'{repo_dir}/tests/gpumd/dump_polarizability/self_consistent/'


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
def md_without_pol(tmp_path, request):
    path = tmp_path / 'without_pol'
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
    path = tmp_path / 'with_pol'
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
    run_md(params, path, repeat=request.param)
    return path, pol_model

@pytest.mark.parametrize('md', [1, 2], indirect=True)
def test_dump_polarizability_self_consistent(md):
    """
    Ensure dump_polarizability writes pols that are consistent with the NEP executable
    Parametrization corresponds to two different system sizes:
        1: 320 atoms
        6: 2560 atoms
    Ensures that the thread reduction for dump_polarizability is working for > 1024 atoms.
    """
    md_path, pol_model = md
    pol = np.loadtxt(f'{md_path}/polarizability.out')
    assert pol.shape == (10, 7)
    assert np.allclose(pol[:, 0], np.arange(10))
    # Read positions, and predict pol with pol model
    for gpu_pol, conf in zip(pol[:, 1:], read(f'{md_path}/movie.xyz', ':')):
        print(len(conf))
        cpu_pol_matrix = get_polarizability(conf, pol_model)  # 3x3
        cpu_pol = np.array(
            [
                cpu_pol_matrix[0, 0],  # xx
                cpu_pol_matrix[1, 1],  # yy
                cpu_pol_matrix[2, 2],  # zz
                cpu_pol_matrix[0, 1],  # xy
                cpu_pol_matrix[1, 2],  # yz
                cpu_pol_matrix[2, 0],  # zx
            ]
        )
        # We can expect the diagonal elements to match to a larger extent than the off-diagonals,
        # due to the off-diagonals typically being small.
        assert np.allclose(cpu_pol[:3], gpu_pol[:3], atol=1e-6, rtol=1e-4)
        assert np.allclose(cpu_pol[3:], gpu_pol[3:], atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize('md', [1], indirect=True)
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
        1675.418264,
        1703.6054623,
        1705.8788132,
        -11.01748065,
        0.9166246,
        -5.5417825,
    ]
    assert np.allclose(cpu_pol[:3], gpu_pol[:3], atol=1e-6, rtol=1e-2)
    assert np.allclose(cpu_pol[3:], gpu_pol[3:], atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize('md, md_without_pol', [[1, 1]], indirect=True)
def test_dump_polarizability_does_not_change_forces_and_virials(md, md_without_pol):
    """Ensure that all regular observables are unchanged"""
    md_path, _ = md
    md_without_pol_path = md_without_pol

    files = ('thermo.out', 'force.out', 'velocity.out')
    for file in files:
        pol_content = np.loadtxt(os.path.join(md_path, file))
        reg_content = np.loadtxt(os.path.join(md_without_pol_path, file))
        assert np.allclose(pol_content, reg_content, atol=1e-12, rtol=1e-6)


def test_dump_polarizability_invalid_potential(tmp_path):
    """Should raise an error when the second specified potential is not a pol model."""
    params = [
        ("potential", f"{test_folder}/nep.txt"),
        ("potential", f"{test_folder}/nep.txt"),
        ("time_step", 1),
        ("velocity", 300),
        ("ensemble", "nve"),
        ("dump_polarizability", 1),
        ("run", 10),
    ]
    process = run_md(params, tmp_path)
    assert (
        'dump_polarizability requires the second NEP potential to be a dipole model'
        in str(process.stderr)
    )


def test_dump_polarizability_missing_potential(tmp_path):
    """Should raise an error when only a single NEP potential is specified."""
    params = [
        ("potential", f"{test_folder}/nep.txt"),
        ("time_step", 1),
        ("velocity", 300),
        ("ensemble", "nve"),
        ("dump_polarizability", 1),
        ("run", 10),
    ]
    process = run_md(params, tmp_path)
    assert 'dump_polarizability requires two potentials to be specified.' in str(
        process.stderr
    )
