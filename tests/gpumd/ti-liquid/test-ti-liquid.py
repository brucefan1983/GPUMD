import os
from subprocess import CalledProcessError, run

import numpy as np
import pytest
from ase.io import read
from calorine.calculators import CPUNEP, GPUNEP
import yaml

suite_path = 'gpumd/ti-liquid'
repo_dir = f'{os.path.expanduser("~")}/repos/GPUMD/'
test_folder = f'{repo_dir}/tests/gpumd/ti-liquid/self_consistent/'


def run_md(params, path):
    gpumd_command = f'{repo_dir}/src/gpumd'
    structure = read(f'{test_folder}/model.xyz')
    calc = GPUNEP(f"{test_folder}/nep_al.txt", command=gpumd_command)
    structure.calc = calc
    calc.set_directory(path)
    calc.run_custom_md(params, only_prepare=True)
    run('ls', cwd=path, check=True)
    return run(gpumd_command, cwd=path, capture_output=True)


@pytest.fixture
def frenkel_ladd(tmp_path, request):
    path = tmp_path / 'ti-liquid-results'
    params = [
        ("potential", f"{test_folder}/nep_al.txt"),
        ("time_step", 2),
        ("ensemble", "ti_liquid temp 2500 tswitch 4000 tequil 500 press 0 tperiod 100 sigmasqrd 5 p 1"),
        ("dump_position", 100),
        ("dump_thermo", 1),
        ("run", 10000),
    ]
    run_md(params, path)
  
    print(path)
    return path


@pytest.mark.parametrize('frenkel_ladd', [1], indirect=True)
def test_reference_free_energy(frenkel_ladd):
    """
    Compare with a hardcoded value from the self consistent case above, in case
    later updates breaks one of the implemenetations in either CPUNEP or GPUMD.
    """
    md_path = frenkel_ladd
    with open(str(md_path)+"/ti_liquid.yaml", 'r') as file:
        data = yaml.safe_load(file).get("E_UFmodel", None)

    gpu_EUF = data
    cpu_EUF = -2.623165161101518
    assert np.allclose(cpu_EUF, gpu_EUF, atol=1e-3, rtol=1e-6)
