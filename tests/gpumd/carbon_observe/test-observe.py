from typing import List
from subprocess import run
import numpy as np
import pandas as pd
from ase.io import read

test_folder = 'gpumd/carbon_observe'


def _copy_files(files: List[str], tmp_path: str):
    for file in files:
        with open(f'{test_folder}/{file}', 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        content = '\n'.join(lines)
        p = tmp_path / file
        p.write_text(content)


def _load_observer_files(path: str):
    data = []
    for k, (frame0, frame1, row0, row1) in enumerate(zip(read(f'{path}/observer0.xyz', ':'),
                                                         read(f'{path}/observer1.xyz', ':'),
                                                         np.loadtxt(f'{path}/observer0.out'),
                                                         np.loadtxt(f'{path}/observer1.out'),
                                                         )):
        energy0 = frame0.get_potential_energy()
        energy1 = frame1.get_potential_energy()
        forces0 = frame0.get_forces()
        forces1 = frame1.get_forces()
        data.append(dict(timestep=k,
                         energy0_exyz=energy0,
                         energy1_exyz=energy1,
                         energy0_thermo=row0[2],
                         energy1_thermo=row1[2],
                         forces0_exyz=forces0.flatten(),
                         forces1_exyz=forces1.flatten(),
                         ))
    df = pd.DataFrame.from_dict(data)
    ref = read(f'{path}/reference_observer.xyz', ':')[0]
    energy_ref = ref.get_potential_energy()
    forces_ref = ref.get_forces()
    ref_df = pd.DataFrame.from_dict([{
            'energy': energy_ref,
            'forces': forces_ref.flatten()
        }]) 
    return df, ref_df


def test_observe_single_species(tmp_path):
    files = [
        'C_2022_NEP3.txt',
        'C_2022_NEP3_MODIFIED.txt',
        'model.xyz',
        'reference_observer.xyz',
        'run.in'
    ]
    _copy_files(files, tmp_path)
    run('/home/elindgren/repos/GPUMD/src/gpumd', cwd=tmp_path, check=True)
    df, ref_df = _load_observer_files(tmp_path)
    atol = 1e-8  # Anything smaller is considered ~0
    rtol = 1e-7
    assert not np.any(np.isclose(
        a=df['energy0_exyz'],
        b=df['energy1_exyz'],
        atol=atol,
        rtol=rtol
    )), 'Energies should be different'
    assert not np.any(np.isclose(
        a=df['energy0_thermo'],
        b=df['energy1_thermo'],
        atol=atol,
        rtol=rtol
    )), 'Energies should be different'
    assert np.all(np.isclose(
        a=df['energy0_exyz'],
        b=df['energy0_thermo'],
        atol=atol,
        rtol=rtol
    )), 'Energies should be consistent between exyz and thermo'
    assert np.all(np.isclose(
        a=df['energy1_exyz'],
        b=df['energy1_thermo'],
        atol=atol,
        rtol=rtol
    )), 'Energies should be consistent between exyz and thermo'
    assert np.all(np.isclose(
        a=df['energy0_exyz'][0],
        b=ref_df['energy'],
        atol=atol,
        rtol=rtol
    )), 'Energies should match reference'

    forces0 = np.concatenate(df['forces0_exyz'])
    forces1 = np.concatenate(df['forces1_exyz'])

    indices = np.argwhere(np.isclose(
        a=forces0,
        b=forces1,
        atol=atol,
        rtol=rtol
    ))
    print(forces0[indices])
    print(forces1[indices])
    assert not np.any(np.isclose(
        a=forces0,
        b=forces1,
        atol=atol,
        rtol=rtol
    )), 'Forces should be different'
    print(df['forces0_exyz'][0])
    print(ref_df['forces'][0])
    indices = np.argwhere(np.invert(np.isclose(
        a=df['forces0_exyz'][0],
        b=ref_df['forces'][0],
        atol=atol,
        rtol=rtol
    )))
    print(indices)
    print(df['forces0_exyz'][0][indices])
    print(ref_df['forces'][0][indices])
    assert np.all(np.isclose(
        a=df['forces0_exyz'][0],
        b=ref_df['forces'][0],
        atol=atol,
        rtol=rtol
    )), 'Forces should match reference'


def test_observe_change_order(tmp_path: str):
    """Using the same potential but changing the order of species
    should yield different energies and forces.
    """
    pass
