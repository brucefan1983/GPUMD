from typing import List
from subprocess import run
import numpy as np
import pandas as pd
from ase.io import read

test_folder = 'gpumd/carbon_average'


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
    for k, (frame, row) in enumerate(zip(read(f'{path}/observer.xyz', ':'),
                                             np.loadtxt(f'{path}/observer.out'),
                                         )):
        energy = frame.get_potential_energy()
        forces = frame.get_forces()
        data.append(dict(timestep=k,
                         energy_exyz=energy,
                         energy_thermo=row[2],
                         forces_exyz=forces.flatten(),
                         ))
    df = pd.DataFrame.from_dict(data)
 
    ref0 = read(f'{path}/ref-observer0.xyz', ':')[0]
    ref1 = read(f'{path}/ref-observer1.xyz', ':')[0]
    energy_ref0 = ref0.get_potential_energy()
    forces_ref0 = ref0.get_forces().flatten()
    energy_ref1 = ref1.get_potential_energy()
    forces_ref1 = ref1.get_forces().flatten()

    energy = np.vstack([energy_ref0, energy_ref1])
    forces = np.vstack([forces_ref0, forces_ref1])
    ref_df = pd.DataFrame.from_dict([{
            'energy': energy.mean(axis=0)[0],
            'forces': forces.mean(axis=0)
        }]) 
    return df, ref_df


def test_average_single_species(tmp_path):
    files = [
        'C_2022_NEP3.txt',
        'C_2022_NEP3_MODIFIED.txt',
        'model.xyz',
        'ref-observer0.xyz',
        'ref-observer1.xyz',
        'run.in'
    ]
    _copy_files(files, tmp_path)
    run('/home/elindgren/repos/GPUMD/src/gpumd', cwd=tmp_path, check=True)
    df, ref_df = _load_observer_files(tmp_path)
    print(df)
    print(ref_df)
    atol = 1e-8  # Anything smaller is considered ~0
    rtol = 1e-7
    assert np.all(np.isclose(
        a=df['energy_exyz'],
        b=df['energy_thermo'],
        atol=atol,
        rtol=rtol
    )), 'Energies should be consistent between exyz and thermo'

    # Compare to reference
    atol = 1e-4  # should be close to reference
    rtol = 1e-3
    assert np.all(np.isclose(
        a=df['energy_exyz'][0],
        b=ref_df['energy'],
        atol=atol,
        rtol=rtol
    )), 'Energies should match reference; did you compile with DDEBUG?'
    assert np.all(np.isclose(
        a=df['forces_exyz'][0],
        b=ref_df['forces'][0],
        atol=atol,
        rtol=rtol
    )), 'Forces should match reference; did you compile with DDEBUG?'


