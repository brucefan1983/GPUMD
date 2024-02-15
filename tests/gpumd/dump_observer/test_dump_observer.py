from subprocess import run
from typing import List

import numpy as np
import pandas as pd
from ase.io import read


def _copy_files(files: List[str], test_folder: str, tmp_path: str):
    for file in files:
        with open(f'{test_folder}/{file}', 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        content = '\n'.join(lines)
        p = tmp_path / file
        p.write_text(content)


def _load_observer_files(path: str, mode: str):
    n_output_files = 1 if mode == 'average' else 2
    n_reference_files = 2 if mode == 'average' else 1
    data = []
    if n_output_files == 1:
        bundles = (read(f'{path}/observer.xyz', ':'), np.loadtxt(f'{path}/observer.out'))
    else:
        bundles = ()
        for i in range(n_output_files):
            bundles += (read(f'{path}/observer{i}.xyz', ':'), np.loadtxt(f'{path}/observer{i}.out'))
            # bundles += ([1*i,2,3], [1*i,2,3])
    for k, bundle in enumerate(zip(*bundles)):
        data_bundle = {
                'timestep': k,
        }
        # Group bundle into twos
        frame_bundle = [(bundle[i], bundle[i+1]) for i in range(0, len(bundle), 2)]
        for m, (frame, row) in enumerate(frame_bundle):
            energy = frame.get_potential_energy()
            forces = frame.get_forces()
            data_bundle[f'energy{m}_exyz'] = energy
            data_bundle[f'energy{m}_thermo'] = row[2]
            data_bundle[f'forces{m}_exyz'] = forces.flatten()
        data.append(data_bundle)
    df = pd.DataFrame.from_dict(data)
    
    ref_data = {}
    for i in range(n_reference_files):
        ref = read(f'{path}/reference_observer{i}.xyz', ':')[0]
        ref_data[f'energy_ref{i}']= ref.get_potential_energy()
        ref_data[f'forces_ref{i}']= ref.get_forces().flatten()
        forces_ref0 = ref.get_forces().flatten()
        ref_data['energy_ref0']

    ref_df = pd.DataFrame.from_dict([ref_data])
    return df, ref_df


def test_observe_single_species(tmp_path):
    test_folder = 'gpumd/dump_observer/carbon_observe'
    files = [
        'C_2022_NEP3.txt',
        'C_2022_NEP3_MODIFIED.txt',
        'model.xyz',
        'reference_observer0.xyz',
        'run.in'
    ]
    _copy_files(files, test_folder, tmp_path)
    run('/home/elindgren/repos/GPUMD/src/gpumd', cwd=tmp_path, check=True)
    df, ref_df = _load_observer_files(tmp_path, mode='observe')
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

    forces0 = np.concatenate(df['forces0_exyz'])
    forces1 = np.concatenate(df['forces1_exyz'])

    assert not np.any(np.isclose(
        a=forces0,
        b=forces1,
        atol=atol,
        rtol=rtol
    )), 'Forces should be different'

    # Compare to reference
    atol = 1e-4  # should be close to reference
    rtol = 1e-3
    assert np.all(np.isclose(
        a=df['energy0_exyz'][0],
        b=ref_df['energy_ref0'],
        atol=atol,
        rtol=rtol
    )), 'Energies should match reference; did you compile with DDEBUG?'
    assert np.all(np.isclose(
        a=df['forces0_exyz'][0],
        b=ref_df['forces_ref0'][0],
        atol=atol,
        rtol=rtol
    )), 'Forces should match reference; did you compile with DDEBUG'


def test_average_single_species(tmp_path):
    test_folder = 'gpumd/dump_observer/carbon_average'
    files = [
        'C_2022_NEP3.txt',
        'C_2022_NEP3_MODIFIED.txt',
        'model.xyz',
        'reference_observer0.xyz',
        'reference_observer1.xyz',
        'run.in'
    ]
    _copy_files(files, test_folder, tmp_path)
    run('/home/elindgren/repos/GPUMD/src/gpumd', cwd=tmp_path, check=True)
    df, ref_df = _load_observer_files(tmp_path, mode='average')
    print(df)
    print(ref_df)
    atol = 1e-8  # Anything smaller is considered ~0
    rtol = 1e-7
    assert np.all(np.isclose(
        a=df['energy0_exyz'],
        b=df['energy0_thermo'],
        atol=atol,
        rtol=rtol
    )), 'Energies should be consistent between exyz and thermo'

    # Compare to reference

    energy = np.vstack([ref_df['energy_ref0'], ref_df['energy_ref1']]).mean(axis=0)
    forces = np.vstack([ref_df['forces_ref0'], ref_df['forces_ref1']]).mean(axis=0)
    print("Energy")
    print(energy, ref_df['energy_ref0'][0], ref_df['energy_ref1'][0])
    print(df['energy0_thermo'][0],df['energy0_exyz'][0])
    print("Force")
    print(forces[0][0], ref_df['forces_ref0'][0][0], ref_df['forces_ref1'][0][0])
    print(df['forces0_exyz'][0][0], df['forces0_exyz'][1][0])
    energy = energy[0]
    atol = 1e-4  # should be close to reference
    rtol = 1e-3
    assert np.all(np.isclose(
        a=df['energy0_exyz'][0],
        b=energy,
        atol=atol,
        rtol=rtol
    )), 'Energies should match reference; did you compile with DDEBUG?'
    assert np.all(np.isclose(
        a=df['forces0_exyz'][0],
        b=forces[0],
        atol=atol,
        rtol=rtol
    )), 'Forces should match reference; did you compile with DDEBUG?'


def test_species_order(tmp_path):
    '''Trigger check in gpumd if atom species are in a different order'''
    test_folder = 'gpumd/dump_observer/PbTe_species'
    files = [
        'model.xyz',
        'run.in',
        'PbTe.txt',
        'PbTe_modified.txt'
    ]
    _copy_files(files, test_folder, tmp_path)
    out = run('/home/elindgren/repos/GPUMD/src/gpumd', cwd=tmp_path, check=False, capture_output=True)
    print(str(out.stderr))
    assert 'The atomic species and/or the order of the species are not consistent between the multiple potential' in str(out.stderr)
    assert out.returncode == 1
