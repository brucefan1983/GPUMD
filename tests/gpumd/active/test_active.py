import pytest
from typing import List
from subprocess import run
from pathlib import Path
from ase.io.trajectory import PropertyNotImplementedError
import numpy as np
from ase.io.formats import UnknownFileTypeError
from ase.calculators.calculator import PropertyNotImplementedError
from ase.io import read


def _compute_uncertainty_cpp(forces: np.ndarray):
    """Python implementation of the C++ algorithm for
    computing maximum force uncertainty."""
    M = forces.shape[0]
    L = forces.shape[1]
    N = forces.shape[2]
    m = np.zeros((L, N*3))
    m_sq = np.zeros((L, N*3))
    for j in range(M):
        for i in range(N):
            fx = forces[j, :, i, 0]
            fy = forces[j, :, i, 1]
            fz = forces[j, :, i, 2]
            m[:, i + 0 * N] += fx/M
            m[:, i + 1 * N] += fy/M
            m[:, i + 2 * N] += fz/M
            m_sq[:, i + 0 * N] += fx*fx/M
            m_sq[:, i + 1 * N] += fy*fy/M
            m_sq[:, i + 2 * N] += fz*fz/M

    E = np.zeros((L, N*3))
    for i in range(3*N):
        E[:,i] = np.sqrt(m_sq[:,i] - m[:,i]*m[:,i])
    u = np.zeros(L)
    for l in range(L):
        for i in range(3*N):
            if E[l,i] > u[l]:
                u[l] = E[l,i]
    return u

def _compute_uncertainty_numpy(forces: np.ndarray):
    """Numpy implementation of what the C++ algorithm is supposed to do,
    i.e, compute maximum force uncertainty."""
    F = forces
    M = F.shape[0]
    L = F.shape[1]
    N = F.shape[2]
    F = F.reshape((M, L, 3*N))
    F_std = np.std(F, axis=0)

    U = np.max(F_std, axis=1)
    return U


def _copy_files(files: List[str], tmp_path: str):
    for file in files:
        with open(f'{suite_path}/{file}', 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        content = '\n'.join(lines)
        filename = file.split('/')[-1]
        print(filename)
        p = tmp_path / filename
        p.write_text(content)


def _load_active_files(path: str):
    unc_out = np.loadtxt(f'{path}/active.out')
    forces = {}
    uncertain_structures = {}
    for xyz in Path(path).glob('observer*.xyz'):
        observer = str(xyz.name).split('.')[0]
        structures = read(xyz, ':')
        uncertain_structures[observer] = structures
        concatenated = np.array([structure.get_forces() for structure in structures])
        forces[observer] = concatenated
    forces = np.array([data for data in forces.values()])
    
    try:
        active_structures = read(f'{path}/active.xyz', ':')
    except UnknownFileTypeError:
        active_structures = []

    return unc_out[:,1], forces, uncertain_structures, active_structures


suite_path = 'gpumd/active'

def test_active_no_threshold(tmp_path):
    """Run active learning with no threshold, such that all structures will be written."""
    test_folder = 'no_threshold/'
    files = [
        'model/nep_full.txt',
        'model/nep_split1.txt',
        'model/nep_split2.txt',
        'model/nep_split3.txt',
        'model/nep_split4.txt',
        'model/nep_split5.txt',
        'model.xyz',
        f'{test_folder}/run.in'
    ]
    _copy_files(files, tmp_path)
    run('/home/elindgren/repos/GPUMD/src/gpumd', cwd=tmp_path, check=True)
    uncertainties, forces, structures, active_structures = _load_active_files(tmp_path)
    
    # Check that uncertainties and forces match with numpy and Python implementations
    u_python = _compute_uncertainty_cpp(forces)
    u_numpy = _compute_uncertainty_numpy(forces)


    atol = 1e-8  # Anything smaller is considered ~0
    rtol = 1e-7
    
    assert np.allclose(u_python, u_numpy, atol=atol, rtol=rtol)
    assert np.allclose(u_python, uncertainties, atol=atol, rtol=rtol)
    
    # Check that structures have forces and velocities
    for observer_structures in structures.values():
        for structure in observer_structures:
            sf = structure.get_forces()
            assert not np.allclose(0, sf, atol=atol, rtol=rtol)

    # Check that uncertainties and structures.uncertainty matches
    structure_uncertainties = [structure.info['uncertainty'] for structure in active_structures]
    assert np.allclose(uncertainties, structure_uncertainties, atol=atol, rtol=rtol)

    # Compare forces and velocities to dump_observer
    # Should match with observer0, since that corresponds to the main potential.
    observer_structures = structures['observer0']
    for i in range(len(observer_structures)):
        of = observer_structures[i].get_forces()
        ov = observer_structures[i].get_velocities()
        af = active_structures[i].get_forces()
        av = active_structures[i].get_velocities()
        assert np.allclose(of, af, atol=atol, rtol=rtol)
        assert np.allclose(ov, av, atol=atol, rtol=rtol)


def test_active_no_threshold_every_tenth(tmp_path):
    """Run active learning with no threshold every tenth step"""
    test_folder = 'no_threshold/'
    files = [
        'model/nep_full.txt',
        'model/nep_split1.txt',
        'model/nep_split2.txt',
        'model/nep_split3.txt',
        'model/nep_split4.txt',
        'model/nep_split5.txt',
        'model.xyz',
        f'{test_folder}/run.in'
    ]
    _copy_files(files, tmp_path)
    run('/home/elindgren/repos/GPUMD/src/gpumd', cwd=tmp_path, check=True)
    uncertainties, forces, structures, active_structures = _load_active_files(tmp_path)
    
    # Check that uncertainties and forces match with numpy and Python implementations
    forces = forces[::10] # get every tenth force
    print(forces)

    u_python = _compute_uncertainty_cpp(forces)
    u_numpy = _compute_uncertainty_numpy(forces)


    atol = 1e-8  # Anything smaller is considered ~0
    rtol = 1e-7
    
    assert np.allclose(u_python, u_numpy, atol=atol, rtol=rtol)
    assert np.allclose(u_python, uncertainties, atol=atol, rtol=rtol)
    
    # Check that uncertainties and structures.uncertainty matches
    structure_uncertainties = [structure.info['uncertainty'] for structure in active_structures]
    assert np.allclose(uncertainties, structure_uncertainties, atol=atol, rtol=rtol)

    # Compare forces and velocities to dump_observer
    # Should match with observer0, since that corresponds to the main potential.
    observer_structures = structures['observer0']
    for i in range(len(active_structures)):
        of = observer_structures[i*10].get_forces()
        ov = observer_structures[i*10].get_velocities()
        af = active_structures[i].get_forces()
        av = active_structures[i].get_velocities()
        assert np.allclose(of, af, atol=atol, rtol=rtol)
        assert np.allclose(ov, av, atol=atol, rtol=rtol)


def test_active_low_threshold(tmp_path):
    """Run active learning with a low threshold, such that some structures will be written."""
    test_folder = 'low_threshold/'
    files = [
        'model/nep_full.txt',
        'model/nep_split1.txt',
        'model/nep_split2.txt',
        'model/nep_split3.txt',
        'model/nep_split4.txt',
        'model/nep_split5.txt',
        'model.xyz',
        f'{test_folder}/run.in'
    ]
    _copy_files(files, tmp_path)
    run('/home/elindgren/repos/GPUMD/src/gpumd', cwd=tmp_path, check=True)
    uncertainties, _, structures, active_structures = _load_active_files(tmp_path)
    
    # Make sure that there are equally many observer structures as uncertainties
    assert len(uncertainties) == len(structures['observer0'])

    # Make sure that the number of active structures are less than total steps
    assert len(uncertainties) > len(active_structures)

    # Make sure that the active structures have uncertainties exceeding the threshold
    structure_uncertainties = np.array([structure.info['uncertainty'] for structure in active_structures])
    assert np.all(structure_uncertainties > 0.015)


def test_active_high_threshold(tmp_path):
    """Run active learning with a very high threshold, such that no structures will be written."""
    test_folder = 'high_threshold/'
    files = [
        'model/nep_full.txt',
        'model/nep_split1.txt',
        'model/nep_split2.txt',
        'model/nep_split3.txt',
        'model/nep_split4.txt',
        'model/nep_split5.txt',
        'model.xyz',
        f'{test_folder}/run.in'
    ]
    _copy_files(files, tmp_path)
    run('/home/elindgren/repos/GPUMD/src/gpumd', cwd=tmp_path, check=True)
    uncertainties, _, structures, active_structures = _load_active_files(tmp_path)
    
    # Make sure that there are equally many observer structures as uncertainties
    assert len(uncertainties) == len(structures['observer0'])

    # Make sure that the number of active structures are less than total steps
    assert len(uncertainties) > len(active_structures)

    # Make sure that there are no structures written
    assert len(active_structures) == 0


def test_active_no_velocities_or_forces(tmp_path):
    """Otherwise the same as no_threshold."""
    # Assert that uncertainties and structures.uncertainty matches

    test_folder = 'no_forces_or_velocities/'
    files = [
        'model/nep_full.txt',
        'model/nep_split1.txt',
        'model/nep_split2.txt',
        'model/nep_split3.txt',
        'model/nep_split4.txt',
        'model/nep_split5.txt',
        'model.xyz',
        f'{test_folder}/run.in'
    ]
    _copy_files(files, tmp_path)
    run('/home/elindgren/repos/GPUMD/src/gpumd', cwd=tmp_path, check=True)
    uncertainties, forces, structures, active_structures = _load_active_files(tmp_path)
    
    # Check that uncertainties and forces match with numpy and Python implementations
    u_python = _compute_uncertainty_cpp(forces)
    u_numpy = _compute_uncertainty_numpy(forces)


    atol = 1e-8  # Anything smaller is considered ~0
    rtol = 1e-7
    
    assert np.allclose(u_python, u_numpy, atol=atol, rtol=rtol)
    assert np.allclose(u_python, uncertainties, atol=atol, rtol=rtol)
    
    # Check that structures have forces and velocities
    for observer_structures in structures.values():
        for structure in observer_structures:
            sf = structure.get_forces()
            assert not np.allclose(0, sf, atol=atol, rtol=rtol)

    # Check that uncertainties and structures.uncertainty matches
    structure_uncertainties = [structure.info['uncertainty'] for structure in active_structures]
    assert np.allclose(uncertainties, structure_uncertainties, atol=atol, rtol=rtol)

    # Make sure that active structures have no velocities or forces
    observer_structures = structures['observer0']
    for i in range(len(active_structures)):
        with pytest.raises(PropertyNotImplementedError) as e:
            active_structures[i].get_forces()
            assert 'The property "forces" is not available.' in str(e)
        with pytest.raises(PropertyNotImplementedError) as e:
            active_structures[i].get_velocities()
            assert 'The property "velocities" is not available.' in str(e)


def test_active_dump_observer_average(tmp_path_factory):
    """Active learning should not be affected by dump_observer running in average mode"""
    # First run with no_threshold to get expected uncertainties etc.
    test_folder = 'no_threshold/'
    files = [
        'model/nep_full.txt',
        'model/nep_split1.txt',
        'model/nep_split2.txt',
        'model/nep_split3.txt',
        'model/nep_split4.txt',
        'model/nep_split5.txt',
        'model.xyz',
        f'{test_folder}/run.in'
    ]
    tmp_path = tmp_path_factory.mktemp('no_threshold')
    _copy_files(files, tmp_path)
    run('/home/elindgren/repos/GPUMD/src/gpumd', cwd=tmp_path, check=True)
    expected_uncertainties, _, _, _ = _load_active_files(tmp_path)
    
    test_folder = 'dump_observer_average/'
    files = [
        'model/nep_full.txt',
        'model/nep_split1.txt',
        'model/nep_split2.txt',
        'model/nep_split3.txt',
        'model/nep_split4.txt',
        'model/nep_split5.txt',
        'model.xyz',
        f'{test_folder}/run.in'
    ]
    tmp_path = tmp_path_factory.mktemp('average')
    _copy_files(files, tmp_path)
    run('/home/elindgren/repos/GPUMD/src/gpumd', cwd=tmp_path, check=True)
    uncertainties, _, _, _ = _load_active_files(tmp_path)

    atol = 1e-8  # Anything smaller is considered ~0
    rtol = 1e-7
    assert np.allclose(uncertainties, expected_uncertainties, atol=atol, rtol=rtol)
