import os
from subprocess import CalledProcessError, run

import numpy as np
import pytest
from ase.io import read
from calorine.calculators import CPUNEP, GPUNEP
from ase.units import Bohr
from ase import Atoms
from cavity_calculator import TimeDependentCavityCalculator, CavityCalculator, DipoleCalculator

suite_path = 'gpumd/cavity'
repo_dir = f'{os.path.expanduser("~")}/repos/GPUMD/'
test_folder = f'{repo_dir}/tests/gpumd/cavity/self-consistent/'


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
def md_without_cavity(tmp_path, request):
    path = tmp_path / 'without_cavity'
    params = [
        ("potential", f"{test_folder}/nep.txt"),
        ("time_step", 1),
        ("velocity", 300),
        ("ensemble", "nve"),
        ("dump_position", 1),
        ("dump_force", 1),
        ("dump_thermo", 1),
        ("dump_velocity", 1),
        ("run", 2),
    ]
    run_md(params, path, repeat=request.param)
    return path


@pytest.fixture
def md(tmp_path, request):
    path = tmp_path / 'with_cavity'
    dipole_model = f"{test_folder}/nep-dipole.txt"
    params = [
        ("potential", f"{test_folder}/nep.txt"),
        ("time_step", 1),
        ("velocity", 300),
        ("ensemble", "nve"),
        ("cavity", (dipole_model, 1.0, 1.0, 1)),
        ("dump_position", 1),
        ("dump_force", 1),
        ("dump_thermo", 1),
        ("dump_velocity", 1),
        ("run", 2),
    ]
    run_md(params, path, repeat=request.param)
    return path, dipole_model


def _compute_dipole(structure: Atoms, dipole_model: str, charge: int):
    COM = structure.get_center_of_mass()
    calc = CPUNEP(dipole_model)
    structure.calc = calc
    cpu_dipole = structure.get_dipole_moment() * Bohr + charge * COM
    cpu_jacobian = calc.get_dipole_gradient(displacement=0.001, method='second order central difference', charge=charge/Bohr) * Bohr # CPUNEP corrects for center of mass, but not the unit conversion from au to ASE units. 
    return cpu_dipole, cpu_jacobian


@pytest.mark.parametrize('md', [1], indirect=True)
def test_cavity_self_consistent(md):
    """Ensure cavity writes dipoles and dipole jacobians that are consistent with the NEP executable"""
    md_path, dipole_model = md
    jacobian = np.loadtxt(f'{md_path}/jacobian.out')
    charge = 0
    # Read positions, and predict dipole with dipole model
    # TODO fails atm, could be due to change in when dipoles are computed
    # in run.cu.
    for gpu_dipole, conf in zip(jacobian[:, 1:4], read(f'{md_path}/movie.xyz', ':')):
        COM = conf.get_center_of_mass()
        conf.calc = CPUNEP(dipole_model)
        cpu_dipole = conf.get_dipole_moment() * Bohr + charge * COM
        print(cpu_dipole, gpu_dipole)
        assert np.allclose(cpu_dipole, gpu_dipole, atol=1e-1, rtol=1e-6)
    for gpu_jacobian, conf in zip(jacobian[:, 4:], read(f'{md_path}/movie.xyz', ':')):
        calc = CPUNEP(dipole_model)
        conf.calc = calc
        cpu_jacobian = calc.get_dipole_gradient(displacement=0.001, method='second order central difference', charge=charge/Bohr) * Bohr # CPUNEP corrects for center of mass, but not the unit conversion from au to ASE units. 
        gj = gpu_jacobian.reshape(len(conf), 3, 3)
        print(gj)
        assert np.allclose(cpu_jacobian, gj, atol=1e-1, rtol=1e-6)


@pytest.mark.parametrize('md', [1], indirect=True)
def test_cavity_time_dependent_cavity(md):
    """Ensure the cavity forces matches those from the TimeDependentCavityCalculator"""
    md_path, dipole_model = md
    cavity = np.loadtxt(f'{md_path}/cavity.out')
    charge = 0
    # Read positions, and predict dipole with dipole model
    # TODO fails atm, could be due to change in when dipoles are computed
    # in run.cu.
    initial_dipole, initial_jacobian = _compute_dipole(read(f'{test_folder}/model.xyz'), dipole_model, charge)
    coupling_strength = [0.0, 0.0, 1.0]
    cavity_calc = TimeDependentCavityCalculator(resonance_frequency=1.0,
                                                coupling_strength=coupling_strength,
                                                dipole_v=initial_dipole)
    for cav_properties, gpu_cavity, conf in zip(cavity[:, 1:8], cavity[:,8:], 
                                read(f'{md_path}/movie.xyz', ':')):
        dipole, jacobian = _compute_dipole(conf, dipole_model, charge)
        time, q, p, cavity_pot, cavity_kin, cos_integral, sin_integral = cav_properties
        # Step cavity calculator to current timestep
        cavity_calc._time = time

        changed = cavity_calc.step_if_time_changed(dipole)
        cpu_cavity = cavity_calc.cavity_force(dipole, jacobian)

        # Test cavity properties
        assert np.allclose(cavity_calc.canonical_position, q, atol=1e-4, rtol=1e-6)
        assert np.allclose(cavity_calc.canonical_momentum, p, atol=1e-4, rtol=1e-6)
        assert np.allclose(cavity_calc.cavity_potential_energy(dipole), cavity_pot, atol=1e-4, rtol=1e-6)
        assert np.allclose(cavity_calc.cavity_kinetic_energy(), cavity_kin, atol=1e-4, rtol=1e-6)

        # gpumd forces are in order [f_x1,..., f_xN, ..., f_z1, ..., f_zN]
        N = len(conf)
        gpu_cav = np.zeros((N, 3))
        gpu_cav[:, 0] = gpu_cavity[:N]
        gpu_cav[:, 1] = gpu_cavity[N:2*N]
        gpu_cav[:, 2] = gpu_cavity[2*N:]
        assert np.allclose(cpu_cavity, gpu_cav, atol=1e-4, rtol=1e-6)


# @pytest.mark.parametrize('md', [1], indirect=True)
# def test_cavity_total_forces(md):
#     """Ensure the total forces matches those from the CavityCalculator"""
#     md_path, dipole_model = md
#     cavity = np.loadtxt(f'{md_path}/cavity.out')
#     forces = np.loadtxt(f'{md_path}/force.out')
#     N = 27
#     forces = forces.reshape((-1, N, 3))
#     charge = 0.0
#     coupling_strength = [0.0, 0.0, 1.2]
#     # Read positions, and predict dipole with dipole model
#     # TODO fails atm, could be due to change in when dipoles are computed
#     # in run.cu.
#     dipole_model = f"{test_folder}/nep4_dipole.txt"
#     initial_atoms = read(f'{test_folder}/model.xyz')
#     nep_calc = CPUNEP(f"{test_folder}/nep.txt")
#     dipole_calc = DipoleCalculator(dipole_filename=dipole_model,
#                                    resonance_frequency=0.7,
#                                    coupling_strength=coupling_strength,
#                                    charge=charge,
#                                    gradient_mode='fd')
#     cavity_calc = CavityCalculator(nep_calc, dipole_calc)
#     # Set the initial dipole on the td cavity calculator
#     cavity_calc.calculate(initial_atoms)
#     print(cavity_calc.dipole_calc.td_cav.canonical_position)
#     for cav_properties, gpu_force, conf in zip(cavity[:, 1:6], forces, read(f'{md_path}/movie.xyz', ':')):
#         time, q, p, cavity_pot, cavity_kin = cav_properties
#         # Step cavity calculator to current timestep
#         cavity_calc.dipole_calc.td_cav._time = time
# 
#         print(cavity_calc.dipole_calc.td_cav._time,
#               cavity_calc.dipole_calc.td_cav.canonical_position, 
#               q)
#         # Compute the new forces
#         cavity_calc.calculate(conf, ['energy', 'forces'])
#         cpu_force = cavity_calc.results['forces'] 
# 
#         # gpumd forces are in order [f_x1,..., f_xN, ..., f_z1, ..., f_zN]
#         assert np.allclose(cpu_force, gpu_force, atol=1e-2, rtol=1e-6)


# @pytest.mark.parametrize('md', [1], indirect=True)
# def test_cavity_numeric(md):
#     """
#     Compare with a hardcoded value from the self consistent case above, in case
#     later updates breaks one of the implemenetations in either CPUNEP or GPUMD.
#     """
#     md_path, _ = md
#     dipole = np.loadtxt(f'{md_path}/dipole.out')
#     gpu_dipole = dipole[0, 1:]
#     cpu_dipole = [
#         4.79686227,
#         0.01536603,
#         0.07771485,
#     ]
#     assert np.allclose(cpu_dipole, gpu_dipole, atol=1e-3, rtol=1e-6)
# 
# 
# @pytest.mark.parametrize('md, md_without_dip', [[1, 1]], indirect=True)
# def test_cavity_does_not_change_forces_and_virials(md, md_without_dip):
#     """Ensure that all regular observables are unchanged"""
#     md_path, _ = md
#     md_without_dip_path = md_without_dip
#     files = ('thermo.out', 'force.out', 'velocity.out')
#     for file in files:
#         pol_content = np.loadtxt(os.path.join(md_path, file))
#         reg_content = np.loadtxt(os.path.join(md_without_dip_path, file))
#         assert np.allclose(pol_content, reg_content, atol=1e-12, rtol=1e-6)
# 
# 
# def test_cavity_invalid_potential(tmp_path):
#     """
#     Should raise an error when the second specified potential
#     is not a dipole model.
#     """
#     params = [
#         ("potential", f"{test_folder}/nep.txt"),
#         ("potential", f"{test_folder}/nep.txt"),
#         ("time_step", 1),
#         ("velocity", 300),
#         ("ensemble", "nve"),
#         ("cavity", 1),
#         ("run", 10),
#     ]
#     process = run_md(params, tmp_path)
#     assert 'cavity requires the second NEP potential to be a dipole model' in str(
#         process.stderr
#     )
# 
# 
# def test_cavity_missing_potential(tmp_path):
#     """Should raise an error when only a single NEP potential is specified."""
#     params = [
#         ("potential", f"{test_folder}/nep.txt"),
#         ("time_step", 1),
#         ("velocity", 300),
#         ("ensemble", "nve"),
#         ("cavity", 1),
#         ("run", 10),
#     ]
#     process = run_md(params, tmp_path)
#     assert 'cavity requires two potentials to be specified.' in str(process.stderr)
