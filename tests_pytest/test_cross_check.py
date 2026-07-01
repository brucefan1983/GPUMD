"""GPUNEP vs CPUNEP cross-check.

Compares GPUMD's own single-point evaluation against calorine's independently implemented CPU
reference (nep_cpu) -- the primary signal for GPU-kernel correctness issues, since it validates
against a wholly separate codebase rather than against GPUMD's own prior output. Uses
make_gpunep/CPUNEP explicitly rather than the shared `calculator` fixture, since the point here
is to compare them against each other directly.

This cross-check previously caught a real bug in calorine's nep_cpu charge-model force formula
(charge neutrality shifts predicted charges by -mean(charge), but the corresponding dE/dQ term
wasn't shifted to match, breaking the force chain rule for qNEP charge_mode 1/2) -- fixed in
calorine commit bf5cac1 by mirroring GPUMD's own zero_mean_D_real (nep_charge.cu:608-635).

It also surfaced a second, unrelated issue: GPUMD defaults to the PPPM method for the
reciprocal-space electrostatic contribution (doc/gpumd/input_parameters/kspace.rst), while
calorine's CPUNEP only implements Ewald (src/nepy/ewald_nep.cpp). Comparing PPPM against Ewald
is comparing two different approximations to the same physical quantity, not validating
agreement between two implementations of the same method -- so make_gpunep (conftest.py)
explicitly requests `kspace ewald` for qNEP models here, keeping this comparison
methodologically apples-to-apples. GPUMD's own PPPM-vs-Ewald agreement is validated separately
in test_kspace_consistency.py.
"""
import numpy as np
import pytest
from calorine.calculators import CPUNEP

from conftest import CROSS_CHECK_BEC_TOLERANCE, CROSS_CHECK_ENERGY_TOLERANCE, \
    CROSS_CHECK_FORCE_TOLERANCE, approx_tol, make_gpunep

pytestmark = [pytest.mark.fast, pytest.mark.cross_check]


def test_energy_cross_check(structure, model_type, model_path, gpumd_command):
    gpu_atoms = structure.copy()
    gpu_atoms.calc = make_gpunep(model_path, gpumd_command, model_type)
    gpu_energy = gpu_atoms.get_potential_energy()

    cpu_atoms = structure.copy()
    cpu_atoms.calc = CPUNEP(str(model_path))
    cpu_energy = cpu_atoms.get_potential_energy()

    assert gpu_energy == approx_tol(cpu_energy, CROSS_CHECK_ENERGY_TOLERANCE)


def test_forces_cross_check(structure, model_type, model_path, gpumd_command):
    gpu_atoms = structure.copy()
    gpu_atoms.calc = make_gpunep(model_path, gpumd_command, model_type)
    gpu_forces = gpu_atoms.get_forces()

    cpu_atoms = structure.copy()
    cpu_atoms.calc = CPUNEP(str(model_path))
    cpu_forces = cpu_atoms.get_forces()

    assert np.allclose(gpu_forces, cpu_forces, **CROSS_CHECK_FORCE_TOLERANCE)


def test_born_effective_charges_cross_check(structure, model_type, model_path, gpumd_command):
    if model_type == 'nep':
        pytest.skip('Born effective charges are only defined for qNEP (charge) models.')

    gpu_atoms = structure.copy()
    gpu_calc = make_gpunep(model_path, gpumd_command, model_type)
    gpu_atoms.calc = gpu_calc
    gpu_bec = gpu_calc.get_born_effective_charges(gpu_atoms)

    cpu_atoms = structure.copy()
    cpu_calc = CPUNEP(str(model_path))
    cpu_atoms.calc = cpu_calc
    cpu_bec = cpu_calc.get_born_effective_charges(cpu_atoms)

    assert np.allclose(gpu_bec, cpu_bec, **CROSS_CHECK_BEC_TOLERANCE)
