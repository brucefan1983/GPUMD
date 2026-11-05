"""Central finite-difference dE/dr check against GPUNEP's own analytic forces.

GPUNEP only, deliberately: the point of this file is to validate GPUMD's (this repo's) own
force/energy self-consistency, which is the suite's sole numeric-validation strategy for
NEP-family architectures.

Unlike test_invariances.py, this file intentionally does NOT request `kspace ewald` for qNEP
models (see conftest.py's make_gpunep) -- the residual here is not primarily a
reciprocal-space-method artifact (checked directly), so there is no mismatch to correct, and
testing GPUMD's default (PPPM) self-consistency is exactly the point.
"""
import numpy as np
import pytest
from calorine.calculators import GPUNEP

from conftest import GPU_FINITE_DIFFERENCE_FORCE_TOLERANCE, approx_tol

# Central-difference truncation error scales as DISPLACEMENT**2, which would favor a small
# step. But GPUNEP's GPU-computed energy has its own fp32-scale precision floor (measured
# non-monotonic behavior when sweeping DISPLACEMENT from 1e-2 down to 1e-7 -- smaller isn't
# always better), so DISPLACEMENT is chosen larger than pure truncation-error minimization would
# pick, favoring a step where the energy difference is comfortably above that floor.
DISPLACEMENT = 1e-2  # Angstrom
N_ATOMS_TO_CHECK = 2

pytestmark = pytest.mark.fast


def _central_difference_force(atoms, model_path, gpumd_command, atom_index, direction):
    plus = atoms.copy()
    positions = plus.get_positions()
    positions[atom_index, direction] += DISPLACEMENT
    plus.set_positions(positions)
    plus.calc = GPUNEP(str(model_path), command=gpumd_command)
    energy_plus = plus.get_potential_energy()

    minus = atoms.copy()
    positions = minus.get_positions()
    positions[atom_index, direction] -= DISPLACEMENT
    minus.set_positions(positions)
    minus.calc = GPUNEP(str(model_path), command=gpumd_command)
    energy_minus = minus.get_potential_energy()

    return -(energy_plus - energy_minus) / (2 * DISPLACEMENT)


def test_finite_difference_forces(structure, model_path, gpumd_command):
    atoms = structure.copy()
    atoms.calc = GPUNEP(str(model_path), command=gpumd_command)
    analytic_forces = atoms.get_forces()

    n_check = min(N_ATOMS_TO_CHECK, len(atoms))
    atom_indices = np.linspace(0, len(atoms) - 1, n_check, dtype=int)

    for atom_index in atom_indices:
        for direction in range(3):
            numeric_force = _central_difference_force(
                atoms, model_path, gpumd_command, atom_index, direction)
            assert numeric_force == approx_tol(
                analytic_forces[atom_index, direction], GPU_FINITE_DIFFERENCE_FORCE_TOLERANCE)
