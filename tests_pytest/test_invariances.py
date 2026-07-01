"""Translation / rotation / permutation / PBC invariance checks.

Parametrized over `structure` x `calculator` (see conftest.py) so both GPUNEP and CPUNEP get
the same checks.
"""
import numpy as np
import pytest

from conftest import approx_tol, transform_tolerance

RNG_SEED = 7

pytestmark = pytest.mark.fast


def _rotation_matrix(angle_deg, axis):
    """Rodrigues' rotation formula; used instead of ase.Atoms.rotate() so the exact matrix is
    available to also transform the expected forces for comparison."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(angle_deg)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def test_translation_invariance(structure, calculator, calculator_kind):
    atoms = structure.copy()
    atoms.calc = calculator
    energy_before = atoms.get_potential_energy()

    rng = np.random.default_rng(RNG_SEED)
    shifted = structure.copy()
    shifted.translate(rng.uniform(-3.0, 3.0, size=3))
    if any(shifted.pbc):
        shifted.wrap()
    shifted.calc = calculator
    energy_after = shifted.get_potential_energy()

    energy_tol = transform_tolerance('energy', calculator_kind)
    assert energy_after == approx_tol(energy_before, energy_tol)


def test_rotation_invariance(structure, calculator, calculator_kind):
    atoms = structure.copy()
    atoms.calc = calculator
    energy_before = atoms.get_potential_energy()
    forces_before = atoms.get_forces()

    angle = 37.0  # degrees; arbitrary, fixed for reproducibility
    axis = np.array([0.3, 0.5, 0.8113883008])  # arbitrary fixed axis, normalized in the helper
    rot = _rotation_matrix(angle, axis)

    rotated = structure.copy()
    center = rotated.get_center_of_mass()
    rotated.set_positions((rotated.get_positions() - center) @ rot.T + center)
    if any(rotated.pbc):
        rotated.set_cell(rotated.get_cell() @ rot.T, scale_atoms=False)
        rotated.wrap()
    rotated.calc = calculator
    energy_after = rotated.get_potential_energy()
    forces_after = rotated.get_forces()

    energy_tol = transform_tolerance('energy', calculator_kind)
    force_tol = transform_tolerance('force', calculator_kind)
    assert energy_after == approx_tol(energy_before, energy_tol)
    expected_forces_after = forces_before @ rot.T
    assert np.allclose(forces_after, expected_forces_after, **force_tol)


def test_permutation_invariance(structure, calculator, calculator_kind):
    atoms = structure.copy()
    atoms.calc = calculator
    energy_before = atoms.get_potential_energy()
    forces_before = atoms.get_forces()

    symbols = np.array(atoms.get_chemical_symbols())
    # Cyclic shift within each species group rather than a random permutation: guarantees a
    # non-identity relabeling even for a 2-atom group (a random permutation of 2 elements has a
    # 50% chance of landing back on the identity for any given fixed seed).
    permutation = np.arange(len(atoms))
    for symbol in set(symbols):
        idx = np.where(symbols == symbol)[0]
        if len(idx) > 1:
            permutation[idx] = np.roll(idx, 1)
    assert not np.array_equal(permutation, np.arange(len(atoms))), (
        'test structure has no species with >1 atom; permutation invariance check is a no-op')

    permuted = structure.copy()[permutation]
    permuted.calc = calculator
    energy_after = permuted.get_potential_energy()
    forces_after = permuted.get_forces()

    energy_tol = transform_tolerance('energy', calculator_kind)
    force_tol = transform_tolerance('force', calculator_kind)
    assert energy_after == approx_tol(energy_before, energy_tol)
    assert np.allclose(forces_after, forces_before[permutation], **force_tol)


def test_lattice_vector_shift_invariance(structure, structure_name, calculator, calculator_kind):
    if not any(structure.pbc):
        pytest.skip(f'{structure_name} is not periodic; lattice-vector shift is N/A.')

    atoms = structure.copy()
    atoms.calc = calculator
    energy_before = atoms.get_potential_energy()

    shifted = structure.copy()
    shifted.translate(shifted.get_cell()[0])  # shift by the 'a' lattice vector
    shifted.wrap()
    shifted.calc = calculator
    energy_after = shifted.get_potential_energy()

    energy_tol = transform_tolerance('energy', calculator_kind)
    assert energy_after == approx_tol(energy_before, energy_tol)
