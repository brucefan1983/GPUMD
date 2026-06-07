"""Export second- and third-order force constants from a GPUMD NEP potential
into the kaldo ``gpumd_fc.npz`` format (readable by kaldo ``format='gpumd'``).

Pipeline:
    calorine CPUNEP forces -> phonopy (fc2) + phono3py (fc3) finite
    displacements -> kaldo-compatible output.

Dependencies:
    ase, calorine, phonopy, phono3py, numpy, scipy
"""
import numpy as np
from ase import Atoms
from calorine.calculators import CPUNEP
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phono3py import Phono3py
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# B.1 — Structure helpers and NEP force engine
# ---------------------------------------------------------------------------

def silicon_unitcell(a=5.43):
    """Return a 2-atom diamond-cubic Si primitive cell.

    Parameters
    ----------
    a : float
        Cubic lattice constant in Angstrom.

    Returns
    -------
    ase.Atoms
        2-atom Si primitive cell with PBC enabled.
    """
    cell = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    atoms = Atoms('Si2', cell=cell, pbc=True,
                  scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]])
    return atoms


def nep_forces(atoms, nep_path):
    """Return forces (eV/Å) on *atoms* from the NEP potential at *nep_path*.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to evaluate.
    nep_path : str
        Path to the NEP potential file.

    Returns
    -------
    numpy.ndarray, shape (n_atoms, 3)
        Forces in eV/Å.
    """
    a = atoms.copy()
    a.calc = CPUNEP(nep_path)
    return a.get_forces()


def relaxed_lattice_constant(nep_path, a_min=5.40, a_max=5.47):
    """Find the equilibrium cubic lattice constant for the Si NEP potential.

    Minimises the total energy of the 2-atom diamond-cubic primitive cell
    over the interval [*a_min*, *a_max*] using Brent's method.

    Parameters
    ----------
    nep_path : str
        Path to the NEP potential file.
    a_min, a_max : float
        Search bounds in Angstrom.

    Returns
    -------
    float
        Relaxed cubic lattice constant in Angstrom.
    """
    def _energy(a):
        atoms = silicon_unitcell(a=a)
        atoms.calc = CPUNEP(nep_path)
        return atoms.get_potential_energy()

    result = minimize_scalar(_energy, bounds=(a_min, a_max), method='bounded',
                             options={'xatol': 1e-5})
    return float(result.x)


# ---------------------------------------------------------------------------
# B.2 — fc2 (phonopy) and fc3 (phono3py)
# ---------------------------------------------------------------------------

def _to_phonopy_atoms(atoms):
    """Convert an ase.Atoms object to phonopy PhonopyAtoms."""
    return PhonopyAtoms(symbols=atoms.get_chemical_symbols(),
                        scaled_positions=atoms.get_scaled_positions(),
                        cell=atoms.cell[:])


def _forces_on_phonopy_supercell(phonopy_sc, nep_path):
    """Evaluate NEP forces on a PhonopyAtoms supercell.

    Parameters
    ----------
    phonopy_sc : phonopy.structure.atoms.PhonopyAtoms
        Displaced supercell from phonopy/phono3py.
    nep_path : str
        Path to the NEP potential file.

    Returns
    -------
    numpy.ndarray, shape (n_atoms, 3)
        Forces in eV/Å.
    """
    ase_atoms = Atoms(symbols=phonopy_sc.symbols,
                      scaled_positions=phonopy_sc.scaled_positions,
                      cell=phonopy_sc.cell, pbc=True)
    ase_atoms.calc = CPUNEP(nep_path)
    return ase_atoms.get_forces()


def compute_fc2(atoms, nep_path, supercell=(3, 3, 3), displacement=0.01):
    """Compute second-order force constants via phonopy finite displacements.

    Parameters
    ----------
    atoms : ase.Atoms
        Primitive unit cell (e.g. from ``silicon_unitcell``).
    nep_path : str
        Path to the NEP potential file.
    supercell : tuple of int
        Supercell repetitions along each lattice vector.
    displacement : float
        Displacement amplitude in Angstrom.

    Returns
    -------
    phonopy.Phonopy
        Phonopy object with force constants set.  Call
        ``ph.run_mesh([1, 1, 1])`` and ``ph.get_mesh_dict()`` for
        frequencies.
    """
    ph = Phonopy(_to_phonopy_atoms(atoms),
                 supercell_matrix=np.diag(supercell),
                 primitive_matrix='auto')
    ph.generate_displacements(distance=displacement)
    forces = [_forces_on_phonopy_supercell(sc, nep_path)
              for sc in ph.supercells_with_displacements if sc is not None]
    ph.forces = forces
    ph.produce_force_constants()
    return ph


def compute_fc3(atoms, nep_path, supercell=(3, 3, 3), displacement=0.03):
    """Compute third- and second-order force constants via phono3py.

    Uses a single supercell matrix for both fc2 and fc3 (the default
    phono3py workflow without a separate ``phonon_supercell_matrix``).
    When no separate phonon_supercell_matrix is set, phono3py 4.1.0
    uses ``ph3.forces`` (fc3 displacements) for both fc2 and fc3 via
    ``produce_fc3()`` + ``produce_fc2()``.

    Parameters
    ----------
    atoms : ase.Atoms
        Primitive unit cell.
    nep_path : str
        Path to the NEP potential file.
    supercell : tuple of int
        Supercell repetitions along each lattice vector.
    displacement : float
        Displacement amplitude in Angstrom.

    Returns
    -------
    phono3py.Phono3py
        Phono3py object with ``fc2`` and ``fc3`` attributes set.
    """
    ph3 = Phono3py(_to_phonopy_atoms(atoms),
                   supercell_matrix=np.diag(supercell),
                   primitive_matrix='auto')
    ph3.generate_displacements(distance=displacement)

    scs = ph3.supercells_with_displacements
    forces = [_forces_on_phonopy_supercell(sc, nep_path)
              for sc in scs if sc is not None]
    ph3.forces = forces
    ph3.produce_fc3()
    # When phonon_supercell_matrix is not set, produce_fc2 falls back to
    # ph3.dataset (same displacements as fc3), so forces are already set.
    ph3.produce_fc2()
    return ph3
