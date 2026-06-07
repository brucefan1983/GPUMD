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
