"""Export second- and third-order force constants from a GPUMD NEP potential
into the kaldo ``gpumd_fc.npz`` format (readable by kaldo ``format='gpumd'``).

Pipeline:
    calorine CPUNEP forces -> phonopy (fc2) + phono3py (fc3) finite
    displacements -> kaldo-compatible output.

Dependencies:
    ase, calorine, phonopy, phono3py, numpy, scipy, sparse, kaldo (kaldo only
    for the C-grid replica ordering used by ``to_kaldo_layout`` / the writer).
"""
import argparse
import hashlib

import numpy as np
import sparse
from ase import Atoms
from calorine.calculators import CPUNEP
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phono3py import Phono3py
from scipy.optimize import minimize_scalar as _minimize_scalar


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

    result = _minimize_scalar(_energy, bounds=(a_min, a_max), method='bounded',
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
              for sc in ph.supercells_with_displacements]
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
    # cutoff_pair_distance is intentionally not used: every displaced supercell
    # is force-evaluated, so phono3py's force ordering stays 1:1 with the
    # displacement dataset. (With a cutoff, phono3py emits None placeholders and
    # the forces list must keep them — unsupported here by design.)
    ph3.generate_displacements(distance=displacement)

    scs = ph3.supercells_with_displacements
    forces = [_forces_on_phonopy_supercell(sc, nep_path) for sc in scs]
    ph3.forces = forces
    ph3.produce_fc3()
    # When phonon_supercell_matrix is not set, produce_fc2 falls back to
    # ph3.dataset (same displacements as fc3), so forces are already set.
    ph3.produce_fc2()
    return ph3


# ---------------------------------------------------------------------------
# B.3 — remap phono3py force constants into kaldo's canonical C-grid layout
# ---------------------------------------------------------------------------

def _supercell_atom_to_replica_unit(phono_supercell, primitive, atoms, supercell):
    """Map each phono3py supercell atom to (kaldo C-grid replica id, unit atom).

    The replica id is the row of ``Grid(supercell, 'C').grid(is_wrapping=False)``
    whose integer cell index equals the lattice image the supercell atom sits in.
    The unit-atom index is ``u2u_map[s2u_map[a]]`` (0..n_uc-1).

    Parameters
    ----------
    phono_supercell : phonopy.structure.cells.Supercell
        The phono3py supercell (``ph3.supercell``).
    primitive : phonopy.structure.cells.Primitive
        The phono3py primitive cell (``ph3.primitive``); used for ``p2s_map``.
    atoms : ase.Atoms
        The primitive unit cell whose ``cell``/positions define the C grid.
    supercell : tuple of int
        Supercell repetitions; defines the Grid shape.

    Returns
    -------
    rep_of : numpy.ndarray, shape (n_satom,), int
        Kaldo C-grid replica id for each supercell atom.
    unit_of : numpy.ndarray, shape (n_satom,), int
        Unit-atom index (0..n_uc-1) for each supercell atom.
    compact_row_of : numpy.ndarray, shape (n_satom,), int
        For each supercell atom, the compact-fc row index of its primitive
        image (i.e. position of ``s2u_map[a]`` within ``primitive.p2s_map``).
    """
    from kaldo.grid import Grid

    grid_cells = Grid(tuple(int(s) for s in supercell), order='C').grid(is_wrapping=False)
    cell = np.array(atoms.cell, dtype=np.float64)
    inv_cell = np.linalg.inv(cell)
    prim_pos = atoms.get_positions()
    sc_dim = np.array([int(s) for s in supercell])

    s2u = np.array(phono_supercell.s2u_map)
    u2u = phono_supercell.u2u_map               # dict: supercell index -> 0..n_uc-1
    p2s = list(np.array(primitive.p2s_map))     # supercell indices of the n_uc prim atoms
    sc_cart = np.array(phono_supercell.scaled_positions) @ np.array(phono_supercell.cell)

    n_satom = len(sc_cart)
    rep_of = np.empty(n_satom, dtype=int)
    unit_of = np.empty(n_satom, dtype=int)
    compact_row_of = np.empty(n_satom, dtype=int)
    for a in range(n_satom):
        prim_satom = int(s2u[a])
        j = int(u2u[prim_satom])
        unit_of[a] = j
        compact_row_of[a] = p2s.index(prim_satom)
        offset = sc_cart[a] - prim_pos[j]
        cell_idx = np.rint(offset @ inv_cell).astype(int) % sc_dim
        match = np.where((grid_cells == cell_idx).all(axis=1))[0]
        if match.size == 0:
            raise ValueError(f"Supercell atom {a} cell index {cell_idx} not on the C grid.")
        rep_of[a] = int(match[0])
    return rep_of, unit_of, compact_row_of


def to_kaldo_layout(ph3, atoms, supercell, third_supercell, threshold=0.0):
    """Remap phono3py fc2/fc3 into kaldo's canonical C-grid layout.

    Works directly from phono3py structure data (no hiphive). Handles both the
    compact (axis 0 == n_uc) and full (axis 0 == n_satom) phono3py fc storage.

    Parameters
    ----------
    ph3 : phono3py.Phono3py
        Object with ``fc2``, ``fc3``, ``supercell``, ``primitive`` set.
    atoms : ase.Atoms
        Primitive unit cell (defines the C grid and atom ordering).
    supercell : tuple of int
        Supercell used for fc2 (the C grid for the second-order replicas).
    third_supercell : tuple of int
        Supercell used for fc3 (the C grid for the third-order replicas).
    threshold : float
        fc3 entries with ``|value| <= threshold`` are dropped from the COO.

    Returns
    -------
    fc2 : numpy.ndarray, float64, shape (1, n_uc, 3, n_rep2, n_uc, 3)
        Second-order force constants in eV/Angstrom^2.
    fc3 : sparse.COO, shape (n_uc*3, n_rep3*n_uc*3, n_rep3*n_uc*3)
        Third-order force constants in eV/Angstrom^3.
    """
    n_uc = len(atoms)

    # --- fc2 ---
    rep2, unit2, crow2 = _supercell_atom_to_replica_unit(
        ph3.supercell, ph3.primitive, atoms, supercell)
    fc2_arr = np.array(ph3.fc2, dtype=np.float64)
    n_satom2 = len(rep2)
    n_rep2 = int(np.prod(supercell))
    # Compact-vs-full detection assumes n_uc != n_satom, true for any supercell
    # larger than (1, 1, 1); a (1, 1, 1) supercell would make both ambiguous.
    is_compact2 = (fc2_arr.shape[0] == n_uc)
    if not is_compact2 and fc2_arr.shape[0] != n_satom2:
        raise ValueError(f"Unexpected fc2 axis-0 length {fc2_arr.shape[0]} "
                         f"(expected {n_uc} compact or {n_satom2} full).")

    fc2 = np.zeros((1, n_uc, 3, n_rep2, n_uc, 3), dtype=np.float64)
    for b in range(n_satom2):
        j = unit2[b]
        r = rep2[b]
        for a in range(n_satom2):
            i = unit2[a]
            if rep2[a] != 0:
                continue                        # first index must be the reference cell
            if is_compact2:
                fc2[0, i, :, r, j, :] = fc2_arr[crow2[a], b]
            else:
                fc2[0, i, :, r, j, :] = fc2_arr[a, b]

    # --- fc3 ---
    rep3, unit3, crow3 = _supercell_atom_to_replica_unit(
        ph3.supercell, ph3.primitive, atoms, third_supercell)
    fc3_arr = np.array(ph3.fc3, dtype=np.float64)
    n_satom3 = len(rep3)
    n_rep3 = int(np.prod(third_supercell))
    is_compact3 = (fc3_arr.shape[0] == n_uc)
    if not is_compact3 and fc3_arr.shape[0] != n_satom3:
        raise ValueError(f"Unexpected fc3 axis-0 length {fc3_arr.shape[0]} "
                         f"(expected {n_uc} compact or {n_satom3} full).")

    shape3 = (n_uc * 3, n_rep3 * n_uc * 3, n_rep3 * n_uc * 3)
    rows, cols1, cols2, vals = [], [], [], []
    ref_atoms = np.where(rep3 == 0)[0]          # supercell atoms in the reference cell
    for a in ref_atoms:
        i = unit3[a]
        a_src = crow3[a] if is_compact3 else a
        for b in range(n_satom3):
            j, l2 = unit3[b], rep3[b]
            for c in range(n_satom3):
                k, l3 = unit3[c], rep3[c]
                block = fc3_arr[a_src, b, c]    # (3, 3, 3)
                nz = np.argwhere(np.abs(block) > threshold)
                for (al, be, ga) in nz:
                    rows.append(i * 3 + al)
                    cols1.append((l2 * n_uc + j) * 3 + be)
                    cols2.append((l3 * n_uc + k) * 3 + ga)
                    vals.append(block[al, be, ga])
    coords = np.array([rows, cols1, cols2], dtype=np.int64)
    if coords.size == 0:
        coords = np.zeros((3, 0), dtype=np.int64)
    fc3 = sparse.COO(coords, np.array(vals, dtype=np.float64), shape=shape3)
    return fc2, fc3


# ---------------------------------------------------------------------------
# B.4 — acoustic sum rule, npz writer, and CLI
# ---------------------------------------------------------------------------

def apply_acoustic_sum_rule(fc2):
    """Enforce per-reference-atom translational invariance on fc2.

    Mirrors kaldo's ``acoustic_sum_rule``: for each reference atom ``i``,
    subtract the sum over all (replica, atom) pairs from the on-site block.

    Parameters
    ----------
    fc2 : numpy.ndarray, shape (1, n_uc, 3, n_rep, n_uc, 3)
        Second-order force constants.

    Returns
    -------
    numpy.ndarray
        A copy of ``fc2`` with the acoustic sum rule applied.
    """
    fc2 = np.array(fc2, dtype=np.float64, copy=True)
    n_uc = fc2.shape[1]
    for i in range(n_uc):
        off_diag_sum = np.sum(fc2[0, i, :, :, :, :], axis=(-2, -3))   # (3, 3)
        fc2[0, i, :, 0, i, :] -= off_diag_sum
    return fc2


def _nep_sha256(nep_path):
    """Return the full hex sha256 digest of the NEP potential file."""
    h = hashlib.sha256()
    with open(nep_path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_gpumd_fc(path, atoms, supercell, third_supercell, fc2, fc3,
                   nep_path, acoustic_sum_applied):
    """Write a ``gpumd_fc.npz`` archive readable by kaldo ``format='gpumd'``.

    Keys/dtypes match exactly what ``kaldo.interfaces.gpumd_io.read_gpumd_fc``
    expects (format_version=1, geometry, supercells, fc2 dense, fc3 sparse COO
    split into coords/data/shape, units, grid_order='C', ASR flag, provenance).

    Parameters
    ----------
    path : str
        Output ``.npz`` path.
    atoms : ase.Atoms
        Primitive unit cell.
    supercell, third_supercell : tuple of int
        Second- and third-order supercell shapes.
    fc2 : numpy.ndarray
        Second-order force constants, shape (1, n_uc, 3, n_rep2, n_uc, 3).
    fc3 : sparse.COO
        Third-order force constants.
    nep_path : str
        Path to the NEP potential (a sha256 of it is embedded for provenance).
    acoustic_sum_applied : bool
        Whether the acoustic sum rule has already been applied to ``fc2``.
    """
    nep_hash = _nep_sha256(nep_path)
    np.savez_compressed(
        path,
        format_version=np.int64(1),
        atomic_numbers=atoms.get_atomic_numbers().astype(np.int64),
        positions=atoms.get_positions().astype(np.float64),
        cell=np.array(atoms.cell).astype(np.float64),
        supercell=np.array(supercell, dtype=np.int64),
        third_supercell=np.array(third_supercell, dtype=np.int64),
        fc2=np.ascontiguousarray(fc2, dtype=np.float64),
        fc3_coords=fc3.coords.astype(np.int32),
        fc3_data=fc3.data.astype(np.float64),
        fc3_shape=np.array(fc3.shape, dtype=np.int64),
        units_fc2='eV/angstrom^2',
        units_fc3='eV/angstrom^3',
        grid_order='C',
        acoustic_sum_applied=np.bool_(acoustic_sum_applied),
        nep_potential=f'{nep_path}#sha256:{nep_hash}',
        generator='gpumd_to_kaldo v1')


def main(argv=None):
    """CLI: compute NEP force constants and export a kaldo ``gpumd_fc.npz``."""
    parser = argparse.ArgumentParser(
        description='Export NEP force constants to a kaldo gpumd_fc.npz archive.')
    parser.add_argument('--nep', required=True, help='Path to the NEP potential file.')
    parser.add_argument('--out', default='gpumd_fc.npz', help='Output npz path.')
    parser.add_argument('--supercell', type=int, nargs=3, default=[3, 3, 3],
                        metavar=('NX', 'NY', 'NZ'), help='fc2/fc3 supercell.')
    parser.add_argument('--third-supercell', type=int, nargs=3, default=None,
                        metavar=('NX', 'NY', 'NZ'),
                        help='Separate fc3 supercell (defaults to --supercell).')
    parser.add_argument('--a', type=float, default=None,
                        help='Si lattice constant (Angstrom); default: relaxed value.')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Drop fc3 entries with |value| <= threshold.')
    parser.add_argument('--acoustic-sum', action='store_true',
                        help='Apply the acoustic sum rule to fc2 before writing.')
    args = parser.parse_args(argv)

    third = tuple(args.third_supercell) if args.third_supercell else tuple(args.supercell)
    a0 = args.a if args.a is not None else relaxed_lattice_constant(args.nep)
    atoms = silicon_unitcell(a=a0)
    ph3 = compute_fc3(atoms, args.nep, supercell=tuple(args.supercell))
    fc2, fc3 = to_kaldo_layout(ph3, atoms, tuple(args.supercell), third,
                               threshold=args.threshold)
    if args.acoustic_sum:
        fc2 = apply_acoustic_sum_rule(fc2)
    write_gpumd_fc(args.out, atoms, tuple(args.supercell), third, fc2, fc3,
                   nep_path=args.nep, acoustic_sum_applied=args.acoustic_sum)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
