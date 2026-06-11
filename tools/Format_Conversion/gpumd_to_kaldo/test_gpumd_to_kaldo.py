"""Tests for gpumd_to_kaldo.py — B.1 through B.4.

Run from this directory:
    conda run -n gpumd-kaldo python -m pytest test_gpumd_to_kaldo.py -v

The SI_NEP env var (or the relative path default) must point to the
Si_2022_NEP4_4body.txt potential file.
"""
import os

import numpy as np
import pytest

import gpumd_to_kaldo as g

NEP = os.environ.get(
    'SI_NEP',
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'potentials', 'nep',
                 'Si_2022_NEP4_4body.txt'))

_NEP_FOUND = os.path.isfile(NEP)


# ---------------------------------------------------------------------------
# B.1 — Si unit cell and NEP force helpers
# ---------------------------------------------------------------------------

def test_silicon_unitcell_shape():
    """silicon_unitcell returns a 2-atom primitive cell."""
    atoms = g.silicon_unitcell(a=5.43)
    assert len(atoms) == 2
    assert list(atoms.get_chemical_symbols()) == ['Si', 'Si']
    assert atoms.cell.volume > 0
    assert all(atoms.pbc)


@pytest.mark.skipif(not _NEP_FOUND, reason='Si NEP not found')
def test_nep_forces_shape():
    """nep_forces returns an array of shape (n_atoms, 3)."""
    atoms = g.silicon_unitcell(a=5.43)
    f = g.nep_forces(atoms, NEP)
    assert f.shape == (2, 3)


@pytest.mark.skipif(not _NEP_FOUND, reason='Si NEP not found')
def test_relaxed_lattice_constant():
    """relaxed_lattice_constant returns an a0 near 5.45 Å for the Fan-2022 Si NEP."""
    a0 = g.relaxed_lattice_constant(NEP)
    # Fan-2022 Si NEP4 equilibrium is around 5.455 Å (not 5.43)
    assert 5.43 < a0 < 5.47, f"a0={a0:.4f} out of expected range 5.43–5.47"


@pytest.mark.skipif(not _NEP_FOUND, reason='Si NEP not found')
def test_equilibrium_forces_small():
    """Forces at the relaxed lattice constant are near machine zero."""
    a0 = g.relaxed_lattice_constant(NEP)
    atoms = g.silicon_unitcell(a=a0)
    f = g.nep_forces(atoms, NEP)
    assert f.shape == (2, 3)
    assert np.abs(f).max() < 1e-6, (
        f"Forces not near zero at relaxed a0={a0:.4f}: max|F|={np.abs(f).max():.2e}")


# ---------------------------------------------------------------------------
# B.2 — fc2 via phonopy, fc3 via phono3py
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _NEP_FOUND, reason='Si NEP not found')
def test_fc2_returns_phonopy():
    """compute_fc2 returns a Phonopy object."""
    from phonopy import Phonopy
    a0 = g.relaxed_lattice_constant(NEP)
    atoms = g.silicon_unitcell(a=a0)
    ph = g.compute_fc2(atoms, NEP, supercell=(2, 2, 2), displacement=0.01)
    assert isinstance(ph, Phonopy)


@pytest.mark.skipif(not _NEP_FOUND, reason='Si NEP not found')
def test_fc2_gamma_optical_frequency():
    """Gamma-point optical frequency of Si is physical (~15.5 THz).

    Uses the relaxed NEP lattice constant a0 so the acoustic sum rule is
    not contaminated by residual stress.  The Fan-2022 Si NEP4 gives an
    optical mode near 14.3 THz at the relaxed volume (accept 14.0–16.5 THz).
    """
    a0 = g.relaxed_lattice_constant(NEP)
    atoms = g.silicon_unitcell(a=a0)
    ph = g.compute_fc2(atoms, NEP, supercell=(3, 3, 3), displacement=0.01)
    ph.run_mesh([1, 1, 1])  # Gamma only
    freqs = ph.get_mesh_dict()['frequencies'][0]  # THz
    print(f"\nGamma frequencies (THz): {freqs}")
    # triply-degenerate optical mode near 15.5 THz for diamond Si
    assert 14.0 < max(freqs) < 16.5, (
        f"Max Gamma frequency {max(freqs):.2f} THz outside 14.0–16.5 THz window. "
        f"Full list: {freqs}")


@pytest.mark.skipif(not _NEP_FOUND, reason='Si NEP not found')
def test_fc3_returns_phono3py():
    """compute_fc3 returns a Phono3py object with fc2 and fc3 set."""
    from phono3py import Phono3py
    a0 = g.relaxed_lattice_constant(NEP)
    atoms = g.silicon_unitcell(a=a0)
    ph3 = g.compute_fc3(atoms, NEP, supercell=(2, 2, 2), displacement=0.03)
    assert isinstance(ph3, Phono3py)
    assert ph3.fc2 is not None, "fc2 should be set after compute_fc3"
    assert ph3.fc3 is not None, "fc3 should be set after compute_fc3"
    assert ph3.fc2.ndim == 4, f"fc2 should be 4-D, got {ph3.fc2.ndim}-D"
    assert ph3.fc3.ndim == 6, f"fc3 should be 6-D, got {ph3.fc3.ndim}-D"


# ---------------------------------------------------------------------------
# Shared, cached phono3py object for the (expensive) B.3 / B.4 tests
# ---------------------------------------------------------------------------

_SC = (2, 2, 2)


@pytest.fixture(scope='module')
def si_ph3():
    """A phono3py object for relaxed Si at supercell (2, 2, 2), computed once."""
    a0 = g.relaxed_lattice_constant(NEP)
    atoms = g.silicon_unitcell(a=a0)
    ph3 = g.compute_fc3(atoms, NEP, supercell=_SC, displacement=0.03)
    return atoms, ph3


def _full_fc2_phonopy(ph3):
    """Return a FULL (N, N, 3, 3) fc2 in the phono3py supercell atom order.

    Independent of the exporter: expands a compact fc2 by symmetry-translating
    the reference-cell rows over every lattice image, mirroring how the full
    array would have been produced.  Used only as test scaffolding.
    """
    import numpy as np
    sc = ph3.supercell
    s2u = np.array(sc.s2u_map)
    p2s = list(np.array(ph3.primitive.p2s_map))
    fc2 = np.array(ph3.fc2, dtype=np.float64)
    n_satom = len(sc)
    if fc2.shape[0] == n_satom:
        return fc2
    # Compact -> full: row a maps to (prim atom of a) translated to a's image.
    scaled = np.array(sc.scaled_positions)
    full = np.zeros((n_satom, n_satom, 3, 3), dtype=np.float64)
    for a in range(n_satom):
        prim_satom = s2u[a]                 # supercell index of a's prim image
        c = p2s.index(prim_satom)           # compact row for that prim atom
        # lattice shift from the reference image to atom a (in scaled coords)
        shift = scaled[a] - scaled[prim_satom]
        for b in range(n_satom):
            # find b' = b shifted back by the same lattice vector
            target = (scaled[b] - shift)
            diff = scaled - target
            diff -= np.rint(diff)
            bp = int(np.argmin((diff ** 2).sum(axis=1)))
            full[a, b] = fc2[c, bp]
    return full


def _kaldo_supercell_order(atoms, supercell):
    """ase supercell built in kaldo's replication order (rep*n_uc + unit)."""
    sx, sy, sz = supercell
    return atoms * (sx, 1, 1) * (1, sy, 1) * (1, 1, sz)


def _reorder_fc2_to_kaldo(ph3, atoms, supercell, full_fc2):
    """Reorder a phono3py-ordered full fc2 into kaldo supercell atom order.

    Maps each kaldo supercell atom (index rep*n_uc + unit) to the phono3py
    supercell atom occupying the same Cartesian site, then permutes both axes.
    """
    import numpy as np
    from kaldo.grid import Grid
    sc = ph3.supercell
    n_uc = len(atoms)
    n_rep = int(np.prod(supercell))
    grid = Grid(tuple(supercell), order='C').grid(is_wrapping=False)
    cell = np.array(atoms.cell)
    inv = np.linalg.inv(cell)
    prim_pos = atoms.get_positions()
    sc_cart = np.array(sc.scaled_positions) @ np.array(sc.cell)
    s2u = np.array(sc.s2u_map)
    u2u = sc.u2u_map
    # phono3py supercell atom -> (rep, unit)
    perm = np.empty(n_rep * n_uc, dtype=int)  # kaldo idx -> phono3py idx
    for a in range(len(sc_cart)):
        j = u2u[s2u[a]]
        cidx = np.rint((sc_cart[a] - prim_pos[j]) @ inv).astype(int) % np.array(supercell)
        r = np.where((grid == cidx).all(axis=1))[0][0]
        perm[r * n_uc + j] = a
    return full_fc2[np.ix_(perm, perm)]


# ---------------------------------------------------------------------------
# B.3 — fc2 layout matches an INDEPENDENT hiphive oracle (highest-risk bug)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _NEP_FOUND, reason='Si NEP not found')
def test_fc2_layout_matches_hiphive_oracle(si_ph3):
    """Direct fc2 mapping must equal the hiphive route element-wise.

    The oracle is built entirely in the TEST: full phono3py fc2 -> reorder to
    kaldo replication order -> hiphive ForceConstants.from_arrays -> kaldo's
    own transform (transpose(0,2,1,3).reshape(...)[0][None]).  This exercises
    the replica permutation, which Gamma-only checks cannot see.
    """
    pytest.importorskip('kaldo')
    pytest.importorskip('hiphive')
    from hiphive import ForceConstants as HFC
    atoms, ph3 = si_ph3
    n_uc = len(atoms)
    n_rep = int(np.prod(_SC))

    fc2_mine, _ = g.to_kaldo_layout(ph3, atoms, _SC, _SC)
    assert fc2_mine.shape == (1, n_uc, 3, n_rep, n_uc, 3)
    assert fc2_mine.dtype == np.float64

    full = _full_fc2_phonopy(ph3)
    fc2_kaldo_order = _reorder_fc2_to_kaldo(ph3, atoms, _SC, full)
    ase_sc = _kaldo_supercell_order(atoms, _SC)
    hfc = HFC.from_arrays(ase_sc, fc2_array=fc2_kaldo_order)
    arr = hfc.get_fc_array(2).transpose(0, 2, 1, 3)
    arr = arr.reshape((n_rep, n_uc, 3, n_rep, n_uc, 3))
    fc2_oracle = arr[0][np.newaxis]

    max_diff = np.abs(fc2_mine - fc2_oracle).max()
    print(f"\nfc2 max abs diff vs hiphive oracle: {max_diff:.3e}")
    np.testing.assert_allclose(fc2_mine, fc2_oracle, atol=1e-8)


@pytest.mark.skipif(not _NEP_FOUND, reason='Si NEP not found')
def test_fc2_dispersion_matches_phonopy_nonGamma(si_ph3, tmp_path):
    """Non-Gamma frequencies from the exported fc match phonopy's native ones.

    This is an INDEPENDENT cross-check of the replica -> R-vector pairing in
    ``to_kaldo_layout``.  The hiphive fc2 oracle and the Gamma frequency test
    cannot catch a replica-permutation bug: the oracle reuses the exporter's own
    C-grid formula, and D(Gamma) = sum_r phi_r is invariant under any permutation
    of the replicas.  Phonopy, by contrast, builds the dynamical matrix from its
    OWN real-space R-vectors (it never touches kaldo's C grid), so comparing
    frequencies at non-Gamma q-points pins down whether each phi_r block sits on
    the correct R-vector.  A wrong permutation shifts whole modes by several THz
    (verified offline: rolling the replica axis by 1 drives the max diff to
    ~24 THz), so the 2e-2 THz tolerance below is tight enough to expose it while
    absorbing the small finite-displacement / ASR numerical noise (~7e-3 THz).

    Both routes use the SAME primitive ``atoms`` and the SAME (2, 2, 2)
    supercell, so reduced-coordinate q maps 1:1 between phonopy's reduced
    q-points and kaldo's fractional-reciprocal ``q_point``.  The q-points are all
    COMMENSURATE with the supercell (components in {0, 1/2}); at those points the
    Fourier sum is exact and independent of the min-image R-vector choice, so the
    comparison probes the replica mapping rather than interpolation conventions.
    (Incommensurate points such as [0.25, 0.25, 0.25] legitimately differ between
    the two codes because of differing real-space image conventions and are
    therefore excluded.)
    """
    pytest.importorskip('kaldo')
    from kaldo.forceconstants import ForceConstants
    from kaldo.observables.harmonic_with_q import HarmonicWithQ
    atoms, ph3 = si_ph3

    # --- phonopy-native reference (its own R-vectors, no kaldo C grid) ---
    ph = g.compute_fc2(atoms, NEP, supercell=_SC, displacement=0.01)
    gamma = [0.0, 0.0, 0.0]
    nongamma = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5],
                [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
                [0.5, 0.5, 0.5]]
    qpoints = [gamma] + nongamma
    ph.run_qpoints(qpoints)
    phonopy_freqs = ph.get_qpoints_dict()['frequencies']  # (nq, n_modes) THz

    # --- kaldo-from-exported (round-trips through to_kaldo_layout + writer) ---
    fc2, fc3 = g.to_kaldo_layout(ph3, atoms, _SC, _SC)
    fc2 = g.apply_acoustic_sum_rule(fc2)
    g.write_gpumd_fc(str(tmp_path / 'gpumd_fc.npz'), atoms, _SC, _SC, fc2, fc3,
                     nep_path=NEP, acoustic_sum_applied=True)
    fc = ForceConstants.from_folder(folder=str(tmp_path), format='gpumd')

    def kaldo_freqs(q):
        hwq = HarmonicWithQ(q_point=np.array(q, dtype=float), second=fc.second,
                            storage='numpy')
        return hwq.frequency[0]  # (n_modes,) THz

    # Sanity anchor: the two q-conventions must agree at Gamma first.
    np.testing.assert_allclose(np.sort(kaldo_freqs(gamma)),
                               np.sort(phonopy_freqs[0]), atol=2e-2)

    max_diff = 0.0
    for q, ph_f in zip(nongamma, phonopy_freqs[1:]):
        k_sorted = np.sort(kaldo_freqs(q))
        p_sorted = np.sort(ph_f)
        max_diff = max(max_diff, float(np.abs(k_sorted - p_sorted).max()))
        np.testing.assert_allclose(k_sorted, p_sorted, atol=2e-2)
    print(f"\nnon-Gamma freq max abs diff (kaldo vs phonopy): {max_diff:.3e} THz")


@pytest.mark.skipif(not _NEP_FOUND, reason='Si NEP not found')
def test_fc2_acoustic_sum_rule_zeroes_rows(si_ph3):
    """After ASR, each reference atom's full fc2 row sums to ~0 (3x3 block)."""
    atoms, ph3 = si_ph3
    fc2, _ = g.to_kaldo_layout(ph3, atoms, _SC, _SC)
    fc2_asr = g.apply_acoustic_sum_rule(fc2)
    n_uc = fc2_asr.shape[1]
    for i in range(n_uc):
        row_sum = np.sum(fc2_asr[0, i, :, :, :, :], axis=(-2, -3))  # (3, 3)
        assert np.abs(row_sum).max() < 1e-9, (
            f"ASR residual for atom {i}: {np.abs(row_sum).max():.2e}")
    # ASR must not mutate the input array
    assert not np.shares_memory(fc2, fc2_asr)


# ---------------------------------------------------------------------------
# B.3 — fc3 layout sanity (shape/dtype, non-empty, acoustic sum residual)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _NEP_FOUND, reason='Si NEP not found')
def test_fc3_layout_shape_and_sum_rule(si_ph3):
    """fc3 has kaldo shape/dtype, is non-empty, and obeys the acoustic sum."""
    import sparse
    atoms, ph3 = si_ph3
    n_uc = len(atoms)
    n_rep = int(np.prod(_SC))
    _, fc3 = g.to_kaldo_layout(ph3, atoms, _SC, _SC)
    assert isinstance(fc3, sparse.COO)
    assert fc3.shape == (n_uc * 3, n_rep * n_uc * 3, n_rep * n_uc * 3)
    assert fc3.data.dtype == np.float64
    assert fc3.nnz > 0, "fc3 should be non-empty"

    dense = fc3.todense().reshape(
        (n_uc, 3, n_rep, n_uc, 3, n_rep, n_uc, 3))
    # Sum over the third leg (l3, k, gamma) for each (i, alpha, l2, j, beta).
    residual = np.abs(dense.sum(axis=(5, 6, 7))).max()
    print(f"\nfc3 acoustic sum residual: {residual:.3e} eV/A^3")
    # phono3py produce_fc3 default does NOT symmetrize, so allow a loose bound.
    assert residual < 1e-2, f"fc3 acoustic sum residual too large: {residual:.3e}"


@pytest.mark.skipif(not _NEP_FOUND, reason='Si NEP not found')
def test_fc3_matches_hiphive_oracle(si_ph3, tmp_path):
    """Strong fc3 oracle: element-wise match against hiphive read_phono3py.

    Builds the oracle in the TEST by writing phono3py fc3 to hdf5, reading it
    back through hiphive, reordering to kaldo replication order, and applying
    kaldo's own fc3 transform.  Independent of the exporter's mapping.
    """
    pytest.importorskip('kaldo')
    pytest.importorskip('hiphive')
    from phono3py.file_IO import write_fc3_to_hdf5
    from hiphive import ForceConstants as HFC
    atoms, ph3 = si_ph3
    n_uc = len(atoms)
    n_rep = int(np.prod(_SC))

    _, fc3_mine = g.to_kaldo_layout(ph3, atoms, _SC, _SC)

    # Expand compact fc3 to full, then reorder to kaldo supercell order.
    full3 = _full_fc3_phonopy(ph3)
    full3 = _reorder_fc3_to_kaldo(ph3, atoms, _SC, full3)
    h5 = tmp_path / 'fc3.hdf5'
    write_fc3_to_hdf5(full3, filename=str(h5))
    ase_sc = _kaldo_supercell_order(atoms, _SC)
    hfc = HFC.read_phono3py(ase_sc, str(h5))
    arr = hfc.get_fc_array(3).transpose(0, 3, 1, 4, 2, 5)
    arr = arr.reshape((n_rep, n_uc, 3, n_rep, n_uc, 3, n_rep, n_uc, 3))
    # kaldo fc3 logical layout: row=(i,alpha) at reference cell (rep 0),
    # col1=(l2,j,beta), col2=(l3,k,gamma).
    oracle = np.zeros((n_uc * 3, n_rep * n_uc * 3, n_rep * n_uc * 3),
                      dtype=np.float64)
    for i in range(n_uc):
        for al in range(3):
            for l2 in range(n_rep):
                for j in range(n_uc):
                    for be in range(3):
                        for l3 in range(n_rep):
                            for k in range(n_uc):
                                block = arr[0, i, al, l2, j, be, l3, k, :]
                                row = i * 3 + al
                                c1 = (l2 * n_uc + j) * 3 + be
                                base = (l3 * n_uc + k) * 3
                                oracle[row, c1, base:base + 3] = block
    mine = fc3_mine.todense()
    max_diff = np.abs(mine - oracle).max()
    print(f"\nfc3 max abs diff vs hiphive oracle: {max_diff:.3e}")
    np.testing.assert_allclose(mine, oracle, atol=1e-8)


def _full_fc3_phonopy(ph3):
    """Expand a compact phono3py fc3 to full (N, N, N, 3, 3, 3). Test-only."""
    import numpy as np
    sc = ph3.supercell
    s2u = np.array(sc.s2u_map)
    p2s = list(np.array(ph3.primitive.p2s_map))
    fc3 = np.array(ph3.fc3, dtype=np.float64)
    n_satom = len(sc)
    if fc3.shape[0] == n_satom:
        return fc3
    scaled = np.array(sc.scaled_positions)

    def shifted_index(idx, shift):
        target = scaled[idx] - shift
        diff = scaled - target
        diff -= np.rint(diff)
        return int(np.argmin((diff ** 2).sum(axis=1)))

    full = np.zeros((n_satom, n_satom, n_satom, 3, 3, 3), dtype=np.float64)
    for a in range(n_satom):
        prim_satom = s2u[a]
        c = p2s.index(prim_satom)
        shift = scaled[a] - scaled[prim_satom]
        for b in range(n_satom):
            bp = shifted_index(b, shift)
            for cc in range(n_satom):
                cp = shifted_index(cc, shift)
                full[a, b, cc] = fc3[c, bp, cp]
    return full


def _reorder_fc3_to_kaldo(ph3, atoms, supercell, full_fc3):
    """Reorder a phono3py-ordered full fc3 into kaldo supercell atom order."""
    import numpy as np
    from kaldo.grid import Grid
    sc = ph3.supercell
    n_uc = len(atoms)
    n_rep = int(np.prod(supercell))
    grid = Grid(tuple(supercell), order='C').grid(is_wrapping=False)
    cell = np.array(atoms.cell)
    inv = np.linalg.inv(cell)
    prim_pos = atoms.get_positions()
    sc_cart = np.array(sc.scaled_positions) @ np.array(sc.cell)
    s2u = np.array(sc.s2u_map)
    u2u = sc.u2u_map
    perm = np.empty(n_rep * n_uc, dtype=int)
    for a in range(len(sc_cart)):
        j = u2u[s2u[a]]
        cidx = np.rint((sc_cart[a] - prim_pos[j]) @ inv).astype(int) % np.array(supercell)
        r = np.where((grid == cidx).all(axis=1))[0][0]
        perm[r * n_uc + j] = a
    return full_fc3[np.ix_(perm, perm, perm)]


# ---------------------------------------------------------------------------
# B.4 — writer round-trips through the kaldo reader + end-to-end load
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _NEP_FOUND, reason='Si NEP not found')
def test_write_npz_roundtrips_through_kaldo_reader(si_ph3, tmp_path):
    """write_gpumd_fc produces an npz the kaldo reader accepts byte-for-byte."""
    pytest.importorskip('kaldo')
    from kaldo.interfaces import gpumd_io
    atoms, ph3 = si_ph3
    fc2, fc3 = g.to_kaldo_layout(ph3, atoms, _SC, _SC)
    out = tmp_path / 'gpumd_fc.npz'
    g.write_gpumd_fc(str(out), atoms, _SC, _SC, fc2, fc3,
                     nep_path=NEP, acoustic_sum_applied=False)
    meta = gpumd_io.read_gpumd_fc(str(tmp_path))
    assert meta['fc2'].shape == fc2.shape
    np.testing.assert_allclose(meta['fc2'], fc2)
    assert meta['fc3'].shape == fc3.shape
    np.testing.assert_allclose(meta['fc3'].todense(), fc3.todense())
    assert tuple(meta['supercell']) == _SC
    assert tuple(meta['third_supercell']) == _SC
    assert meta['acoustic_sum_applied'] is False


@pytest.mark.skipif(not _NEP_FOUND, reason='Si NEP not found')
def test_write_npz_records_nep_sha256(si_ph3, tmp_path):
    """The npz embeds a sha256 of the NEP file in nep_potential."""
    import hashlib
    atoms, ph3 = si_ph3
    fc2, fc3 = g.to_kaldo_layout(ph3, atoms, _SC, _SC)
    out = tmp_path / 'gpumd_fc.npz'
    g.write_gpumd_fc(str(out), atoms, _SC, _SC, fc2, fc3,
                     nep_path=NEP, acoustic_sum_applied=True)
    with open(NEP, 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    data = np.load(out, allow_pickle=False)
    nep_field = str(data['nep_potential'])
    assert sha in nep_field
    assert bool(data['acoustic_sum_applied']) is True


@pytest.mark.skipif(not _NEP_FOUND, reason='Si NEP not found')
def test_end_to_end_force_constants_from_folder(si_ph3, tmp_path):
    """ForceConstants.from_folder(format='gpumd') builds with correct shapes."""
    pytest.importorskip('kaldo')
    from kaldo.forceconstants import ForceConstants
    atoms, ph3 = si_ph3
    n_uc = len(atoms)
    n_rep = int(np.prod(_SC))
    fc2, fc3 = g.to_kaldo_layout(ph3, atoms, _SC, _SC)
    fc2 = g.apply_acoustic_sum_rule(fc2)
    g.write_gpumd_fc(str(tmp_path / 'gpumd_fc.npz'), atoms, _SC, _SC, fc2, fc3,
                     nep_path=NEP, acoustic_sum_applied=True)
    fc = ForceConstants.from_folder(folder=str(tmp_path), format='gpumd')
    assert fc.second.value.shape == (1, n_uc, 3, n_rep, n_uc, 3)
    assert fc.third.value.shape == (n_uc * 3, n_rep * n_uc * 3, n_rep * n_uc * 3)
    assert tuple(fc.supercell) == _SC


# ---------------------------------------------------------------------------
# B.4 — CLI smoke test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _NEP_FOUND, reason='Si NEP not found')
def test_cli_writes_npz(tmp_path):
    """main() wires compute -> layout -> ASR -> write and produces an npz."""
    pytest.importorskip('kaldo')
    from kaldo.interfaces import gpumd_io
    out = tmp_path / 'gpumd_fc.npz'
    g.main(['--nep', NEP, '--supercell', '2', '2', '2',
            '--acoustic-sum', '--out', str(out)])
    assert out.is_file()
    meta = gpumd_io.read_gpumd_fc(str(tmp_path))
    assert meta['acoustic_sum_applied'] is True
    assert tuple(meta['supercell']) == (2, 2, 2)


# ---------------------------------------------------------------------------
# B.5 — CLI --structure option
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _NEP_FOUND, reason='Si NEP not found')
def test_cli_structure_option(tmp_path):
    """--structure PATH produces the same Gamma frequencies as the built-in Si path.

    Writes the built-in Si cell to a POSCAR in tmp_path, then runs main() with
    ``--structure``.  The resulting gpumd_fc.npz is loaded via kaldo
    ``ForceConstants.from_folder(format='gpumd')`` and the Gamma-point
    frequencies are compared against a reference run through the default
    (built-in Si) path.  Both runs use ``--acoustic-sum`` and the same (2,2,2)
    supercell so any index/unit bug would show as a large frequency discrepancy.
    """
    pytest.importorskip('kaldo')
    from kaldo.forceconstants import ForceConstants
    from kaldo.observables.harmonic_with_q import HarmonicWithQ

    # --- reference: built-in Si path ---
    ref_dir = tmp_path / 'ref'
    ref_dir.mkdir()
    g.main(['--nep', NEP, '--supercell', '2', '2', '2',
            '--acoustic-sum', '--out', str(ref_dir / 'gpumd_fc.npz')])
    fc_ref = ForceConstants.from_folder(folder=str(ref_dir), format='gpumd')
    freqs_ref = np.sort(HarmonicWithQ(q_point=np.array([0.0, 0.0, 0.0]),
                                      second=fc_ref.second,
                                      storage='numpy').frequency[0])

    # --- --structure path: write the built-in cell as a POSCAR ---
    a0 = g.relaxed_lattice_constant(NEP)
    atoms = g.silicon_unitcell(a=a0)
    poscar = str(tmp_path / 'POSCAR')
    import ase.io
    ase.io.write(poscar, atoms, format='vasp')

    struct_dir = tmp_path / 'struct'
    struct_dir.mkdir()
    g.main(['--nep', NEP, '--structure', poscar,
            '--supercell', '2', '2', '2',
            '--acoustic-sum', '--out', str(struct_dir / 'gpumd_fc.npz')])
    fc_struct = ForceConstants.from_folder(folder=str(struct_dir), format='gpumd')
    freqs_struct = np.sort(HarmonicWithQ(q_point=np.array([0.0, 0.0, 0.0]),
                                         second=fc_struct.second,
                                         storage='numpy').frequency[0])

    np.testing.assert_allclose(freqs_struct, freqs_ref, atol=1e-6,
                               err_msg='--structure Gamma freqs differ from built-in Si path')
