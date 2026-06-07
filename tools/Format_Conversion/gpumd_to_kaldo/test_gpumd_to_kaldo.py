"""Tests for gpumd_to_kaldo.py — B.1 and B.2.

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
