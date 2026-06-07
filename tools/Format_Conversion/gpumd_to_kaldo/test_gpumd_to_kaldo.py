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
