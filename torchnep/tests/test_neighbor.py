# Copyright 2025 Yongchao Wu and the GPUMD development team
# This file is part of GPUMD (Torchnep project).
# GPUMD is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# GPUMD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.

"""Tests for the MD-oriented PyTorch cell-list neighbor search.

The linked-cell builder in ``torchnep.neighbor`` is the fast path used by the
ASE calculator for large MD cells; ``build_neighbor_list_np`` is the numpy
brute-force builder used for small fit-set boxes. They must agree on the pair
*set* (ordering is irrelevant — every consumer scatter-sums over pairs).

Coverage:
  1. the cell-list pair set is identical (to round-off) to the numpy builder for
     orthorhombic and triclinic large cells, including atoms outside the box;
  2. tiny boxes (< 3 bins per direction) transparently fall back to numpy;
  3. the empty / single-atom edge cases return empty pair lists;
  4. a periodic supercell reproduces its unit cell's energy-per-atom and forces
     to machine precision (the end-to-end correctness signal for the fast path).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torchnep.data import build_neighbor_list_np          # noqa: E402
from torchnep.neighbor import build_neighbor_list, CellList  # noqa: E402
from torchnep.nep import NEPCalculator                    # noqa: E402
from torchnep.data import read_xyz                        # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"


def _canonical(pi, pj, rij):
    """Order-independent canonical form: rows of (i, j, round(rij)) sorted."""
    pi = np.asarray(pi); pj = np.asarray(pj); rij = np.round(np.asarray(rij), 6)
    key = np.concatenate([pi[:, None], pj[:, None], rij], axis=1)
    return key[np.lexsort(key.T[::-1])]


def _assert_same_pairs(pos, cell, cutoff):
    pi_np, pj_np, rij_np = build_neighbor_list_np(pos, cell, cutoff)
    pi, pj, rij = build_neighbor_list(pos, cell, cutoff, device="cpu",
                                      dtype=torch.float64)
    a = _canonical(pi_np, pj_np, rij_np)
    b = _canonical(pi.numpy(), pj.numpy(), rij.numpy())
    assert a.shape == b.shape, f"pair count differs: {a.shape} vs {b.shape}"
    assert np.allclose(a, b, atol=1e-5)
    return pi


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cell_list_matches_numpy_orthorhombic(seed):
    rng = np.random.default_rng(seed)
    cell = np.eye(3) * 30.0
    pos = rng.random((400, 3)) @ cell
    # also exercise atoms pushed outside the primary cell
    pos[:20] += cell[0] * 2.0
    _assert_same_pairs(pos, cell, 6.0)
    # confirm the fast path actually ran (30/6 = 5 bins per dir >= 3)
    assert CellList(pos, cell, 6.0, device="cpu").ok


@pytest.mark.parametrize("seed", [0, 1])
def test_cell_list_matches_numpy_triclinic(seed):
    rng = np.random.default_rng(seed)
    L = 40.0
    cell = np.array([[L, 0, 0], [4.0, L, 0], [3.0, 2.0, L]])
    pos = rng.random((600, 3)) @ cell
    _assert_same_pairs(pos, cell, 6.0)


def test_small_box_falls_back_to_numpy():
    # 8 A box, cutoff 6 -> 1 bin per direction -> fast path declines (None),
    # build_neighbor_list still returns the correct numpy result.
    cell = np.eye(3) * 8.0
    pos = np.random.default_rng(0).random((20, 3)) @ cell
    assert not CellList(pos, cell, 6.0, device="cpu").ok
    _assert_same_pairs(pos, cell, 6.0)


def test_empty_and_single_atom():
    cell = np.eye(3) * 30.0
    for n in (0, 1):
        pos = np.zeros((n, 3))
        pi, pj, rij = build_neighbor_list(pos, cell, 6.0, "cpu", torch.float64)
        assert len(pi) == len(pj) == len(rij) == 0


def test_supercell_reproduces_unit_cell():
    """A periodic replica has identical local environments -> identical E/atom
    and per-atom forces. Drives the full compute() through the cell-list path."""
    fr = read_xyz(str(DATA_DIR / "CrCoNi.xyz"))[0]
    cell0 = np.asarray(fr["cell"]); pos0 = np.asarray(fr["positions"])
    sp0 = list(fr["species"])
    calc = NEPCalculator(str(DATA_DIR / "nep_CrCoNi.txt"), dtype=torch.float64)

    r0 = calc.compute(sp0, pos0, cell0)               # numpy fallback (small box)
    epa0 = float(r0["energy"].sum()) / len(pos0)

    reps = 3                                           # 31.8 A box -> cell-list path
    cell = cell0 * reps
    pos, sp = [], []
    for i in range(reps):
        for j in range(reps):
            for k in range(reps):
                pos.append(pos0 + np.array([i, j, k]) @ cell0)
                sp += sp0
    pos = np.concatenate(pos)
    r = calc.compute(sp, pos, cell)                    # cell-list path
    epa = float(r["energy"].sum()) / len(pos)

    assert abs(epa - epa0) < 1e-10
    # forces on the first replica equal the unit-cell forces
    assert np.abs(r["forces"].numpy()[:len(pos0)] - r0["forces"].numpy()).max() < 1e-9


def test_tiled_matches_autograd():
    """The memory-bounded tiled analytical path must reproduce the autograd
    compute() to round-off, including across block boundaries (small block_size
    forces many tiles) and for a ZBL model (CrCoNi has typewise ZBL)."""
    fr = read_xyz(str(DATA_DIR / "CrCoNi.xyz"))[0]
    cell0 = np.asarray(fr["cell"]); pos0 = np.asarray(fr["positions"])
    sp0 = list(fr["species"])
    reps = 3
    cell = cell0 * reps
    pos, sp = [], []
    for i in range(reps):
        for j in range(reps):
            for k in range(reps):
                pos.append(pos0 + np.array([i, j, k]) @ cell0)
                sp += sp0
    pos = np.concatenate(pos)

    calc = NEPCalculator(str(DATA_DIR / "nep_CrCoNi.txt"), dtype=torch.float64)
    ref = calc.compute(sp, pos, cell)
    til = calc.compute_tiled(sp, pos, cell, block_size=500)  # ~6 tiles

    assert abs(float(ref["energy"].sum()) - float(til["energy"].sum())) < 1e-9
    assert np.abs(ref["energy"].numpy() - til["energy"].numpy()).max() < 1e-9
    assert np.abs(ref["forces"].numpy() - til["forces"].numpy()).max() < 1e-9
    assert np.abs(ref["virial"].numpy().sum(0)
                  - til["virial"].numpy().sum(0)).max() < 1e-8


def test_mps_search_runs_in_float64():
    """Regression: an MPS-targeted CellList must run its geometry on CPU in
    float64 (MPS has no float64). Positions reach ~1e2 A where float32 has only
    ~1e-5 A resolution, which at stiff short bonds caused a visible force error.
    """
    cl = CellList(np.random.default_rng(0).random((50, 3)) * 30.0,
                  np.eye(3) * 30.0, 6.0, device="mps")
    assert cl.search_device.type == "cpu"
    assert cl.sdtype == torch.float64

    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return
    # On real MPS: forces on a large-coordinate cell must match CPU float32
    # (both now fed a float64-accurate neighbor list).
    fr = read_xyz(str(DATA_DIR / "CrCoNi.xyz"))[0]
    cell0 = np.asarray(fr["cell"]); pos0 = np.asarray(fr["positions"])
    sp0 = list(fr["species"])
    reps = 3
    cell = cell0 * reps
    pos, sp = [], []
    for i in range(reps):
        for j in range(reps):
            for k in range(reps):
                pos.append(pos0 + np.array([i, j, k]) @ cell0)
                sp += sp0
    pos = np.concatenate(pos) + 200.0  # large coordinates stress float32 geometry
    c_cpu = NEPCalculator(str(DATA_DIR / "nep_CrCoNi.txt"), dtype=torch.float32, device="cpu")
    c_mps = NEPCalculator(str(DATA_DIR / "nep_CrCoNi.txt"), dtype=torch.float32, device="mps")
    f_cpu = c_cpu.compute_tiled(sp, pos, cell, block_size=700)["forces"].cpu().numpy()
    f_mps = c_mps.compute_tiled(sp, pos, cell, block_size=700)["forces"].cpu().numpy()
    assert np.abs(f_cpu - f_mps).max() < 1e-3


def test_auto_block_size():
    """block_size='auto' stays in [256, N], shrinks for float64 vs float32, and
    produces the same result as an explicit block size."""
    fr = read_xyz(str(DATA_DIR / "CrCoNi.xyz"))[0]
    cell0 = np.asarray(fr["cell"]); pos0 = np.asarray(fr["positions"])
    sp0 = list(fr["species"])
    reps = 3
    cell = cell0 * reps
    pos, sp = [], []
    for i in range(reps):
        for j in range(reps):
            for k in range(reps):
                pos.append(pos0 + np.array([i, j, k]) @ cell0)
                sp += sp0
    pos = np.concatenate(pos)
    N = len(sp)

    c32 = NEPCalculator(str(DATA_DIR / "nep_CrCoNi.txt"), dtype=torch.float32)
    c64 = NEPCalculator(str(DATA_DIR / "nep_CrCoNi.txt"), dtype=torch.float64)
    b32 = c32._auto_block_size(N, cell)
    b64 = c64._auto_block_size(N, cell)
    assert 256 <= b32 <= N and 256 <= b64 <= N
    assert b64 <= b32  # float64 needs more memory per atom -> smaller block

    # 'auto' must give the same physics as a fixed block size.
    auto = c64.compute_tiled(sp, pos, cell, block_size="auto")
    fixed = c64.compute_tiled(sp, pos, cell, block_size=500)
    assert abs(float(auto["energy"].sum()) - float(fixed["energy"].sum())) < 1e-9
    assert np.abs(auto["forces"].numpy() - fixed["forces"].numpy()).max() < 1e-9


def test_tiled_small_box_falls_back():
    """compute_tiled on a box too small for the cell-list defers to compute()
    and still returns the correct energy/forces."""
    fr = read_xyz(str(DATA_DIR / "CrCoNi.xyz"))[0]
    calc = NEPCalculator(str(DATA_DIR / "nep_CrCoNi.txt"), dtype=torch.float64)
    ref = calc.compute(list(fr["species"]), np.asarray(fr["positions"]),
                       np.asarray(fr["cell"]))
    til = calc.compute_tiled(list(fr["species"]), np.asarray(fr["positions"]),
                             np.asarray(fr["cell"]))
    assert abs(float(ref["energy"].sum()) - float(til["energy"].sum())) < 1e-9
    assert np.abs(ref["forces"].numpy() - til["forces"].numpy()).max() < 1e-9
