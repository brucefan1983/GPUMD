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

"""PyTorch cell-list neighbor search for MD-sized structures.

The training / prediction path uses :func:`torchnep.data.build_neighbor_list_np`,
an O(N**2 * n_images) brute-force builder. That is perfectly fine for the small
periodic cells in a fit set (tens to a few hundred atoms), but it materialises a
dense (N, N, n_images) displacement tensor and so blows up — in both memory and
time — once an ASE-driven MD run reaches thousands of atoms.

This module provides a linked-cell (a.k.a. cell-list) search written entirely in
PyTorch so it stays on the model's device (CPU / CUDA / MPS), avoids the numpy
round-trip, and is O(N) in the common condensed-matter regime.

Two entry points:

* :func:`build_neighbor_list` — build the full directed neighbor list in one go.
  Used by the standard (small-to-medium) ASE path.

* :class:`CellList` — build the spatial bin table **once**, then query the
  neighbors of an arbitrary subset of centre atoms with :meth:`CellList.query`.
  This lets a caller stream over centre-atom blocks without ever materialising
  the whole pair list — the basis of the memory-bounded tiled compute path.

When the cell is too small for the linked-cell decomposition (fewer than 3 bins
along some lattice direction — i.e. a tiny training-style box) the builder
returns ``None`` / falls back to the numpy builder, which is correct and cheap
there. The pair *set* produced is identical (to round-off) to the numpy builder;
pair ordering may differ, which is irrelevant because every downstream consumer
scatter-sums over pairs.
"""

import numpy as np
import torch

from .data import build_neighbor_list_np


def _search_dtype(device: torch.device) -> torch.dtype:
    """Float precision for the geometry of the search.

    float64 everywhere it is supported (CPU, CUDA) so the pair geometry is
    bit-identical to the numpy reference; MPS has no float64, so fall back to
    float32 there (models on MPS are float32 anyway).
    """
    return torch.float32 if device.type == "mps" else torch.float64


class CellList:
    """Spatial bin table for repeated neighbor queries.

    Built once from positions + cell + cutoff; ``query(center_ids)`` then returns
    the neighbors of those centre atoms. ``ok`` is False when the cell is too
    small for the +/-1 stencil (caller should fall back to a brute-force builder).
    """

    def __init__(self, positions, cell, cutoff, device="cpu"):
        self.out_device = torch.device(device)
        # The geometry (binning, wrapping, displacements) is done on a
        # float64-capable device, then results are moved to ``out_device``.
        # Positions reach ~1e2 A, where float32 has only ~1e-5 A resolution;
        # at stiff short bonds (e.g. high-pressure ZBL) that feeds a visible
        # force error. MPS has no float64, so for MPS the search runs on CPU
        # (float64) and only the indices + rij are shipped to the GPU — the
        # heavy NN / force math still runs on MPS. CPU / CUDA search in place.
        self.search_device = (torch.device("cpu")
                              if self.out_device.type == "mps" else self.out_device)
        self.device = self.search_device  # internal tensors live here
        self.sdtype = _search_dtype(self.search_device)
        self.cutoff = float(cutoff)
        pos = torch.as_tensor(np.asarray(positions), dtype=self.sdtype, device=self.device)
        cell_t = torch.as_tensor(np.asarray(cell), dtype=self.sdtype, device=self.device)
        self.N = pos.shape[0]
        self.ok = self._build(pos, cell_t)

    def _build(self, pos, cell):
        if self.N == 0:
            return False
        inv = torch.linalg.inv(cell)
        # Perpendicular width between lattice planes along reciprocal direction i
        # is 1/|inv[:, i]| (columns of inv are the reciprocal vectors).
        widths = 1.0 / torch.linalg.norm(inv, dim=0)
        n_cells = torch.floor(widths / self.cutoff).to(torch.long)
        # The +/-1 stencil only covers the cutoff when each direction has >= 3
        # bins (bin width >= cutoff and the bin and its two periodic neighbours
        # are distinct). Tiny boxes are not handled here.
        if bool((n_cells < 3).any()):
            return False

        self.cell = cell
        self.n_cells = n_cells
        self.ncx, self.ncy, self.ncz = (int(n_cells[0]), int(n_cells[1]), int(n_cells[2]))
        total = self.ncx * self.ncy * self.ncz

        # Wrap into the primary cell (translation-invariant under full PBC).
        frac = pos @ inv
        frac = frac - torch.floor(frac)
        self.pos_w = frac @ cell

        bin_xyz = torch.floor(frac * n_cells).to(torch.long)
        self.bin_xyz = torch.minimum(bin_xyz, n_cells - 1).clamp_(min=0)
        cell_id = (self.bin_xyz[:, 0] * self.ncy + self.bin_xyz[:, 1]) * self.ncz + self.bin_xyz[:, 2]

        # Dense bin table (total, max_per) of atom indices, -1 padded, via one sort.
        counts = torch.bincount(cell_id, minlength=total)
        self.max_per = int(counts.max())
        order = torch.argsort(cell_id)
        offsets = torch.zeros(total + 1, dtype=torch.long, device=self.device)
        offsets[1:] = torch.cumsum(counts, 0)
        intra = torch.arange(self.N, device=self.device) - offsets[cell_id[order]]
        table = torch.full((total * self.max_per,), -1, dtype=torch.long, device=self.device)
        table[cell_id[order] * self.max_per + intra] = order
        self.table = table.view(total, self.max_per)

        rng = torch.tensor([-1, 0, 1], dtype=torch.long, device=self.device)
        self.offs = torch.stack(
            torch.meshgrid(rng, rng, rng, indexing="ij"), dim=-1).reshape(-1, 3)
        return True

    def query(self, center_ids, sub_chunk=4000):
        """Neighbors of ``center_ids``. Returns (pi, pj, rij) torch tensors.

        ``pi`` are the (global) centre ids, ``pj`` the neighbor ids, ``rij`` the
        displacement r_j(image) - r_i. ``sub_chunk`` bounds the candidate tensor
        so peak memory stays ~ sub_chunk * 27 * max_per regardless of how many
        centres are queried at once.
        """
        c2 = self.cutoff * self.cutoff
        center_ids = torch.as_tensor(center_ids, dtype=torch.long, device=self.device)
        out = self.out_device
        # rij is computed in float64 on the search device; MPS can't hold
        # float64, so cast it down before shipping there (CPU/CUDA keep float64).
        out_rdtype = torch.float32 if out.type == "mps" else self.sdtype
        pis, pjs, rijs = [], [], []
        for st in range(0, center_ids.shape[0], sub_chunk):
            cidx = center_ids[st:st + sub_chunk]
            nb = self.bin_xyz[cidx].unsqueeze(1) + self.offs.unsqueeze(0)    # (C, S, 3)
            img = torch.div(nb, self.n_cells, rounding_mode="floor")        # whole cells crossed
            nb_w = nb - img * self.n_cells
            nb_flat = (nb_w[..., 0] * self.ncy + nb_w[..., 1]) * self.ncz + nb_w[..., 2]
            cand = self.table[nb_flat]                                      # (C, S, max_per)
            shift = (img.to(self.sdtype) @ self.cell)                       # (C, S, 3)
            valid = cand >= 0
            q = torch.where(valid, cand, torch.zeros_like(cand))
            rij = (self.pos_w[q] + shift.unsqueeze(2)
                   - self.pos_w[cidx].unsqueeze(1).unsqueeze(1))            # (C, S, max_per, 3)
            d2 = (rij * rij).sum(-1)
            keep = valid & (d2 < c2) & (d2 > 1e-20)
            if not bool(keep.any()):
                continue
            ci = cidx.unsqueeze(1).unsqueeze(2).expand_as(cand)
            pis.append(ci[keep]); pjs.append(q[keep]); rijs.append(rij[keep])
        if not pis:
            z = torch.zeros(0, dtype=torch.long, device=out)
            return z, z.clone(), torch.zeros(0, 3, dtype=out_rdtype, device=out)
        # Geometry computed on search_device (float64 for CPU/CUDA); ship the
        # integer pairs and displacements to the compute device.
        return (torch.cat(pis).to(out), torch.cat(pjs).to(out),
                torch.cat(rijs).to(out_rdtype).to(out))


def build_neighbor_list(positions, cell, cutoff, device="cpu",
                        dtype=torch.float64, max_pairs_chunk=2_000_000):
    """Build a directed neighbor list (each physical pair appears as i->j and j->i).

    Parameters
    ----------
    positions : (N, 3) array-like        atomic positions (A); may lie outside the cell.
    cell : (3, 3) array-like              lattice vectors as ROWS.
    cutoff : float                        neighbor cutoff (A).
    device : str or torch.device          device for the search and the output tensors.
    dtype : torch.dtype                   dtype of the returned ``rij`` (matches the model).
    max_pairs_chunk : int                 candidate-tensor size budget per chunk (tunes peak memory).

    Returns
    -------
    pair_i, pair_j : (P,) int64 tensors   central / neighbor atom indices, on ``device``.
    rij : (P, 3) ``dtype`` tensor         displacement vectors r_j(image) - r_i, on ``device``.
    """
    device = torch.device(device)
    cl = CellList(positions, cell, cutoff, device=device)
    if not cl.ok:
        # Small / degenerate cell — defer to the numpy brute-force builder.
        pi, pj, rij = build_neighbor_list_np(
            np.asarray(positions), np.asarray(cell), float(cutoff))
        return (torch.from_numpy(pi).to(device),
                torch.from_numpy(pj).to(device),
                torch.from_numpy(np.ascontiguousarray(rij)).to(device=device, dtype=dtype))

    sub = max(1, max_pairs_chunk // (27 * max(cl.max_per, 1)))
    pi, pj, rij = cl.query(torch.arange(cl.N, device=device), sub_chunk=sub)
    return pi, pj, rij.to(dtype)
