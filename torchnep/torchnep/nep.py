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

"""
NEP4 calculator for PyTorch.

Loads trained models from nep.txt (GPUMD NEP4 format) and computes
energy, forces, virial, and descriptors.

The descriptor/force computation is implemented in ops.py (pure PyTorch
by default, with optional CUDA kernel acceleration).
"""

import math
import torch
import numpy as np
from typing import Dict

from .constants import ELEMENTS, COVALENT_RADIUS, C3B, C4B, C5B, C4B2
from .neighbor import build_neighbor_list, CellList
from . import ops


def _available_memory_bytes(device: torch.device) -> int:
    """Best-effort free memory for the compute device.

    CUDA -> free VRAM; CPU / MPS (unified memory) -> free system RAM. Tries
    psutil, then OS-specific probes, then a conservative 4 GB fallback so the
    auto block sizer never assumes more than it can verify.
    """
    if device.type == "cuda":
        try:
            return int(torch.cuda.mem_get_info(device)[0])
        except Exception:
            return 4 * 1024 ** 3
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    import sys
    if sys.platform == "linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
    elif sys.platform == "darwin":
        try:
            import subprocess
            out = subprocess.check_output(["vm_stat"]).decode()
            page, free, inactive = 4096, 0, 0
            for line in out.splitlines():
                if "page size of" in line:
                    page = int(line.split("page size of")[1].split("bytes")[0])
                elif line.startswith("Pages free"):
                    free = int(line.split(":")[1].strip().rstrip("."))
                elif line.startswith("Pages inactive"):
                    inactive = int(line.split(":")[1].strip().rstrip("."))
            if free or inactive:
                return (free + inactive) * page
        except Exception:
            pass
    return 4 * 1024 ** 3


class NEPCalculator:
    """NEP4 calculator: loads a trained model and computes atomic properties.

    Parameters
    ----------
    model_file : str
        Path to nep.txt (GPUMD NEP4 format).
    dtype : torch.dtype
        Precision (default: float64 for accuracy).
    device : str or torch.device
        Compute device (default: 'cpu').
    """

    def __init__(self, model_file: str, dtype=torch.float64, device="cpu"):
        self.dtype = dtype
        self.device = torch.device(device)
        self._load_model(model_file)

    def _load_model(self, path: str):
        with open(path) as f:
            lines = f.readlines()

        idx = 0

        # Line 1: version and types
        header = lines[idx].split()
        version_str = header[0]  # "nep4", "nep4_zbl", etc.
        self.has_zbl = "zbl" in version_str
        self.num_types = int(header[1])
        self.type_names = header[2 : 2 + self.num_types]
        # Real atomic numbers (Z = 1 for H, 6 for C, ...). ELEMENTS is
        # 0-indexed, so we add 1. ZBL uses these directly (Z*Z', Z^0.23).
        self.atomic_numbers = [ELEMENTS.index(n) + 1 for n in self.type_names]
        idx += 1

        # ZBL line
        if self.has_zbl:
            parts = lines[idx].split()
            self.zbl_rc_inner = float(parts[1])
            self.zbl_rc_outer = float(parts[2])
            self.zbl_typewise_factor = float(parts[3]) if len(parts) > 3 else None
            idx += 1

            if self.zbl_typewise_factor is not None:
                # Per-type cutoffs. COVALENT_RADIUS is 0-indexed (H at 0),
                # while self.atomic_numbers holds real Z (H=1), hence z-1.
                self.zbl_rc_inner_per_type = torch.tensor(
                    [self.zbl_typewise_factor * COVALENT_RADIUS[z - 1]
                     for z in self.atomic_numbers],
                    dtype=self.dtype, device=self.device)
                self.zbl_rc_outer_per_type = 2.0 * self.zbl_rc_inner_per_type
        else:
            self.zbl_rc_inner = None
            self.zbl_rc_outer = None
            self.zbl_typewise_factor = None

        # Cutoff
        parts = lines[idx].split()
        self.rc_radial = float(parts[1])
        self.rc_angular = float(parts[2])
        idx += 1

        # n_max, basis_size, l_max
        parts = lines[idx].split()
        self.n_max_radial = int(parts[1])
        self.n_max_angular = int(parts[2])
        idx += 1

        parts = lines[idx].split()
        self.basis_size_radial = int(parts[1])
        self.basis_size_angular = int(parts[2])
        idx += 1

        parts = lines[idx].split()
        self.l_max_3b   = int(parts[1])
        self.has_q_222  = 1 if (len(parts) > 2 and int(parts[2]) > 0) else 0
        self.has_q_1111 = 1 if (len(parts) > 3 and int(parts[3]) > 0) else 0
        self.has_q_112  = 1 if (len(parts) > 4 and int(parts[4]) > 0) else 0
        self.has_q_123  = 1 if (len(parts) > 5 and int(parts[5]) > 0) else 0
        self.has_q_233  = 1 if (len(parts) > 6 and int(parts[6]) > 0) else 0
        self.has_q_134  = 1 if (len(parts) > 7 and int(parts[7]) > 0) else 0
        idx += 1

        # ANN
        parts = lines[idx].split()
        self.num_neurons = int(parts[1])
        idx += 1

        # Descriptor dimension — order must match GPUMD save layout:
        # radial -> 3-body -> q_222 -> q_1111 -> q_112 -> q_123 -> q_233 -> q_134.
        n_ap1 = self.n_max_angular + 1
        self.dim_radial = self.n_max_radial + 1
        self.dim_angular_3b   = n_ap1 * self.l_max_3b
        self.dim_angular_4b   = n_ap1 if self.has_q_222  else 0
        self.dim_angular_5b   = n_ap1 if self.has_q_1111 else 0
        self.dim_angular_112  = n_ap1 if self.has_q_112  else 0
        self.dim_angular_123  = n_ap1 if self.has_q_123  else 0
        self.dim_angular_233  = n_ap1 if self.has_q_233  else 0
        self.dim_angular_134  = n_ap1 if self.has_q_134  else 0
        self.dim = (self.dim_radial + self.dim_angular_3b +
                    self.dim_angular_4b + self.dim_angular_5b +
                    self.dim_angular_112 +
                    self.dim_angular_123 + self.dim_angular_233 +
                    self.dim_angular_134)
        self.num_lm = sum(2 * ll + 1 for ll in range(1, self.l_max_3b + 1))

        # Parse data
        data = [float(l) for l in lines[idx:] if l.strip()]
        di = 0
        n = self.num_neurons
        d = self.dim

        # Per-type NN weights
        self.w0, self.b0, self.w1 = [], [], []
        for t in range(self.num_types):
            w0 = np.array(data[di:di + n * d]).reshape(n, d)
            di += n * d
            b0 = np.array(data[di:di + n])
            di += n
            w1 = np.array(data[di:di + n])
            di += n
            self.w0.append(torch.tensor(w0, dtype=self.dtype, device=self.device))
            self.b0.append(torch.tensor(b0, dtype=self.dtype, device=self.device))
            self.w1.append(torch.tensor(w1, dtype=self.dtype, device=self.device))

        # Common output bias
        self.b1 = torch.tensor(data[di], dtype=self.dtype, device=self.device)
        di += 1

        # c parameters
        nt2 = self.num_types ** 2
        c2_size = (self.n_max_radial + 1) * (self.basis_size_radial + 1) * nt2
        c2 = np.array(data[di:di + c2_size]).reshape(
            self.n_max_radial + 1, self.basis_size_radial + 1,
            self.num_types, self.num_types)
        # save_nep_txt stores c2 transposed as (n_max+1, basis+1, nt, nt);
        # ops.compute_descriptors expects (nt, nt, n_max+1, basis+1).
        self.c2 = torch.tensor(c2, dtype=self.dtype, device=self.device).permute(2, 3, 0, 1).contiguous()
        di += c2_size

        if self.l_max_3b > 0:
            c3_size = (self.n_max_angular + 1) * (self.basis_size_angular + 1) * nt2
            c3 = np.array(data[di:di + c3_size]).reshape(
                self.n_max_angular + 1, self.basis_size_angular + 1,
                self.num_types, self.num_types)
            self.c3 = torch.tensor(c3, dtype=self.dtype, device=self.device).permute(2, 3, 0, 1).contiguous()
            di += c3_size

        # q_scaler
        self.q_scaler = torch.tensor(
            data[di:di + self.dim], dtype=self.dtype, device=self.device)
        di += self.dim

        # Pre-build constants
        self._c3b  = torch.tensor(C3B[:self.num_lm], dtype=self.dtype, device=self.device)
        self._c4b  = torch.tensor(C4B,  dtype=self.dtype, device=self.device)
        self._c5b  = torch.tensor(C5B,  dtype=self.dtype, device=self.device)
        self._c4b2 = torch.tensor(C4B2, dtype=self.dtype, device=self.device)

    def compute(
        self,
        species: list,
        positions: np.ndarray,
        cell: np.ndarray,
        compute_descriptor: bool = False,
        return_components: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute energy, forces, and per-atom virial.

        Parameters
        ----------
        species : list of str
            Element symbols for each atom.
        positions : (N, 3) array
            Atomic positions in Angstrom.
        cell : (3, 3) array
            Lattice vectors (row-major).
        compute_descriptor : bool
            Also return scaled descriptors.
        return_components : bool
            Also split the result into the NEP (neural-network) part and the
            ZBL repulsive part.  Adds ``energy_nep / forces_nep / virial_nep``
            and ``energy_zbl / forces_zbl / virial_zbl`` to the output; the
            plain ``energy / forces / virial`` keys remain the NEP+ZBL total
            (their exact sum).  For models without ZBL the ZBL part is zero.

        Returns
        -------
        dict with 'energy', 'forces', 'virial', optionally 'descriptor' and
        (if ``return_components``) the per-contribution breakdown above.
        """
        atom_types = torch.tensor(
            [self.type_names.index(s) for s in species],
            dtype=torch.long, device=self.device)
        pos_np = np.asarray(positions)
        cell_np = np.asarray(cell)
        N = pos_np.shape[0]

        # Neighbor list: PyTorch linked-cell search (O(N), stays on device) for
        # MD-sized cells, with an automatic fallback to the numpy brute-force
        # builder for tiny training-style boxes. See torchnep.neighbor.
        max_rc = max(self.rc_radial, self.rc_angular)
        pair_i, pair_j, rij = build_neighbor_list(
            pos_np, cell_np, max_rc, device=self.device, dtype=self.dtype)
        dij = torch.norm(rij, dim=-1)

        rad_mask = dij < self.rc_radial
        ang_mask = dij < self.rc_angular

        rij_rad = rij[rad_mask].detach().requires_grad_(True)
        rij_ang = rij[ang_mask].detach().requires_grad_(True)
        pi_rad, pj_rad = pair_i[rad_mask], pair_j[rad_mask]
        pi_ang, pj_ang = pair_i[ang_mask], pair_j[ang_mask]

        q = ops.compute_descriptors(
            rij_rad, rij_ang, pi_rad, pj_rad, pi_ang, pj_ang,
            atom_types, N, self.c2, self.c3,
            self.rc_radial, self.rc_angular,
            self.basis_size_radial, self.basis_size_angular,
            self.n_max_radial, self.n_max_angular,
            self.l_max_3b,
            self.has_q_222, self.has_q_1111, self.has_q_112,
            self.num_lm, self._c3b, self._c4b, self._c5b,
            self._c4b2,
            self.dtype, self.device,
            has_q_123=self.has_q_123, has_q_233=self.has_q_233,
            has_q_134=self.has_q_134,
        )

        descriptor = (q * self.q_scaler).detach() if compute_descriptor else None
        q_scaled = q * self.q_scaler

        Ei_nep = ops.apply_ann(q_scaled, atom_types, self.num_types,
                               self.w0, self.b0, self.w1, self.b1,
                               self.dtype, self.device)

        # ZBL repulsive correction (depends only on angular pairs)
        Ei_zbl = None
        if self.has_zbl:
            Ei_zbl = ops.compute_zbl(
                atom_types, pi_ang, pj_ang, rij_ang, N,
                self.atomic_numbers, self.zbl_rc_inner, self.zbl_rc_outer,
                self.zbl_typewise_factor,
                getattr(self, "zbl_rc_inner_per_type", None),
                getattr(self, "zbl_rc_outer_per_type", None),
                self.dtype, self.device,
            )

        Ei_total = Ei_nep if Ei_zbl is None else Ei_nep + Ei_zbl

        # Force & virial of an energy term, via autograd on the pair vectors.
        def _force_virial(energy_sum, retain_graph):
            # A term with no grad history (e.g. ZBL when every pair is beyond
            # its cutoff) contributes zero force/virial — autograd would raise.
            if not energy_sum.requires_grad:
                return (torch.zeros((N, 3), dtype=self.dtype, device=self.device),
                        torch.zeros((N, 9), dtype=self.dtype, device=self.device))
            grads = torch.autograd.grad(
                energy_sum, [rij_rad, rij_ang],
                allow_unused=True, retain_graph=retain_graph)
            g_rad = grads[0] if grads[0] is not None else torch.zeros_like(rij_rad)
            g_ang = grads[1] if grads[1] is not None else torch.zeros_like(rij_ang)
            return ops.accumulate_forces_virial(
                N, pi_rad, pj_rad, rij_rad.detach(), g_rad.detach(),
                pi_ang, pj_ang, rij_ang.detach(), g_ang.detach(),
                self.dtype, self.device,
            )

        if not return_components:
            forces, virial = _force_virial(Ei_total.sum(), retain_graph=False)
            result = {"energy": Ei_total.detach(), "forces": forces,
                      "virial": virial}
        else:
            if Ei_zbl is None:
                # No ZBL: NEP == total; ZBL part is identically zero.
                f_nep, v_nep = _force_virial(Ei_nep.sum(), retain_graph=False)
                f_zbl = torch.zeros_like(f_nep)
                v_zbl = torch.zeros_like(v_nep)
                e_zbl = torch.zeros_like(Ei_nep)
            else:
                # Two backward passes; the totals are the (exact) component sums.
                f_nep, v_nep = _force_virial(Ei_nep.sum(), retain_graph=True)
                f_zbl, v_zbl = _force_virial(Ei_zbl.sum(), retain_graph=False)
                e_zbl = Ei_zbl
            result = {
                "energy": Ei_total.detach(),
                "forces": f_nep + f_zbl, "virial": v_nep + v_zbl,
                "energy_nep": Ei_nep.detach(),
                "forces_nep": f_nep, "virial_nep": v_nep,
                "energy_zbl": e_zbl.detach(),
                "forces_zbl": f_zbl, "virial_zbl": v_zbl,
            }
        if descriptor is not None:
            result["descriptor"] = descriptor
        return result

    def compute_batch(self, batch: Dict, backend: str = "loop") -> Dict:
        """Compute energy, forces, virial for a pre-built batch dict.

        The batch dict must contain pre-cached basis tensors on the device:
        fk_rad, fkp_rad, d12inv_rad, fk_ang, fkp_ang, d12inv_ang, blm,
        pair_i_rad, pair_j_rad, rij_rad, pair_i_ang, pair_j_ang, rij_ang,
        atom_types, struct_idx, N, num_structures.

        ``backend`` in {"loop", "bmm"} — see ops.resolve_backend.
        """
        dtype, device = self.dtype, self.device
        N = batch["N"]

        # Descriptors from pre-cached basis tensors
        q, s, gn_ang = ops.compute_descriptors_cached(
            batch["fk_rad"], batch["fk_ang"], batch["blm"],
            batch["pair_i_rad"], batch["pair_j_rad"],
            batch["pair_i_ang"], batch["pair_j_ang"],
            batch["atom_types"], N,
            self.c2, getattr(self, "c3", None),
            self.n_max_radial, self.n_max_angular,
            self.l_max_3b,
            self.has_q_222, self.has_q_1111, self.has_q_112,
            self.num_lm, self._c3b, self._c4b, self._c5b,
            self._c4b2,
            dtype, device,
            return_intermediates=True,
            backend=backend,
            has_q_123=self.has_q_123, has_q_233=self.has_q_233,
            has_q_134=self.has_q_134,
        )

        q_scaled = q * self.q_scaler

        # NN forward: Ei and Fp = dEi/dq_scaled
        Ei = torch.zeros(N, dtype=dtype, device=device)
        Fp = torch.zeros(N, self.dim, dtype=dtype, device=device)
        for t in range(self.num_types):
            mask = batch["atom_types"] == t
            if not mask.any():
                continue
            qt = q_scaled[mask]
            # w0[t]: (neurons, dim), b0[t]: (neurons,), w1[t]: (neurons,)
            z = qt @ self.w0[t].T - self.b0[t]
            h = torch.tanh(z)
            Ei[mask] = h @ self.w1[t]
            tanh_der = 1.0 - h * h
            Fp[mask] = (self.w1[t] * tanh_der) @ self.w0[t]

        Fp = Fp * self.q_scaler
        Ei = Ei - self.b1

        # ZBL correction (energy + forces/virial via local autograd on rij_ang).
        # enable_grad: predict_dataset wraps this call in torch.no_grad(),
        # but ZBL forces need a local autograd pass.
        zbl_forces = None
        zbl_virial = None
        if self.has_zbl:
            with torch.enable_grad():
                rij_zbl = batch["rij_ang"].detach().requires_grad_(True)
                Ei_zbl = ops.compute_zbl(
                    batch["atom_types"], batch["pair_i_ang"],
                    batch["pair_j_ang"], rij_zbl, N,
                    self.atomic_numbers,
                    self.zbl_rc_inner, self.zbl_rc_outer,
                    self.zbl_typewise_factor,
                    getattr(self, "zbl_rc_inner_per_type", None),
                    getattr(self, "zbl_rc_outer_per_type", None),
                    dtype, device,
                )
                if Ei_zbl.requires_grad:
                    g_zbl = torch.autograd.grad(
                        Ei_zbl.sum(), rij_zbl, allow_unused=True)[0]
                else:
                    g_zbl = None
            Ei = Ei + Ei_zbl.detach()
            if g_zbl is not None:
                empty_i = torch.zeros(0, dtype=torch.long, device=device)
                empty_r = torch.zeros(0, 3, dtype=dtype, device=device)
                zbl_forces, zbl_virial = ops.accumulate_forces_virial(
                    N, empty_i, empty_i, empty_r, empty_r,
                    batch["pair_i_ang"], batch["pair_j_ang"],
                    batch["rij_ang"].detach(), g_zbl.detach(),
                    dtype, device,
                )

        # Per-structure totals
        Etot = torch.zeros(batch["num_structures"], dtype=dtype, device=device)
        Etot.scatter_add_(0, batch["struct_idx"], Ei)

        # Analytical forces + virial
        forces, virial = ops.compute_analytical_forces(
            Fp, batch["atom_types"], N,
            self.c2, getattr(self, "c3", None),
            batch["fkp_rad"], batch["fkp_ang"], batch["blm"],
            batch["pair_i_rad"], batch["pair_j_rad"],
            batch["rij_rad"], batch["d12inv_rad"],
            batch["pair_i_ang"], batch["pair_j_ang"],
            batch["rij_ang"], batch["d12inv_ang"],
            s, gn_ang,
            self.n_max_radial, self.n_max_angular,
            self.l_max_3b,
            self.has_q_222, self.has_q_1111, self.has_q_112,
            self.num_lm, self._c3b, self._c4b, self._c5b,
            self._c4b2,
            dtype, device,
            compute_virial=True,
            backend=backend,
            has_q_123=self.has_q_123, has_q_233=self.has_q_233,
            has_q_134=self.has_q_134,
        )
        if zbl_forces is not None:
            forces = forces + zbl_forces
            if zbl_virial is not None:
                virial = virial + zbl_virial

        return {"Ei": Ei, "Etot": Etot, "forces": forces, "virial": virial,
                "descriptor": q_scaled.detach()}

    def get_descriptor(self, species, positions, cell):
        """Compute scaled descriptors. Returns (N, dim) numpy."""
        return self.compute(species, positions, cell,
                            compute_descriptor=True)["descriptor"].cpu().numpy()

    # -- Memory-bounded tiled inference (large MD cells) ----------------------
    def _auto_block_size(self, N, cell, reserve_gb=1.5, copy_fudge=1.3):
        """Pick the largest block whose extra allocation fits the device's free
        memory minus ``reserve_gb`` — bigger blocks mean fewer tiles, less
        overhead, faster. The reserve (default 2 GB) is the only headroom kept;
        everything else is used.

        Memory model (fit to measured peak RSS — CrCoNi-ZBL and 512k carbon,
        float32): the *extra* allocation during compute is
        ``n_scratch + per_atom * block_size``. The block term is dominated NOT by
        the neighbor list (~150 MB; transfer is ~3 ms) but by the three
        (P_ang, n_ap1, num_lm) angular intermediates in the descriptor / force
        kernels (gn_blm, gnp_blm, w_i), then the radial basis and dblm_dhat.
        ``copy_fudge`` (1.3) matches the measured 512k-carbon per-atom cost
        (~535 KB/atom, float32). ``_available_*`` is current free memory, so the
        process baseline is already excluded; only ``reserve_gb`` is kept free.
        """
        esize = torch.finfo(self.dtype).bits // 8
        vol = abs(float(np.linalg.det(np.asarray(cell, dtype=float))))
        density = N / vol if vol > 1e-9 else 0.05
        nbr_rad = density * (4.0 / 3.0) * math.pi * self.rc_radial ** 3
        nbr_ang = density * (4.0 / 3.0) * math.pi * self.rc_angular ** 3
        nap1, nlm = self.n_max_angular + 1, self.num_lm
        # Per-block-atom bytes:
        #   3x (n_ap1*num_lm) angular intermediates  <- dominant
        #   dblm_dhat (num_lm*3) + blm (num_lm) + 2x gn factors (n_ap1)
        #   radial basis fk_r+fkp_r (2*(basis_r+1)) + neighbor rij/idx (~7 per pair)
        per_atom = (nbr_ang * (3 * nap1 * nlm + nlm * 3 + nlm + 2 * nap1)
                    + nbr_rad * (2 * (self.basis_size_radial + 1) + 7)) * esize * copy_fudge
        # N-sized scratch (q, Fp, s, w_atom, force/virial accumulators) — fixed.
        n_scratch = N * (2 * self.dim + 2 * nap1 * nlm + 12) * esize * copy_fudge
        budget = _available_memory_bytes(self.device) - reserve_gb * 1e9
        # Always allow at least a small block, even on a tight machine.
        budget = max(budget, 256.0 * per_atom + n_scratch)
        B = int((budget - n_scratch) / max(per_atom, 1.0))
        return max(256, min(B, N))

    def compute_tiled(
        self,
        species: list,
        positions: np.ndarray,
        cell: np.ndarray,
        block_size="auto",
        query_sub_chunk: int = 4000,
        backend: str = "auto",
        compile: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Energy / forces / virial for large cells with bounded peak memory.

        This is the MD-oriented counterpart of :meth:`compute`. Instead of
        building the whole neighbor list and back-propagating through it (which
        scales the peak memory with the total pair count — infeasible for a
        500k-atom, high-density cell), it:

        * builds the spatial bin table once (:class:`~torchnep.neighbor.CellList`);
        * streams over blocks of ``block_size`` centre atoms, querying only that
          block's neighbors at a time;
        * computes per-atom energy with the **analytical** force/virial path
          (``ops.compute_analytical_forces``) — no autograd graph is retained, so
          peak memory is set by a single block, not the whole system. (This is
          the same analytical kernel training/prediction use, hence also
          ``torch.compile``-friendly.)

        Each directed pair is owned by exactly one block (the block holding its
        centre atom ``i``); contributions scatter into global per-atom
        accumulators, so the result is identical to :meth:`compute` to round-off.

        For cells too small for the linked-cell decomposition this transparently
        defers to :meth:`compute` (the autograd path is cheap there).

        Parameters
        ----------
        species, positions, cell : as in :meth:`compute`.
        block_size : int or "auto"
            Number of centre atoms processed per tile. Larger = faster but more
            peak memory (block-independent N-sized scratch + this block's pairs).
            "auto" (default) sizes it from the cell density, model dims, dtype
            and the device's free memory (~35% budget); pass an int to override.
        query_sub_chunk : int
            Inner chunk for the neighbor query (bounds the candidate tensor).
        backend : "auto" | "loop" | "bmm"  type-pair contraction (see ops).
        compile : bool
            torch.compile the descriptor + analytical-force kernels (dynamic
            shapes, so the pair-count changes between MD steps do NOT trigger
            recompilation — one warm-up compile, then a single dynamic graph).
            Forces the ``bmm`` backend (compile-friendly; the per-type Python
            loop graph-breaks). Worth it for many steps / large cells.

        Returns
        -------
        dict with 'energy' (N,), 'forces' (N, 3), 'virial' (N, 9) — same keys and
        conventions as :meth:`compute` (sans the component split / descriptor).
        """
        dtype, device = self.dtype, self.device
        N = len(species)
        atom_types = torch.tensor(
            [self.type_names.index(s) for s in species],
            dtype=torch.long, device=device)

        max_rc = max(self.rc_radial, self.rc_angular)
        cl = CellList(positions, cell, max_rc, device=device)
        if not cl.ok:
            # Small box — the autograd path is cheap and exact here.
            res = self.compute(species, positions, cell)
            return {"energy": res["energy"], "forces": res["forces"],
                    "virial": res["virial"]}

        if block_size == "auto":
            block_size = self._auto_block_size(N, cell)
        block_size = int(block_size)

        if compile and self.device.type == "mps":
            # torch.compile's inductor backend has no MPS support — fall back
            # to eager rather than crashing mid-MD.
            import warnings
            warnings.warn("torch.compile is unsupported on MPS; running eager.",
                          RuntimeWarning)
            compile = False
        if compile:
            # bmm backend is compile-friendly (no per-type Python loop). Compile
            # once and cache on the instance so MD steps reuse the dynamic graph.
            backend = "bmm"
            if not hasattr(self, "_desc_compiled"):
                self._desc_compiled = torch.compile(
                    ops.compute_descriptors_cached, dynamic=True)
                self._force_compiled = torch.compile(
                    ops.compute_analytical_forces, dynamic=True)
            desc_fn, force_fn = self._desc_compiled, self._force_compiled
        else:
            backend = ops.resolve_backend(backend, num_types=self.num_types)
            desc_fn, force_fn = (ops.compute_descriptors_cached,
                                 ops.compute_analytical_forces)
        c2 = self.c2
        c3 = getattr(self, "c3", None)

        E_atom = torch.zeros(N, dtype=dtype, device=device)
        forces = torch.zeros(N, 3, dtype=dtype, device=device)
        virial = torch.zeros(N, 9, dtype=dtype, device=device)

        for a in range(0, N, block_size):
            blk = torch.arange(a, min(a + block_size, N), device=device)
            pi, pj, rij = cl.query(blk, sub_chunk=query_sub_chunk)
            rij = rij.to(dtype)
            dij = torch.norm(rij, dim=-1)

            rad = dij < self.rc_radial
            ang = dij < self.rc_angular
            pir, pjr, rij_r, dr = pi[rad], pj[rad], rij[rad], dij[rad]
            pia, pja, rij_a, da = pi[ang], pj[ang], rij[ang], dij[ang]

            # Cached basis for this block's pairs (analytical-force inputs).
            fk_r, fkp_r = ops.chebyshev_basis_and_deriv(
                dr, self.rc_radial, self.basis_size_radial)
            d12inv_r = 1.0 / dr.clamp(min=1e-10)
            if rij_a.shape[0] > 0:
                fk_a, fkp_a = ops.chebyshev_basis_and_deriv(
                    da, self.rc_angular, self.basis_size_angular)
                d12inv_a = 1.0 / da.clamp(min=1e-10)
                blm = ops.angular_basis(rij_a[:, 0] * d12inv_a,
                                        rij_a[:, 1] * d12inv_a,
                                        rij_a[:, 2] * d12inv_a, self.l_max_3b)
            else:
                fk_a = torch.zeros(0, self.basis_size_angular + 1, dtype=dtype, device=device)
                fkp_a = torch.zeros(0, self.basis_size_angular + 1, dtype=dtype, device=device)
                d12inv_a = torch.zeros(0, dtype=dtype, device=device)
                blm = torch.zeros(0, self.num_lm, dtype=dtype, device=device)

            # Descriptors (scatter into N-sized buffers; only block-centre rows
            # are complete, which is all we read).
            q, s, gn_ang = desc_fn(
                fk_r, fk_a, blm, pir, pjr, pia, pja,
                atom_types, N, c2, c3,
                self.n_max_radial, self.n_max_angular, self.l_max_3b,
                self.has_q_222, self.has_q_1111, self.has_q_112,
                self.num_lm, self._c3b, self._c4b, self._c5b, self._c4b2,
                dtype, device, return_intermediates=True, backend=backend,
                has_q_123=self.has_q_123, has_q_233=self.has_q_233,
                has_q_134=self.has_q_134)

            # NN forward — block centres only.
            q_blk = q[blk] * self.q_scaler
            types_blk = atom_types[blk]
            Ei_blk = torch.zeros(blk.shape[0], dtype=dtype, device=device)
            Fp_blk = torch.zeros(blk.shape[0], self.dim, dtype=dtype, device=device)
            for t in range(self.num_types):
                m = types_blk == t
                if not m.any():
                    continue
                qt = q_blk[m]
                h = torch.tanh(qt @ self.w0[t].T - self.b0[t])
                Ei_blk[m] = h @ self.w1[t]
                Fp_blk[m] = (self.w1[t] * (1.0 - h * h)) @ self.w0[t]
            E_atom[blk] = Ei_blk - self.b1

            Fp = torch.zeros(N, self.dim, dtype=dtype, device=device)
            Fp[blk] = Fp_blk * self.q_scaler

            # ZBL (local autograd on this block's angular pairs — bounded).
            if self.has_zbl:
                with torch.enable_grad():
                    rij_zbl = rij_a.detach().requires_grad_(True)
                    Ei_zbl = ops.compute_zbl(
                        atom_types, pia, pja, rij_zbl, N,
                        self.atomic_numbers, self.zbl_rc_inner, self.zbl_rc_outer,
                        self.zbl_typewise_factor,
                        getattr(self, "zbl_rc_inner_per_type", None),
                        getattr(self, "zbl_rc_outer_per_type", None),
                        dtype, device)
                    g_zbl = (torch.autograd.grad(Ei_zbl.sum(), rij_zbl,
                                                 allow_unused=True)[0]
                             if Ei_zbl.requires_grad else None)
                E_atom[blk] = E_atom[blk] + Ei_zbl[blk].detach()
                if g_zbl is not None:
                    empty_i = torch.zeros(0, dtype=torch.long, device=device)
                    empty_r = torch.zeros(0, 3, dtype=dtype, device=device)
                    zf, zv = ops.accumulate_forces_virial(
                        N, empty_i, empty_i, empty_r, empty_r,
                        pia, pja, rij_a.detach(), g_zbl.detach(), dtype, device)
                    forces += zf
                    virial += zv

            # Analytical NEP forces + virial for this block's pairs.
            f_blk, v_blk = force_fn(
                Fp, atom_types, N, c2, c3, fkp_r, fkp_a, blm,
                pir, pjr, rij_r, d12inv_r, pia, pja, rij_a, d12inv_a,
                s, gn_ang,
                self.n_max_radial, self.n_max_angular, self.l_max_3b,
                self.has_q_222, self.has_q_1111, self.has_q_112,
                self.num_lm, self._c3b, self._c4b, self._c5b, self._c4b2,
                dtype, device, compute_virial=True, backend=backend,
                has_q_123=self.has_q_123, has_q_233=self.has_q_233,
                has_q_134=self.has_q_134)
            forces += f_blk
            virial += v_blk

        return {"energy": E_atom, "forces": forces, "virial": virial}
