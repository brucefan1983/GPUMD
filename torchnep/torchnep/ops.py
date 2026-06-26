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
Core NEP operations — pure PyTorch on CPU / CUDA / MPS.
"""

import torch
from typing import List, Literal, Optional, Tuple

from .constants import (PI, K_C_SP, ZBL_PARA, Z_COEFFICIENT, MAX_L3B,
                        Q123_TERMS, Q233_TERMS, Q134_TERMS)


# ---------------------------------------------------------------------------
# Backend selection
#
# Two concrete implementations of the type-pair contraction plus one meta:
#   "loop" : pure PyTorch, nested for-loop over (t1, t2) type pairs. Wins when
#            ntypes is small (<= ~5) — the outer loop runs few times.
#   "bmm"  : pure PyTorch, fancy index + torch.bmm (dispatched to cuBLAS on
#            CUDA, MKL on CPU, MPS on Apple). Wins when ntypes >= ~8 — one
#            batched GEMM replaces the O(ntypes**2) python-level loop.
#   "auto" : picks by num_types (>= 8 -> bmm, else loop).
#
# Both backends are autograd-compatible and run on any PyTorch backend.
# ---------------------------------------------------------------------------

Backend = Literal["auto", "loop", "bmm"]


def resolve_backend(backend: str = "auto",
                    num_types: Optional[int] = None,
                    use_compile: bool = False) -> str:
    """Resolve "auto" into a concrete backend.

    under torch.compile -> "bmm"   (vectorised path fuses far better; the per-
                                    type Python loop forces graph breaks, so bmm
                                    is consistently fastest once compiled)
    ntypes >= 8         -> "bmm"   (fancy-index + batched GEMM wins)
    otherwise           -> "loop"  (few-types eager; inline Python loop fastest)

    Any non-"auto" string is returned unchanged (explicit override wins).
    """
    if backend != "auto":
        return backend
    if use_compile:
        return "bmm"
    if num_types is not None and num_types >= 8:
        return "bmm"
    return "loop"


def _select_contraction_funcs(backend: str):
    """Return (scatter_fn, type_fn) for the concrete backend."""
    if backend == "bmm":
        return _scatter_contraction_bmm, _type_contraction_bmm
    # "loop" — default
    return _scatter_contraction_loop, _type_contraction_loop


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------

def chebyshev_basis(dij: torch.Tensor, rc: float,
                    basis_size: int) -> torch.Tensor:
    """Chebyshev radial basis: f_k(r) = 0.5*(T_k(x)+1)*fc(r).

    Parameters
    ----------
    dij : (P,) float
        Pair distances in A. P is the total number of pairs in the batch
        (so this works for one frame, many frames, or zero pairs).
    rc : float
        Radial cutoff in A. Basis is designed so f_k(rc) = 0.
    basis_size : int
        Polynomial order (== NEP ``basis_size`` parameter).

    Returns
    -------
    fk : (P, basis_size + 1) float
        Chebyshev basis values, fk[p, k] = 0.5*(T_k(x_p)+1)*fc(r_p).
    """
    rcinv = 1.0 / rc
    fc = 0.5 * torch.cos(PI * dij * rcinv) + 0.5
    x = 2.0 * (dij * rcinv - 1.0) ** 2 - 1.0

    T = [torch.ones_like(dij), x]
    for k in range(2, basis_size + 1):
        T.append(2.0 * x * T[-1] - T[-2])
    T = torch.stack(T, dim=-1)
    return 0.5 * (T + 1.0) * fc.unsqueeze(-1)


def chebyshev_basis_and_deriv(dij: torch.Tensor, rc: float,
                              basis_size: int):
    """Compute Chebyshev basis AND its derivative wrt distance.

    Parameters
    ----------
    dij : (P,) float
        Pair distances in A.
    rc : float
        Radial cutoff in A.
    basis_size : int
        Polynomial order.

    Returns
    -------
    fk  : (P, basis_size + 1) float — same values as ``chebyshev_basis``.
    fkp : (P, basis_size + 1) float — d(fk[k])/d(rij) at each pair distance.

    Writes directly into a preallocated (P, basis_size+1) output buffer and
    keeps only 4 sliding-window scalars (T_{k-2}, T_{k-1}, U_{k-2}, U_{k-1})
    in memory, instead of materializing all 2*(basis_size+1) intermediate
    tensors + a final torch.stack copy. On 19M pairs this avoids ~2 GB of
    GPU->GPU traffic per call.
    """
    rcinv = 1.0 / rc
    arg = PI * dij * rcinv
    fc = 0.5 * torch.cos(arg) + 0.5
    fcp = -0.5 * PI * rcinv * torch.sin(arg)
    dij_m1 = dij * rcinv - 1.0
    x = 2.0 * dij_m1 * dij_m1 - 1.0
    dxdr = 4.0 * dij_m1 * rcinv

    P = dij.shape[0]
    B = basis_size + 1
    fk  = torch.empty(P, B, dtype=dij.dtype, device=dij.device)
    fkp = torch.empty(P, B, dtype=dij.dtype, device=dij.device)

    # k = 0: T_0 = 1, so fn_core = 1 -> fk[..,0] = fc; fkp[..,0] = fcp
    fk[:, 0] = fc
    fkp[:, 0] = fcp

    if B >= 2:
        # k = 1: T_1 = x, U_0 = 1 -> dT_1/dx = 1
        fn_core1 = 0.5 * (x + 1.0)
        fk[:, 1] = fn_core1 * fc
        fkp[:, 1] = 0.5 * dxdr * fc + fn_core1 * fcp

    # Sliding window: T_prev2 = T_{k-2}, T_prev1 = T_{k-1},
    #                 U_prev2 = U_{k-2}, U_prev1 = U_{k-1}
    T_prev1 = x         # T_1
    U_prev1 = 2.0 * x   # U_1
    T_prev2 = torch.ones_like(dij)   # T_0
    U_prev2 = torch.ones_like(dij)   # U_0

    for k in range(2, B):
        T_next = 2.0 * x * T_prev1 - T_prev2     # T_k
        U_next = 2.0 * x * U_prev1 - U_prev2     # U_k
        fn_core = 0.5 * (T_next + 1.0)
        fk[:, k] = fn_core * fc
        # dT_k/dx = k * U_{k-1} = k * U_prev1  (before we shift)
        fkp[:, k] = 0.5 * (k * U_prev1) * dxdr * fc + fn_core * fcp
        T_prev2, T_prev1 = T_prev1, T_next
        U_prev2, U_prev1 = U_prev1, U_next

    return fk, fkp


def _build_xy_powers(x: torch.Tensor, y: torch.Tensor, n_max: int):
    """Real and imaginary parts of (x + iy)^n for n = 0..n_max.

    Returns two lists of tensors (each shape (P,)): Re[n], Im[n].
    """
    Re = [torch.ones_like(x)]
    Im = [torch.zeros_like(x)]
    for _ in range(n_max):
        r_prev, i_prev = Re[-1], Im[-1]
        # (x + iy) * (r + ii) = (xr - yi) + i(xi + yr)
        Re.append(x * r_prev - y * i_prev)
        Im.append(x * i_prev + y * r_prev)
    return Re, Im


def _build_z_powers(z: torch.Tensor, n_max: int):
    """z^0..z^n_max as a list of tensors."""
    zp = [torch.ones_like(z)]
    for _ in range(n_max):
        zp.append(zp[-1] * z)
    return zp


def _z_factor(z_pow, coeff_row, start, stop):
    """Sum_{n2 = start, start+2, ..., stop-1} coeff_row[n2] * z^n2.

    Skips zero coefficients to save ops. ``start`` matches (L+n1) parity.
    """
    out = None
    n2 = start
    while n2 < stop:
        c = coeff_row[n2]
        if c != 0.0:
            term = z_pow[n2] if c == 1.0 else (c * z_pow[n2])
            out = term if out is None else out + term
        n2 += 2
    if out is None:
        return torch.zeros_like(z_pow[0])
    return out


def angular_basis(x: torch.Tensor, y: torch.Tensor,
                  z: torch.Tensor, l_max_3b: int) -> torch.Tensor:
    r"""Solid-harmonics-style angular basis Y_{Ln1} for L = 1..l_max_3b.

    Each basis element is z_factor(z) * {Re or Im}[(x + iy)^n1], using the
    polynomial coefficients in ``Z_COEFFICIENT``. Bit-identical to the
    previous hand-coded formulas for l_max_3b <= 4.

    Parameters
    ----------
    x, y, z : (P,) float
        Components of the unit vector rij / |rij| for each angular pair.
    l_max_3b : int
        Max angular momentum (NEP ``l_max`` first entry).

    Returns
    -------
    blm : (P, num_lm) float, where num_lm = \Sigma_{L=1..l_max_3b} (2L + 1).
    """
    if l_max_3b < 1:
        return torch.zeros(x.shape[0], 0, dtype=x.dtype, device=x.device)
    if l_max_3b > MAX_L3B:
        raise ValueError(f"l_max_3b={l_max_3b} exceeds MAX_L3B={MAX_L3B}")

    z_pow = _build_z_powers(z, l_max_3b)
    Re, Im = _build_xy_powers(x, y, l_max_3b)

    blm = []
    for L in range(1, l_max_3b + 1):
        Z = Z_COEFFICIENT[L]
        for n1 in range(L + 1):
            parity = (L + n1) % 2
            start = parity               # 0 if L+n1 even, 1 if odd
            stop = L - n1 + 1
            zf = _z_factor(z_pow, Z[n1], start, stop)
            if n1 == 0:
                blm.append(zf)
            else:
                blm.append(zf * Re[n1])
                blm.append(zf * Im[n1])
    return torch.stack(blm, dim=-1)


# ---------------------------------------------------------------------------
# Descriptor computation
# ---------------------------------------------------------------------------

def compute_descriptors(
    rij_rad, rij_ang, pi_rad, pj_rad, pi_ang, pj_ang,
    atom_types, N, c2, c3,
    rc_radial, rc_angular, basis_size_radial, basis_size_angular,
    n_max_radial, n_max_angular, l_max_3b,
    has_q_222, has_q_1111, has_q_112,
    num_lm, c3b_coeffs, c4b_coeffs, c5b_coeffs,
    c4b2_coeffs,
    dtype, device,
    backend: str = "auto",
    has_q_123: int = 0, has_q_233: int = 0, has_q_134: int = 0,
) -> torch.Tensor:
    r"""Compute NEP4 descriptors from raw pair geometry. Returns (N, dim).

    This builds the Chebyshev/angular basis internally, unlike
    ``compute_descriptors_cached`` which takes them as inputs. Used by the
    non-training ASE-like path (NEPCalculator.compute).

    Parameters
    ----------
    rij_rad : (P_rad, 3) float   radial displacement vectors (A).
    rij_ang : (P_ang, 3) float   angular displacement vectors (A); may be
                                 empty for isolated atoms / dimers.
    pi_rad, pj_rad : (P_rad,) int64    central / neighbor atom indices.
    pi_ang, pj_ang : (P_ang,) int64    same, angular pairs.
    atom_types     : (N,) int64        per-atom type id in [0, ntypes).
    N              : int               total atoms in the batch.
    c2             : (ntypes, ntypes, basis_size_radial+1, n_max_radial+1)
                                        radial type-pair expansion coeffs.
    c3             : (ntypes, ntypes, basis_size_angular+1, n_max_angular+1)
                                        angular type-pair expansion coeffs.
    rc_radial, rc_angular : float      cutoffs (A).
    basis_size_radial, basis_size_angular : int
    n_max_radial, n_max_angular           : int
    l_max_3b       : int               max L for 3-body angular descriptors.
    has_q_222, has_q_1111, has_q_112 : int (0/1)
        Boolean switches for the three GPUMD-core mixed-body invariants.
        Each enabled flag adds an (n_max_angular+1)-sized block.
    num_lm         : int               \Sigma_{L=1..l_max_3b}(2L+1).
    c3b_coeffs, c4b_coeffs, c5b_coeffs, c4b2_coeffs :
        per-body fixed coefficient tensors.
    dtype, device   : torch dtype / device for outputs.
    backend         : "auto" | "loop" | "bmm" (see resolve_backend).

    Returns
    -------
    q : (N, dim) float   per-atom descriptor
    Zero-filled for atoms with no angular pairs (isolated atoms / dimers).
    """
    backend = resolve_backend(backend, num_types=int(c2.shape[0]))
    scatter_fn, type_fn = _select_contraction_funcs(backend)

    # --- Radial ---
    dij_rad = torch.norm(rij_rad, dim=-1)
    fk_rad = chebyshev_basis(dij_rad, rc_radial, basis_size_radial)
    q_rad = scatter_fn(fk_rad, pi_rad, pj_rad, atom_types, c2, N)

    parts = [q_rad]

    # --- Angular ---
    n_ap1 = n_max_angular + 1

    if l_max_3b > 0 and rij_ang.shape[0] == 0:
        # No angular neighbors (e.g. isolated atom / dimer). Emit zero-filled
        # angular blocks so output dim still matches q_scaler.
        parts.append(torch.zeros(N, n_ap1 * l_max_3b, dtype=dtype, device=device))
        for _ in range(int(has_q_222) + int(has_q_1111) + int(has_q_112)
                       + int(has_q_123) + int(has_q_233) + int(has_q_134)):
            parts.append(torch.zeros(N, n_ap1, dtype=dtype, device=device))

    if l_max_3b > 0 and rij_ang.shape[0] > 0:
        dij_ang = torch.norm(rij_ang, dim=-1)
        fk_ang = chebyshev_basis(dij_ang, rc_angular, basis_size_angular)
        gn_ang = type_fn(fk_ang, pi_ang, pj_ang, atom_types, c3)
        d12inv = 1.0 / torch.clamp(dij_ang, min=1e-10)
        blm = angular_basis(rij_ang[:, 0]*d12inv, rij_ang[:, 1]*d12inv,
                            rij_ang[:, 2]*d12inv, l_max_3b)
        gn_blm = gn_ang.unsqueeze(-1) * blm.unsqueeze(1)
        s = torch.zeros(N, n_ap1, num_lm, dtype=dtype, device=device)
        s.scatter_add_(0, pi_ang.unsqueeze(-1).unsqueeze(-1).expand_as(gn_blm),
                       gn_blm)

        # 3-body q
        q_3b_list = []
        for li in range(l_max_3b):
            L = li + 1
            nt = 2 * L + 1
            st = L * L - 1
            c = c3b_coeffs[st:st + nt]
            sb2 = s[:, :, st:st + nt] ** 2
            ql = c[0] * sb2[:, :, 0]
            if nt > 1:
                ql = ql + 2.0 * (c[1:] * sb2[:, :, 1:]).sum(-1)
            q_3b_list.append(ql)
        q_3b = torch.stack(q_3b_list, dim=-1).transpose(1, 2).reshape(N, -1)
        parts.append(q_3b)

        # Mixed-body blocks. Order (222 -> 1111 -> 112) must match GPUMD's
        # nep_utilities.cuh::find_q and save_nep_txt below.
        if has_q_222 or has_q_112:
            s20, s21r, s21i = s[:, :, 3], s[:, :, 4], s[:, :, 5]
            s22r, s22i = s[:, :, 6], s[:, :, 7]
        if has_q_1111 or has_q_112:
            s10, s11r, s11i = s[:, :, 0], s[:, :, 1], s[:, :, 2]

        if has_q_222:
            cb = c4b_coeffs
            q4 = (cb[0]*s20**3
                  + cb[1]*s20*(s21r**2 + s21i**2)
                  + cb[2]*s20*(s22r**2 + s22i**2)
                  + cb[3]*s22r*(s21i**2 - s21r**2)
                  + cb[4]*s21r*s21i*s22i)
            parts.append(q4)

        if has_q_1111:
            cb = c5b_coeffs
            s0sq = s10 ** 2
            s1sq = s11r ** 2 + s11i ** 2
            q5 = cb[0]*s0sq**2 + cb[1]*s0sq*s1sq + cb[2]*s1sq**2
            parts.append(q5)

        if has_q_112:
            cb = c4b2_coeffs
            q112 = (cb[0]*s10*s10*s20
                    + cb[1]*s10*(s11r*s21r + s11i*s21i)
                    + cb[2]*s20*(s11r*s11r + s11i*s11i)
                    + cb[3]*s22r*(s11r*s11r - s11i*s11i)
                    + cb[4]*s11r*s11i*s22i)
            parts.append(q112)

        if has_q_123:
            parts.append(_eval_extra(s, Q123_TERMS))
        if has_q_233:
            parts.append(_eval_extra(s, Q233_TERMS))
        if has_q_134:
            parts.append(_eval_extra(s, Q134_TERMS))

    q = torch.cat(parts, dim=-1)
    # DDP gradient pin (see compute_descriptors_cached for why).
    if rij_rad.shape[0] == 0:
        q = q + c2.sum() * 0.0
    if l_max_3b > 0 and rij_ang.shape[0] == 0:
        q = q + c3.sum() * 0.0
    return q


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

def apply_ann(
    q: torch.Tensor,
    atom_types: torch.Tensor,
    num_types: int,
    w0_list: List[torch.Tensor],
    b0_list: List[torch.Tensor],
    w1_list: List[torch.Tensor],
    b1: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Per-type NN: E_i = tanh(q_i @ W0_t^T - b0_t) @ w1_t - b1  (per element t).

    Parameters
    ----------
    q          : (N, dim) float       scaled per-atom descriptors.
    atom_types : (N,) int64           per-atom type id in [0, num_types).
    num_types  : int                  number of distinct element types.
    w0_list    : list[(neuron, dim)]  first-layer weights, one per type.
    b0_list    : list[(neuron,)]      first-layer biases, one per type.
    w1_list    : list[(neuron,)]      second-layer weights, one per type.
    b1         : scalar tensor        global energy offset.
    dtype, device : torch dtype / device for the output.

    Returns
    -------
    Ei : (N,) float — per-atom energy contribution.
    """
    N = q.shape[0]
    Ei = torch.zeros(N, dtype=dtype, device=device)
    for t in range(num_types):
        mask = atom_types == t
        if not mask.any():
            continue
        qt = q[mask]
        hidden = torch.tanh(qt @ w0_list[t].T - b0_list[t])
        Ei[mask] = hidden @ w1_list[t] - b1
    return Ei


# ---------------------------------------------------------------------------
# ZBL
# ---------------------------------------------------------------------------

def compute_zbl(
    atom_types, pair_i, pair_j, rij, N,
    atomic_numbers_list, rc_inner_default, rc_outer_default,
    typewise_factor, rc_inner_per_type, rc_outer_per_type,
    dtype, device,
) -> torch.Tensor:
    """ZBL repulsive energy with optional typewise cutoffs.

    Parameters
    ----------
    atom_types : (N,) int64
    pair_i, pair_j : (P,) int64   angular-list pairs (ZBL reuses them).
    rij        : (P, 3) float     displacement vectors (A).
    N          : int              total atoms in the batch.
    atomic_numbers_list : list[int] of length ntypes, Z per element.
    rc_inner_default, rc_outer_default : float
        Global fallback switching window (A) when typewise is off.
    typewise_factor : float or None
    rc_inner_per_type, rc_outer_per_type : (ntypes,) float or None
        Per-element covalent-radius-based switching window; when both are
        provided the per-pair cutoff is
        min((cov_i + cov_j) * typewise_factor, rc_outer_default).
    dtype, device : torch dtype / device for the output.

    Returns
    -------
    Ei_zbl : (N,) float — per-atom ZBL repulsive contribution (half-split
             across pairs, scatter-added onto central atoms).
    """
    dij = torch.norm(rij, dim=-1)
    use_tw = typewise_factor is not None and rc_inner_per_type is not None

    # Coarse cutoff for the initial distance mask. For typewise, the actual
    # per-pair cutoff may be smaller, so we evaluate tighter cutoffs later.
    if use_tw:
        max_rc = min(float(rc_outer_per_type.max().item()), rc_outer_default)
    else:
        max_rc = rc_outer_default

    mask = dij < max_rc
    if not mask.any():
        return torch.zeros(N, dtype=dtype, device=device)

    pi, pj = pair_i[mask], pair_j[mask]
    d = dij[mask]

    if use_tw:
        # NEP_CPU typewise convention (nep.cpp:1795-1801):
        #   rc_outer_pair = min((cov_i + cov_j) * typewise_factor, rc_outer_default)
        #   rc_inner      = 0
        # rc_outer_per_type was built as 2*typewise_factor*cov, so its pair
        # average equals (cov_i + cov_j) * typewise_factor.
        t1 = atom_types[pi]
        t2 = atom_types[pj]
        rc_outer_pair = (rc_outer_per_type[t1] + rc_outer_per_type[t2]) * 0.5
        rc_outer = torch.clamp(rc_outer_pair, max=rc_outer_default)
        rc_inner = torch.zeros_like(rc_outer)
        # Drop pairs outside the tightened per-pair cutoff
        tw_mask = d < rc_outer
        if not tw_mask.all():
            pi, pj = pi[tw_mask], pj[tw_mask]
            d = d[tw_mask]
            rc_inner = rc_inner[tw_mask]
            rc_outer = rc_outer[tw_mask]
    else:
        rc_inner = rc_inner_default
        rc_outer = rc_outer_default

    an = torch.tensor(atomic_numbers_list, dtype=dtype, device=device)
    zi = an[atom_types[pi]]
    zj = an[atom_types[pj]]

    # Constant must match NEP_CPU's `2.134563` literal to get bit-identical
    # forward output; 1/0.46850 = 2.1344717… differs at the 4e-5 level,
    # which accumulates to meV-scale errors on strong ZBL pairs.
    a_inv = (zi ** 0.23 + zj ** 0.23) * 2.134563
    zizj = K_C_SP * zi * zj
    x = d * a_inv
    phi = (ZBL_PARA[0] * torch.exp(-ZBL_PARA[1] * x)
           + ZBL_PARA[2] * torch.exp(-ZBL_PARA[3] * x)
           + ZBL_PARA[4] * torch.exp(-ZBL_PARA[5] * x)
           + ZBL_PARA[6] * torch.exp(-ZBL_PARA[7] * x))

    rc_i = rc_inner  # per-pair tensor in typewise, scalar otherwise
    rc_o = rc_outer

    # Smooth ZBL cutoff: 1 for d<rc_i, 0.5*cos(pi*t)+0.5 taper for
    # rc_i<=d<=rc_o, 0 beyond. Written branch-free / out-of-place: clamping the
    # normalised distance t=(d-rc_i)/(rc_o-rc_i) to [0,1] reproduces all three
    # regions exactly (t=0 -> fc=1, t=1 -> fc=0). The previous masked in-place
    # form (fc[m1]=1; fc[m2]=...) triggered an IndexPutBackward anomaly warning
    # under torch.compile and forced graph breaks via the data-dependent
    # ``if m2.any()`` / boolean indexing — this version avoids both and is
    # numerically identical (rc_o > rc_i always holds). dtype handles the
    # scalar and per-pair-tensor (typewise) cases identically.
    t = torch.clamp((d - rc_i) / (rc_o - rc_i), 0.0, 1.0)
    fc = 0.5 * torch.cos(PI * t) + 0.5

    e_pair = zizj * phi / d * fc
    # Neighbor list is directed: every physical pair (i,j) appears twice
    # (once as (i,j) and once as (j,i)). Halving the pair energy and
    # scattering only to pi gives each atom 0.5*e_pair per neighbour and
    # a total system energy of 1*e_pair per physical pair.
    e_atom = torch.zeros(N, dtype=dtype, device=device)
    e_atom.scatter_add_(0, pi, 0.5 * e_pair)
    return e_atom


def _scatter_contraction_loop(basis, pair_i, pair_j, atom_types, c, N):
    r"""Type-pair loop: \Sigma_k c[t1, t2, n, k]*basis[p, k] then scatter_add into q.

    The outer loop is O(ntypes**2) python iterations. Preferred when ntypes is
    small (few kernel launches, each matmul is fat) and peak memory matters.

    Inputs
    ------
    basis      : (P, K)                         K = basis_size + 1.
    pair_i, pair_j : (P,) int64                 central / neighbor indices.
    atom_types : (N,) int64                     in [0, ntypes).
    c          : (ntypes, ntypes, N_out, K)     expansion coefficients.
    N          : int                            total atoms in the batch.

    Returns
    -------
    q : (N, N_out) float  scatter-summed per-atom contribution.
    """
    ntypes = c.shape[0]
    t1 = atom_types[pair_i]
    t2 = atom_types[pair_j]
    q = torch.zeros(N, c.shape[2], dtype=basis.dtype, device=basis.device)
    for _t1 in range(ntypes):
        for _t2 in range(ntypes):
            _m = (t1 == _t1) & (t2 == _t2)
            if not _m.any():
                continue
            _gn = basis[_m] @ c[_t1, _t2].T
            q.scatter_add_(0, pair_i[_m].unsqueeze(-1).expand_as(_gn), _gn)
    return q


def _type_contraction_loop(basis, pair_i, pair_j, atom_types, c):
    """Type-pair loop, no scatter — returns per-pair contracted basis.

    Shapes: same as ``_scatter_contraction_loop`` but returns
        gn : (P, N_out) float
    (the "pair-level angular radial factor" used by the 3b/4b/5b forward).
    """
    ntypes = c.shape[0]
    t1 = atom_types[pair_i]
    t2 = atom_types[pair_j]
    gn = torch.zeros(basis.shape[0], c.shape[2], dtype=basis.dtype, device=basis.device)
    for _t1 in range(ntypes):
        for _t2 in range(ntypes):
            _m = (t1 == _t1) & (t2 == _t2)
            if not _m.any():
                continue
            gn[_m] = basis[_m] @ c[_t1, _t2].T
    return gn


def _scatter_contraction_bmm(basis, pair_i, pair_j, atom_types, c, N):
    """Vectorised: gather c[t1, t2] per pair, one torch.bmm, scatter_add.

    Allocates a (P, N_out, K) intermediate so peak memory is higher, but
    launches ~1 kernel instead of O(ntypes**2). Wins when ntypes >= ~8."""
    t1 = atom_types[pair_i]
    t2 = atom_types[pair_j]
    c_p = c[t1, t2]                                            # (P, N_out, K)
    gn = torch.bmm(c_p, basis.unsqueeze(-1)).squeeze(-1)        # (P, N_out)
    q = torch.zeros(N, c.shape[2], dtype=basis.dtype, device=basis.device)
    q.scatter_add_(0, pair_i.unsqueeze(-1).expand_as(gn), gn)
    return q


def _type_contraction_bmm(basis, pair_i, pair_j, atom_types, c):
    """Vectorised counterpart of ``_type_contraction_loop``."""
    t1 = atom_types[pair_i]
    t2 = atom_types[pair_j]
    c_p = c[t1, t2]                                            # (P, N_out, K)
    return torch.bmm(c_p, basis.unsqueeze(-1)).squeeze(-1)      # (P, N_out)


def compute_descriptors_cached(
    fk_rad, fk_ang, blm,
    pi_rad, pj_rad, pi_ang, pj_ang,
    atom_types, N, c2, c3,
    n_max_radial, n_max_angular, l_max_3b,
    has_q_222, has_q_1111, has_q_112,
    num_lm, c3b_coeffs, c4b_coeffs, c5b_coeffs,
    c4b2_coeffs,
    dtype, device,
    return_intermediates: bool = False,
    backend: str = "loop",
    has_q_123: int = 0, has_q_233: int = 0, has_q_134: int = 0,
):
    r"""Compute descriptors using precomputed basis functions.

    Faster than compute_descriptors because Chebyshev/angular basis are cached.
    Differentiable through c2, c3 only (not rij).

    Parameters
    ----------
    fk_rad : (P_rad, basis_size_radial + 1) float   cached Chebyshev basis.
    fk_ang : (P_ang, basis_size_angular + 1) float  cached Chebyshev basis
             (may be empty if no angular pairs).
    blm    : (P_ang, num_lm) float  cached angular basis.
    pi_rad, pj_rad : (P_rad,) int64   radial central / neighbor indices.
    pi_ang, pj_ang : (P_ang,) int64   angular central / neighbor indices.
    atom_types     : (N,) int64      per-atom type id in [0, ntypes).
    N              : int             total atoms in the batch.
    c2 : (ntypes, ntypes, basis_size_radial + 1, n_max_radial + 1).
    c3 : (ntypes, ntypes, basis_size_angular + 1, n_max_angular + 1).
    n_max_radial, n_max_angular, l_max_3b : int.
    has_q_222, has_q_1111, has_q_112 : int (0/1) flags.
    num_lm : int  = \Sigma_{L=1..l_max_3b}(2L + 1).
    c3b_coeffs, c4b_coeffs, c5b_coeffs, c4b2_coeffs :
        body-order coefficient tensors (c4b2 is for q_112).
    dtype, device : torch dtype / device for outputs.
    return_intermediates : bool  also return s and gn_ang (needed by the
                                 analytical-force path).
    backend : "loop" | "bmm"  type-pair contraction implementation:
        "loop" — pure-PyTorch ntypes**2 loop (few types)
        "bmm"  — fancy-index + torch.bmm (many types)

    Returns
    -------
    q : (N, dim) float — per-atom descriptor. dim breakdown matches
        ``compute_descriptors``.
    If ``return_intermediates``: also returns
        s      : (N, n_max_angular + 1, num_lm) sum_fxyz moments
        gn_ang : (P_ang, n_max_angular + 1) pair-level angular radial factor
        Both are ``None`` when l_max_3b == 0 or there are no angular pairs.
    """
    _scatter_fn, _type_fn = _select_contraction_funcs(backend)

    q_rad = _scatter_fn(fk_rad, pi_rad, pj_rad, atom_types, c2, N)

    parts = [q_rad]
    s_out = None
    gn_ang_out = None
    n_ap1 = n_max_angular + 1

    if l_max_3b > 0 and fk_ang.shape[0] == 0:
        # No angular neighbors anywhere in the batch (all-monomer bucket, or
        # very sparse structures). Emit zero-filled angular blocks so the
        # returned dim still matches q_scaler / the NN's input layer.
        parts.append(torch.zeros(N, n_ap1 * l_max_3b, dtype=dtype, device=device))
        for _ in range(int(has_q_222) + int(has_q_1111) + int(has_q_112)
                       + int(has_q_123) + int(has_q_233) + int(has_q_134)):
            parts.append(torch.zeros(N, n_ap1, dtype=dtype, device=device))

    if l_max_3b > 0 and fk_ang.shape[0] > 0:
        # Angular: type contraction (no scatter yet — need gn for blm product)
        gn_ang = _type_fn(fk_ang, pi_ang, pj_ang, atom_types, c3)

        gn_blm = gn_ang.unsqueeze(-1) * blm.unsqueeze(1)
        s = torch.zeros(N, n_ap1, num_lm, dtype=dtype, device=device)
        s.scatter_add_(0, pi_ang.unsqueeze(-1).unsqueeze(-1).expand_as(gn_blm),
                       gn_blm)

        if return_intermediates:
            s_out = s
            gn_ang_out = gn_ang

        q_3b_list = []
        for li in range(l_max_3b):
            L = li + 1
            nt = 2 * L + 1
            st = L * L - 1
            c = c3b_coeffs[st:st + nt]
            sb2 = s[:, :, st:st + nt] ** 2
            ql = c[0] * sb2[:, :, 0]
            if nt > 1:
                ql = ql + 2.0 * (c[1:] * sb2[:, :, 1:]).sum(-1)
            q_3b_list.append(ql)
        q_3b = torch.stack(q_3b_list, dim=-1).transpose(1, 2).reshape(N, -1)
        parts.append(q_3b)

        if has_q_222 or has_q_112:
            s20, s21r, s21i = s[:, :, 3], s[:, :, 4], s[:, :, 5]
            s22r, s22i = s[:, :, 6], s[:, :, 7]
        if has_q_1111 or has_q_112:
            s10, s11r, s11i = s[:, :, 0], s[:, :, 1], s[:, :, 2]

        if has_q_222:
            cb = c4b_coeffs
            q4 = (cb[0]*s20**3 + cb[1]*s20*(s21r**2 + s21i**2)
                  + cb[2]*s20*(s22r**2 + s22i**2)
                  + cb[3]*s22r*(s21i**2 - s21r**2)
                  + cb[4]*s21r*s21i*s22i)
            parts.append(q4)

        if has_q_1111:
            cb = c5b_coeffs
            s0sq = s10 ** 2
            s1sq = s11r ** 2 + s11i ** 2
            q5 = cb[0]*s0sq**2 + cb[1]*s0sq*s1sq + cb[2]*s1sq**2
            parts.append(q5)

        if has_q_112:
            cb = c4b2_coeffs
            q112 = (cb[0]*s10*s10*s20
                    + cb[1]*s10*(s11r*s21r + s11i*s21i)
                    + cb[2]*s20*(s11r*s11r + s11i*s11i)
                    + cb[3]*s22r*(s11r*s11r - s11i*s11i)
                    + cb[4]*s11r*s11i*s22i)
            parts.append(q112)

        if has_q_123:
            parts.append(_eval_extra(s, Q123_TERMS))
        if has_q_233:
            parts.append(_eval_extra(s, Q233_TERMS))
        if has_q_134:
            parts.append(_eval_extra(s, Q134_TERMS))

    q = torch.cat(parts, dim=-1)
    # DDP gradient pin: when a batch has no pairs (all-monomer bucket under
    # bucket batching), c2 / c3 never enter the compute graph and DDP errors
    # out with "parameter did not receive grad". Add a zero-valued grad path
    # through c2/c3 so they stay synchronised across ranks. Guarded so the
    # (common) non-empty case pays nothing.
    if fk_rad.shape[0] == 0:
        q = q + c2.sum() * 0.0
    if l_max_3b > 0 and fk_ang.shape[0] == 0:
        q = q + c3.sum() * 0.0
    if return_intermediates:
        return q, s_out, gn_ang_out
    return q


# ---------------------------------------------------------------------------
# Extra angular-invariant channels (q_123 / q_233 bispectrum, and any future
# higher body-order term). Each channel is a polynomial in the single-radial-
# channel angular moments s[:, :, lm], stored as a list of
# (coefficient, (lm_idx, ...)) monomials (see constants.Q123_TERMS / Q233_TERMS;
# new terms can be derived with probe/derive_invariants.py). These helpers
# evaluate a channel and its gradient generically, so adding a term needs only
# the table — no hand-derived gradient.
# ---------------------------------------------------------------------------

_EXTRA_PACK_CACHE = {}


def _pack_terms(terms, device):
    """Pack a channel's term list into (coeffs, idx) tensors.

    All monomials in one channel are homogeneous (same degree d), so idx is a
    dense (T, d) long tensor and no padding is needed. Cached per (terms id,
    device) since the tables are fixed at model-build time.
    """
    key = (id(terms), device)
    hit = _EXTRA_PACK_CACHE.get(key)
    if hit is not None:
        return hit
    deg = len(terms[0][1])
    coeffs = torch.tensor([c for c, _ in terms], dtype=torch.float64,
                          device=device)
    idx = torch.tensor([list(ix) for _, ix in terms], dtype=torch.long,
                       device=device)                       # (T, d)
    _EXTRA_PACK_CACHE[key] = (coeffs, idx, deg)
    return coeffs, idx, deg


def _eval_extra(s, terms):
    """Evaluate one extra invariant channel. Returns (N, n_ap1).

    s: (N, n_ap1, num_lm). q = Σ_t coeff_t · Π_p s[:, :, idx[t, p]].
    Vectorised: gather all monomial factors, take the product over the
    degree axis, weight by coeffs and sum over terms."""
    coeffs, idx, _ = _pack_terms(terms, s.device)
    g = s[:, :, idx]                          # (N, n_ap1, T, d)
    prod = g.prod(dim=-1)                     # (N, n_ap1, T)
    return (prod * coeffs.to(s.dtype)).sum(-1)


def _extra_grad(s, terms):
    """dq_channel / ds[:, :, lm] for one channel. Returns (N, n_ap1, num_lm).

    Uses leave-one-out products (prefix*suffix, division-free so it is exact
    even when a factor is zero) then scatter-adds each monomial's per-factor
    contribution to the lm index it differentiates.
    """
    coeffs, idx, d = _pack_terms(terms, s.device)
    c = coeffs.to(s.dtype)
    g = s[:, :, idx]                          # (N, n_ap1, T, d)
    N, n_ap1, T, _ = g.shape

    # Leave-one-out products via functional prefix / suffix scans over the
    # degree axis (no in-place writes — the analytical-force path runs this
    # with ``s`` in the autograd graph, so in-place ops would break backward).
    factors = [g[:, :, :, p] for p in range(d)]      # each (N, n_ap1, T)
    pref, suff = [None] * d, [None] * d
    acc = torch.ones_like(factors[0])
    for p in range(d):
        pref[p] = acc
        acc = acc * factors[p]
    acc = torch.ones_like(factors[0])
    for p in range(d - 1, -1, -1):
        suff[p] = acc
        acc = acc * factors[p]
    lop = torch.stack([pref[p] * suff[p] for p in range(d)], dim=-1)

    contrib = lop * c.view(1, 1, T, 1)        # (N, n_ap1, T, d)
    flat_idx = idx.reshape(-1).view(1, 1, -1).expand(N, n_ap1, T * d)
    grad = torch.zeros_like(s).scatter_add(
        2, flat_idx, contrib.reshape(N, n_ap1, T * d))
    return grad


def _angular_weight(Fp, s, dim_r, n_ap1, l_max_3b,
                    has_q_222, has_q_1111, has_q_112,
                    c3b_coeffs, c4b_coeffs, c5b_coeffs,
                    c4b2_coeffs,
                    has_q_123=0, has_q_233=0, has_q_134=0):
    """Compute dEi/d(sum_fxyz)[N, n_ap1, num_lm] for ALL body orders.

    This is the "effective Fp" in sum_fxyz space needed for the analytical
    angular force chain rule. Differentiable through s (-> c3) and Fp (-> NN weights).

    ``has_q_123`` / ``has_q_233``: the extra 4-body bispectrum channels, each
    adding one (n_ap1) descriptor block after q_112.
    """
    N = s.shape[0]
    weight = torch.zeros_like(s)  # (N, n_ap1, num_lm)

    # --- 3-body: q3b = sum_l c[m=0]*s[l,m=0]**2 + 2*sum_{m>0} c[m]*s[l,m]**2
    # dq3b_l/ds[n,st+m] = 2*c3b[st]   *s[n,st]   for m=0
    #                    = 4*c3b[st+m] *s[n,st+m] for m>0
    Fp_3b = Fp[:, dim_r:dim_r + l_max_3b * n_ap1].reshape(N, l_max_3b, n_ap1)
    for li in range(l_max_3b):
        L = li + 1
        nt = 2 * L + 1
        st = L * L - 1
        c = c3b_coeffs[st:st + nt]       # (nt,)
        s_lm = s[:, :, st:st + nt]        # (N, n_ap1, nt)
        Fp_l = Fp_3b[:, li, :].unsqueeze(-1)  # (N, n_ap1, 1)
        dq_ds = 2.0 * c * s_lm            # m=0: 2c[0]*s; m>0: 2c[m]*s (*2 below)
        dq_ds[:, :, 1:] = dq_ds[:, :, 1:] * 2.0  # m>0 gets extra factor 2
        weight[:, :, st:st + nt] = weight[:, :, st:st + nt] + Fp_l * dq_ds

    # Block ordering must match compute_descriptors / save_nep_txt:
    # q_222 -> q_1111 -> q_112.
    off = dim_r + l_max_3b * n_ap1

    s10, s11r, s11i = s[:, :, 0], s[:, :, 1], s[:, :, 2]
    if s.shape[2] >= 8:
        s20, s21r, s21i = s[:, :, 3], s[:, :, 4], s[:, :, 5]
        s22r, s22i = s[:, :, 6], s[:, :, 7]

    if has_q_222 and s.shape[2] >= 8:
        Fp_4b = Fp[:, off:off + n_ap1]; off += n_ap1
        cb = c4b_coeffs
        weight[:, :, 3] = weight[:, :, 3] + Fp_4b * (
            3*cb[0]*s20**2 + cb[1]*(s21r**2 + s21i**2) + cb[2]*(s22r**2 + s22i**2))
        weight[:, :, 4] = weight[:, :, 4] + Fp_4b * (
            2*cb[1]*s20*s21r - 2*cb[3]*s22r*s21r + cb[4]*s21i*s22i)
        weight[:, :, 5] = weight[:, :, 5] + Fp_4b * (
            2*cb[1]*s20*s21i + 2*cb[3]*s22r*s21i + cb[4]*s21r*s22i)
        weight[:, :, 6] = weight[:, :, 6] + Fp_4b * (
            2*cb[2]*s20*s22r + cb[3]*(s21i**2 - s21r**2))
        weight[:, :, 7] = weight[:, :, 7] + Fp_4b * (
            2*cb[2]*s20*s22i + cb[4]*s21r*s21i)

    if has_q_1111:
        Fp_5b = Fp[:, off:off + n_ap1]; off += n_ap1
        s0sq = s10**2; s1sq = s11r**2 + s11i**2
        cb5 = c5b_coeffs
        factor_1sq = cb5[1]*s0sq + 2*cb5[2]*s1sq
        weight[:, :, 0] = weight[:, :, 0] + Fp_5b * 2*s10*(2*cb5[0]*s0sq + cb5[1]*s1sq)
        weight[:, :, 1] = weight[:, :, 1] + Fp_5b * 2*s11r*factor_1sq
        weight[:, :, 2] = weight[:, :, 2] + Fp_5b * 2*s11i*factor_1sq

    # q_112 gradient uses shorthand (a,b,c)=L=1 moments, (d,e,f,g,h)=L=2
    # moments. Polynomial form lives in compute_descriptors_cached.
    if has_q_112 and s.shape[2] >= 8:
        Fp_b = Fp[:, off:off + n_ap1]; off += n_ap1
        cb = c4b2_coeffs
        a, b, c = s10, s11r, s11i
        d, e, f, g, h = s20, s21r, s21i, s22r, s22i
        # d/da
        weight[:, :, 0] = weight[:, :, 0] + Fp_b * (
            2*cb[0]*a*d + cb[1]*(b*e + c*f))
        # d/db
        weight[:, :, 1] = weight[:, :, 1] + Fp_b * (
            cb[1]*a*e + 2*cb[2]*d*b + 2*cb[3]*g*b + cb[4]*c*h)
        # d/dc
        weight[:, :, 2] = weight[:, :, 2] + Fp_b * (
            cb[1]*a*f + 2*cb[2]*d*c - 2*cb[3]*g*c + cb[4]*b*h)
        # d/dd
        weight[:, :, 3] = weight[:, :, 3] + Fp_b * (
            cb[0]*a*a + cb[2]*(b*b + c*c))
        # d/de
        weight[:, :, 4] = weight[:, :, 4] + Fp_b * (cb[1]*a*b)
        # d/df
        weight[:, :, 5] = weight[:, :, 5] + Fp_b * (cb[1]*a*c)
        # d/dg
        weight[:, :, 6] = weight[:, :, 6] + Fp_b * (cb[3]*(b*b - c*c))
        # d/dh
        weight[:, :, 7] = weight[:, :, 7] + Fp_b * (cb[4]*b*c)

    # --- q_123 / q_233 bispectrum channels (one block each, after q_112) ---
    if has_q_123:
        Fp_c = Fp[:, off:off + n_ap1]; off += n_ap1
        weight = weight + Fp_c.unsqueeze(-1) * _extra_grad(s, Q123_TERMS)
    if has_q_233:
        Fp_c = Fp[:, off:off + n_ap1]; off += n_ap1
        weight = weight + Fp_c.unsqueeze(-1) * _extra_grad(s, Q233_TERMS)
    if has_q_134:
        Fp_c = Fp[:, off:off + n_ap1]; off += n_ap1
        weight = weight + Fp_c.unsqueeze(-1) * _extra_grad(s, Q134_TERMS)

    return weight  # (N, n_ap1, num_lm)


def compute_analytical_forces(
    Fp, atom_types, N,
    c2, c3, fkp_rad, fkp_ang, blm,
    pi_rad, pj_rad, rij_rad, d12inv_rad,
    pi_ang, pj_ang, rij_ang, d12inv_ang,
    s, gn_ang,
    n_max_radial, n_max_angular, l_max_3b,
    has_q_222, has_q_1111, has_q_112,
    num_lm, c3b_coeffs, c4b_coeffs, c5b_coeffs,
    c4b2_coeffs,
    dtype, device,
    compute_virial: bool = True,
    backend: str = "loop",
    has_q_123: int = 0, has_q_233: int = 0, has_q_134: int = 0,
):
    """Compute forces analytically — no create_graph needed, fully differentiable
    through c2, c3 and NN weights (via Fp).

    Parameters
    ----------
    Fp : (N, dim) float             dEi/dq * q_scaler (already includes q_scaler).
    atom_types : (N,) int64
    N : int                         total atoms in the batch.
    c2 : (ntypes, ntypes, basis_size_radial+1, n_max_radial+1).
    c3 : (ntypes, ntypes, basis_size_angular+1, n_max_angular+1).
    fkp_rad : (P_rad, basis_size_radial + 1) float    d(fk)/d(rij) for radial.
    fkp_ang : (P_ang, basis_size_angular + 1) float   d(fk)/d(rij) for angular.
    blm     : (P_ang, num_lm) float                   cached angular basis.
    pi_rad, pj_rad : (P_rad,) int64
    rij_rad        : (P_rad, 3) float  displacement vectors (A).
    d12inv_rad     : (P_rad,) float    1 / |rij_rad|.
    pi_ang, pj_ang : (P_ang,) int64
    rij_ang        : (P_ang, 3) float
    d12inv_ang     : (P_ang,) float
    s      : (N, n_max_angular + 1, num_lm) sum_fxyz from descriptor forward.
    gn_ang : (P_ang, n_max_angular + 1) pair-level angular radial factor.
    n_max_radial, n_max_angular, l_max_3b : int.
    has_q_222, has_q_1111, has_q_112 : int (0/1) flags.
    num_lm : int.
    c3b_coeffs, c4b_coeffs, c5b_coeffs, c4b2_coeffs :
        body-order coefficient tensors.
    dtype, device : torch dtype / device for outputs.
    compute_virial : bool  if False, ``virial`` output is ``None``.
    backend : "loop" | "bmm" — see ``compute_descriptors_cached``.

    Returns
    -------
    forces : (N, 3) float
    virial : (N, 9) float or None  per-atom 3*3 virial flattened row-major.

    Geometry tensors (rij, fkp, d12inv, blm) are detached so PyTorch does not
    track gradients through them — they are not trainable parameters. Only Fp
    (-> NN weights) and c2/c3 carry gradient information.
    """
    _, _type_fn = _select_contraction_funcs(backend)

    forces = torch.zeros(N, 3, dtype=dtype, device=device)
    virial = torch.zeros(N, 9, dtype=dtype, device=device) if compute_virial else None
    dim_r = n_max_radial + 1

    # Detach all geometry — these are precomputed from fixed atom positions
    rij_rad   = rij_rad.detach()
    d12inv_rad = d12inv_rad.detach()
    fkp_rad   = fkp_rad.detach()
    if rij_ang is not None:
        rij_ang    = rij_ang.detach()
        d12inv_ang = d12inv_ang.detach()
        fkp_ang    = fkp_ang.detach()
        blm        = blm.detach()

    def _exp(idx, t):
        return idx.unsqueeze(-1).expand_as(t)

    # ---- Radial force ----
    Fp_rad = Fp[:, :dim_r]
    gnp_rad = _type_fn(fkp_rad, pi_rad, pj_rad, atom_types, c2)
    tmp_rad = (Fp_rad[pi_rad] * gnp_rad).sum(-1, keepdim=True) * d12inv_rad.unsqueeze(-1)
    f12_rad = tmp_rad * rij_rad

    forces.scatter_add_(0, _exp(pi_rad, f12_rad), f12_rad)
    forces.scatter_add_(0, _exp(pj_rad, f12_rad), -f12_rad)
    if compute_virial:
        v9 = -(rij_rad.unsqueeze(-1) * f12_rad.unsqueeze(-2)).reshape(-1, 9)
        virial.scatter_add_(0, pj_rad.unsqueeze(-1).expand_as(v9), v9)

    # ---- Angular force ----
    if l_max_3b > 0 and pi_ang.shape[0] > 0 and s is not None:
        n_ap1 = n_max_angular + 1

        # gnp_ang: radial distance derivative term (differentiable via c3)
        gnp_ang_v = _type_fn(fkp_ang, pi_ang, pj_ang, atom_types, c3)

        # weight = dEi/d(sum_fxyz): all body orders, differentiable via s->c3 and Fp->NN
        w_atom = _angular_weight(Fp, s, dim_r, n_ap1, l_max_3b,
                                 has_q_222, has_q_1111, has_q_112,
                                 c3b_coeffs, c4b_coeffs, c5b_coeffs,
                                 c4b2_coeffs,
                                 has_q_123=has_q_123,
                                 has_q_233=has_q_233,
                                 has_q_134=has_q_134)  # (N, n_ap1, num_lm)
        w_i = w_atom[pi_ang]   # (P, n_ap1, num_lm) — atom_i's weight per pair

        # Term 1: distance derivative — f12 = (sum_n,lm w_i * gnp * blm) * rij/dij
        gnp_blm = gnp_ang_v.unsqueeze(-1) * blm.unsqueeze(1)  # (P, n_ap1, num_lm)
        scalar_gnp = (w_i * gnp_blm).sum(dim=(1, 2))           # (P,)
        f12_gnp = (scalar_gnp * d12inv_ang).unsqueeze(-1) * rij_ang

        # Term 2: direction derivative — f12 = sum_n,lm w_i * gn * dblm/drij
        # dblm/drij = (dblm/dhat - hat*(hat*dblm/dhat)) / dij
        x_hat = rij_ang[:, 0] * d12inv_ang
        y_hat = rij_ang[:, 1] * d12inv_ang
        z_hat = rij_ang[:, 2] * d12inv_ang
        # dblm_dhat depends only on geometry -> no-grad context
        with torch.no_grad():
            dblm_dhat = _compute_dblm_dhat(x_hat, y_hat, z_hat, l_max_3b)  # (P, num_lm, 3)
        hat = torch.stack([x_hat, y_hat, z_hat], dim=-1)

        w_gn = (w_i * gn_ang.unsqueeze(-1)).sum(dim=1)  # (P, num_lm)
        term1 = (w_gn.unsqueeze(-1) * dblm_dhat).sum(1) * d12inv_ang.unsqueeze(-1)
        hat_dot_dblm = (hat.unsqueeze(1) * dblm_dhat).sum(-1)  # (P, num_lm)
        t2_sc = (w_gn * hat_dot_dblm).sum(1) * d12inv_ang
        f12_ang = f12_gnp + term1 - t2_sc.unsqueeze(-1) * hat

        forces.scatter_add_(0, _exp(pi_ang, f12_ang), f12_ang)
        forces.scatter_add_(0, _exp(pj_ang, f12_ang), -f12_ang)
        if compute_virial:
            v9_a = -(rij_ang.unsqueeze(-1) * f12_ang.unsqueeze(-2)).reshape(-1, 9)
            virial.scatter_add_(0, pj_ang.unsqueeze(-1).expand_as(v9_a), v9_a)

    return forces, virial


def _compute_dblm_dhat(x, y, z, l_max_3b):
    """Derivatives of ``angular_basis`` wrt unit direction (xhat, yhat, zhat).

    For a basis element blm = z_factor(z) * C(x, y) where
      C = 1              (n1 = 0)
      C = Re[(x+iy)^n1]  (n1 > 0, real  component)
      C = Im[(x+iy)^n1]  (n1 > 0, imag component)
    the gradients are:
      d(blm)/dx = z_factor(z)        * d(C)/dx
      d(blm)/dy = z_factor(z)        * d(C)/dy
      d(blm)/dz = z_factor'(z)       * C
    with d(Re_n)/dx =  n*Re_{n-1},  d(Im_n)/dx =  n*Im_{n-1},
         d(Re_n)/dy = -n*Im_{n-1},  d(Im_n)/dy =  n*Re_{n-1}.

    Returns: (P, num_lm, 3) where [..., 0/1/2] = d/d(xhat, yhat, zhat).
    Bit-identical to the old hand-coded implementation for l_max_3b <= 4.
    """
    if l_max_3b < 1:
        return torch.zeros(x.shape[0], 0, 3, dtype=x.dtype, device=x.device)
    if l_max_3b > MAX_L3B:
        raise ValueError(f"l_max_3b={l_max_3b} exceeds MAX_L3B={MAX_L3B}")

    z_pow = _build_z_powers(z, l_max_3b)
    # Need (x+iy)^n for n = 0..l_max_3b-1 (max n1-1) + (x+iy)^n1 for n1 up to l_max_3b
    Re, Im = _build_xy_powers(x, y, l_max_3b)

    zeros = torch.zeros_like(x)
    derivs = []  # list of (P, 3)

    for L in range(1, l_max_3b + 1):
        Z = Z_COEFFICIENT[L]
        for n1 in range(L + 1):
            parity = (L + n1) % 2
            start = parity
            stop = L - n1 + 1
            zf = _z_factor(z_pow, Z[n1], start, stop)
            # z_factor'(z) = sum n2 * coeff[n2] * z^{n2-1}, iterating the same
            # parity as z_factor (start, start+2, ...); only n2>=1 contributes.
            zfp = None
            n2 = start
            while n2 < stop:
                if n2 >= 1:
                    c = Z[n1][n2]
                    if c != 0.0:
                        term = (n2 * c) * z_pow[n2 - 1]
                        zfp = term if zfp is None else zfp + term
                n2 += 2
            if zfp is None:
                zfp = zeros

            if n1 == 0:
                # blm = zf(z). dx=0, dy=0, dz=zf'(z)
                derivs.append(torch.stack([zeros, zeros, zfp], dim=-1))
            else:
                # real component
                dRe_dx = n1 * Re[n1 - 1]
                dRe_dy = -n1 * Im[n1 - 1]
                derivs.append(torch.stack(
                    [zf * dRe_dx, zf * dRe_dy, zfp * Re[n1]], dim=-1))
                # imag component
                dIm_dx = n1 * Im[n1 - 1]
                dIm_dy = n1 * Re[n1 - 1]
                derivs.append(torch.stack(
                    [zf * dIm_dx, zf * dIm_dy, zfp * Im[n1]], dim=-1))

    return torch.stack(derivs, dim=1)  # (P, num_lm, 3)


# ---------------------------------------------------------------------------
# Force / virial accumulation
# ---------------------------------------------------------------------------

def accumulate_forces_virial(
    N, pi_rad, pj_rad, rij_rad, g_rad,
    pi_ang, pj_ang, rij_ang, g_ang,
    dtype, device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Accumulate pair gradients into per-atom forces and virial.

    F_k = sum_{pairs i=k} grad - sum_{pairs j=k} grad
    virial_ab(j) = -rij_a * grad_b  (accumulated on j)

    Parameters
    ----------
    N : int                        total atoms in the batch.
    pi_rad, pj_rad : (P_rad,) int64    radial pairs (central, neighbor).
    rij_rad : (P_rad, 3) float         radial displacement vectors.
    g_rad   : (P_rad, 3) float         pair gradients dE/drij for radial.
    pi_ang, pj_ang : (P_ang,) int64    angular pairs.
    rij_ang : (P_ang, 3) float         angular displacement vectors.
    g_ang   : (P_ang, 3) float         pair gradients dE/drij for angular.
    dtype, device : torch dtype / device for outputs.

    Returns
    -------
    forces : (N, 3) float  per-atom force in eV/A.
    virial : (N, 9) float  per-atom virial tensor (row-major 3*3) in eV.
    """
    forces = torch.zeros(N, 3, dtype=dtype, device=device)
    virial = torch.zeros(N, 9, dtype=dtype, device=device)

    def _acc(pi, pj, r, g):
        e = lambda idx: idx.unsqueeze(-1).expand_as(g)
        forces.scatter_add_(0, e(pi), g)
        forces.scatter_add_(0, e(pj), -g)
        v9 = -(r.unsqueeze(-1) * g.unsqueeze(-2)).reshape(-1, 9)
        virial.scatter_add_(0, pj.unsqueeze(-1).expand_as(v9), v9)

    _acc(pi_rad, pj_rad, rij_rad, g_rad)
    _acc(pi_ang, pj_ang, rij_ang, g_ang)
    return forces, virial
