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

"""Descriptor internals: angular basis, gradients, and higher-body channels.

Self-contained math checks for the descriptor implementation — no GPUMD
fixtures (those live in test_gpumd_parity.py). Three groups:

  Angular basis (L = 1..8)
      ``angular_basis`` reproduces the old hand-coded L<=4 formula, and
      ``_compute_dblm_dhat`` matches both autograd and finite differences.

  Angular-weight gradient (q_222 / q_1111 / q_112)
      ``ops._angular_weight`` (the closed-form dEi/d(sum_fxyz) the analytical
      force path uses) matches autograd on the explicit q-vs-s polynomial that
      ``compute_descriptors_cached`` builds.

  4-body bispectrum (q_123 / q_233 / q_134, GPUMD PR #1517)
      Rotational invariance, bit-identical match to GPUMD's find_q polynomial,
      analytical gradient/force vs autograd, and nep.txt round-trip.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torchnep.ops as ops
from torchnep.ops import angular_basis, _compute_dblm_dhat, _angular_weight
from torchnep.constants import (C3B, C4B, C5B, C4B2,
                                Q123_TERMS, Q233_TERMS, Q134_TERMS,
                                C4B_123, C4B_233, C4B_134)
from torchnep.data import build_neighbor_list_np
from torchnep.model import NEPModel
from torchnep.nep import NEPCalculator

DTYPE = torch.float64


# ===========================================================================
# Angular basis: L = 1..8
# ===========================================================================

def _hand_coded_lmax4(x, y, z):
    """Old implementation, for regression check (L=1..4 only)."""
    x2, y2, z2 = x * x, y * y, z * z
    x2my2 = x2 - y2
    blm = []
    blm.extend([z, x, y])
    blm.extend([3.0*z2 - 1.0, x*z, y*z, x2my2, 2.0*x*y])
    blm.extend([
        (5.0*z2 - 3.0)*z, (5.0*z2 - 1.0)*x, (5.0*z2 - 1.0)*y,
        x2my2*z, 2.0*x*y*z, x*(x2 - 3.0*y2), y*(3.0*x2 - y2),
    ])
    blm.extend([
        (35.0*z2 - 30.0)*z2 + 3.0, (7.0*z2 - 3.0)*x*z,
        (7.0*z2 - 3.0)*y*z, (7.0*z2 - 1.0)*x2my2,
        (7.0*z2 - 1.0)*2.0*x*y, z*x*(x2 - 3.0*y2),
        z*y*(3.0*x2 - y2), x2my2*x2my2 - 4.0*x2*y2,
        4.0*x*y*x2my2,
    ])
    return torch.stack(blm, dim=-1)


def test_regression_lmax4():
    """New data-driven angular_basis matches old hand-coded formula for L <= 4."""
    torch.manual_seed(0)
    x = torch.randn(50, dtype=torch.float64)
    y = torch.randn(50, dtype=torch.float64)
    z = torch.randn(50, dtype=torch.float64)
    for L in [1, 2, 3, 4]:
        new = angular_basis(x, y, z, L)
        ref = _hand_coded_lmax4(x, y, z)
        num_lm_L = sum(2 * ll + 1 for ll in range(1, L + 1))
        diff = (new - ref[:, :num_lm_L]).abs().max().item()
        # Different summation order -> FP roundoff at the last few bits.
        assert diff < 1e-12, f"L={L}: new != hand-coded, max diff = {diff:.3e}"


@pytest.mark.parametrize("L_max", list(range(1, 9)))
def test_dblm_matches_autograd(L_max):
    """_compute_dblm_dhat matches PyTorch autograd on angular_basis."""
    torch.manual_seed(100 + L_max)
    n = 30
    x = torch.randn(n, dtype=torch.float64, requires_grad=True)
    y = torch.randn(n, dtype=torch.float64, requires_grad=True)
    z = torch.randn(n, dtype=torch.float64, requires_grad=True)

    blm = angular_basis(x, y, z, L_max)
    num_lm = blm.shape[1]
    ref = torch.zeros(n, num_lm, 3, dtype=torch.float64)
    for lm in range(num_lm):
        gx, gy, gz = torch.autograd.grad(
            blm[:, lm].sum(), [x, y, z], retain_graph=True)
        ref[:, lm, 0] = gx
        ref[:, lm, 1] = gy
        ref[:, lm, 2] = gz

    with torch.no_grad():
        got = _compute_dblm_dhat(x.detach(), y.detach(), z.detach(), L_max)

    diff = (got - ref).abs().max().item()
    rel = diff / max(ref.abs().max().item(), 1e-12)
    assert diff < 1e-10 or rel < 1e-10, (
        f"L_max={L_max}: dblm_dhat mismatch. max abs diff = {diff:.3e}, "
        f"rel = {rel:.3e}")


@pytest.mark.parametrize("L_max", list(range(1, 9)))
def test_dblm_matches_finite_diff(L_max):
    """Sanity: numerical FD check (independent of autograd)."""
    eps = 1e-6
    torch.manual_seed(200 + L_max)
    n = 8
    x = torch.randn(n, dtype=torch.float64) * 0.8
    y = torch.randn(n, dtype=torch.float64) * 0.8
    z = torch.randn(n, dtype=torch.float64) * 0.8

    analytical = _compute_dblm_dhat(x, y, z, L_max)  # (n, num_lm, 3)

    fd = torch.zeros_like(analytical)
    for axis, var in enumerate((x, y, z)):
        args_p = [var + eps if i == axis else v for i, v in enumerate((x, y, z))]
        args_m = [var - eps if i == axis else v for i, v in enumerate((x, y, z))]
        fd[:, :, axis] = (angular_basis(*args_p, L_max)
                          - angular_basis(*args_m, L_max)) / (2 * eps)

    diff = (analytical - fd).abs().max().item()
    rel = diff / max(fd.abs().max().item(), 1e-12)
    # FD is noisier (~eps**2 + rounding); allow 1e-6.
    assert diff < 1e-6 or rel < 1e-6, (
        f"L_max={L_max}: FD disagrees with analytical dblm. "
        f"max abs diff = {diff:.3e}, rel = {rel:.3e}")


def test_num_lm_shape():
    r"""Output width follows num_lm = \Sigma(2L+1) for L=1..L_max."""
    for L_max in range(1, 9):
        x = torch.randn(5, dtype=torch.float64)
        y = torch.randn(5, dtype=torch.float64)
        z = torch.randn(5, dtype=torch.float64)
        expected = sum(2 * L + 1 for L in range(1, L_max + 1))
        assert angular_basis(x, y, z, L_max).shape == (5, expected)
        assert _compute_dblm_dhat(x, y, z, L_max).shape == (5, expected, 3)


def test_rejects_oob():
    """l_max_3b > 8 must raise."""
    x = torch.randn(3, dtype=torch.float64)
    with pytest.raises(ValueError):
        angular_basis(x, x, x, 9)


# ===========================================================================
# Angular-weight gradient: q_222 / q_1111 / q_112
# ===========================================================================

def _q_112_from_s(s_lm, c4b2):
    """Per-atom q_112 from the L=1, L=2 entries of s. Mirrors the body of
    ``compute_descriptors_cached``. Returns shape (n_ap1,)."""
    s10, s11r, s11i = s_lm[:, 0], s_lm[:, 1], s_lm[:, 2]
    s20, s21r, s21i = s_lm[:, 3], s_lm[:, 4], s_lm[:, 5]
    s22r, s22i = s_lm[:, 6], s_lm[:, 7]
    cb = c4b2
    return (cb[0]*s10*s10*s20
            + cb[1]*s10*(s11r*s21r + s11i*s21i)
            + cb[2]*s20*(s11r*s11r + s11i*s11i)
            + cb[3]*s22r*(s11r*s11r - s11i*s11i)
            + cb[4]*s11r*s11i*s22i)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_q112_polynomial_gradient(seed):
    """Hand-derived q_112 gradient matches torch.autograd. Tests every
    (Fp, s_lm) entry — only the 8 relevant lm indices get non-zero weight."""
    torch.manual_seed(seed)
    n_ap1 = 5            # n_max_angular = 4 -> 5 channels
    num_lm = 24          # L=1..4 -> 3+5+7+9 = 24
    s = torch.randn(1, n_ap1, num_lm, dtype=DTYPE, requires_grad=True)
    Fp_block = torch.randn(1, n_ap1, dtype=DTYPE)
    c4b2 = torch.tensor(C4B2, dtype=DTYPE)

    q = _q_112_from_s(s[0], c4b2)               # (n_ap1,)
    energy = (Fp_block[0] * q).sum()
    grad_auto, = torch.autograd.grad(energy, s)

    dim_r = 5  # arbitrary radial size — doesn't enter q_112 derivative
    dim = dim_r + 4 * n_ap1 + n_ap1   # 3b uses 4 L's; only q_112 block follows
    Fp_full = torch.zeros(1, dim, dtype=DTYPE)
    off = dim_r + 4 * n_ap1   # offset of q_112 block (no q_222 / q_1111)
    Fp_full[:, off:off + n_ap1] = Fp_block

    w = _angular_weight(
        Fp_full, s.detach(), dim_r, n_ap1, l_max_3b=4,
        has_q_222=0, has_q_1111=0, has_q_112=1,
        c3b_coeffs=torch.tensor(C3B[:num_lm], dtype=DTYPE),
        c4b_coeffs=torch.tensor(C4B, dtype=DTYPE),
        c5b_coeffs=torch.tensor(C5B, dtype=DTYPE),
        c4b2_coeffs=c4b2)

    # Only q_112 is enabled, so the 3-body slot is zero and the output equals
    # dE/ds exactly on the 8 contributing lm indices.
    assert torch.allclose(w[:, :, :8], grad_auto[:, :, :8], atol=1e-12)
    assert torch.allclose(w[:, :, 8:], torch.zeros_like(w[:, :, 8:]), atol=1e-12)


@pytest.mark.parametrize("flag_tuple", [
    (1, 0, 0),     # q_222 only (baseline path)
    (0, 0, 1),     # q_112 in isolation
    (0, 1, 0),     # q_1111 in isolation (legacy compatibility)
    (1, 1, 1),     # everything on
])
def test_full_angular_weight(flag_tuple):
    """Full multi-block ``_angular_weight`` matches autograd on the explicit
    polynomial that ``compute_descriptors_cached`` builds. Catches off-by-one
    offset bugs (each enabled flag adds one n_ap1 slot)."""
    torch.manual_seed(7 * sum(flag_tuple) + 1)
    has_222, has_1111, has_112 = flag_tuple
    n_ap1 = 5
    l_max_3b = 4
    num_lm = sum(2 * L + 1 for L in range(1, l_max_3b + 1))  # 24

    s = torch.randn(2, n_ap1, num_lm, dtype=DTYPE, requires_grad=True)

    dim_r = 7  # arbitrary
    dim = (dim_r + l_max_3b * n_ap1
           + (has_222 + has_1111 + has_112) * n_ap1)
    Fp = torch.randn(2, dim, dtype=DTYPE)

    c3b  = torch.tensor(C3B[:num_lm], dtype=DTYPE)
    c4b  = torch.tensor(C4B,  dtype=DTYPE)
    c5b  = torch.tensor(C5B,  dtype=DTYPE)
    c4b2 = torch.tensor(C4B2, dtype=DTYPE)

    # Build the full energy as exec'd by compute_descriptors_cached, then
    # autograd through s.
    parts = []
    for li in range(l_max_3b):
        L = li + 1
        nt = 2 * L + 1
        st = L * L - 1
        c = c3b[st:st + nt]
        sb2 = s[:, :, st:st + nt] ** 2
        ql = c[0] * sb2[:, :, 0]
        if nt > 1:
            ql = ql + 2.0 * (c[1:] * sb2[:, :, 1:]).sum(-1)
        parts.append(ql)
    q_3b = torch.stack(parts, dim=-1).transpose(1, 2).reshape(s.shape[0], -1)
    q_list = [q_3b]

    s10, s11r, s11i = s[:, :, 0], s[:, :, 1], s[:, :, 2]
    s20, s21r, s21i = s[:, :, 3], s[:, :, 4], s[:, :, 5]
    s22r, s22i = s[:, :, 6], s[:, :, 7]

    if has_222:
        q4 = (c4b[0]*s20**3
              + c4b[1]*s20*(s21r**2 + s21i**2)
              + c4b[2]*s20*(s22r**2 + s22i**2)
              + c4b[3]*s22r*(s21i**2 - s21r**2)
              + c4b[4]*s21r*s21i*s22i)
        q_list.append(q4)
    if has_1111:
        s0sq = s10**2
        s1sq = s11r**2 + s11i**2
        q5 = c5b[0]*s0sq**2 + c5b[1]*s0sq*s1sq + c5b[2]*s1sq**2
        q_list.append(q5)
    if has_112:
        q112 = (c4b2[0]*s10*s10*s20
                + c4b2[1]*s10*(s11r*s21r + s11i*s21i)
                + c4b2[2]*s20*(s11r*s11r + s11i*s11i)
                + c4b2[3]*s22r*(s11r*s11r - s11i*s11i)
                + c4b2[4]*s11r*s11i*s22i)
        q_list.append(q112)

    q = torch.cat(q_list, dim=-1)                                # (N, dim)
    # Radial block doesn't depend on s, so dE/ds has no radial contribution.
    pad = torch.zeros(s.shape[0], dim_r, dtype=DTYPE)
    q_full = torch.cat([pad, q], dim=-1)
    energy = (Fp * q_full).sum()
    grad_auto, = torch.autograd.grad(energy, s)

    w = _angular_weight(Fp, s.detach(), dim_r, n_ap1, l_max_3b,
                        has_222, has_1111, has_112, c3b, c4b, c5b, c4b2)

    assert torch.allclose(w, grad_auto, atol=1e-12), (
        f"flags={flag_tuple}  max diff = {(w - grad_auto).abs().max().item():.3e}")


# ===========================================================================
# 4-body bispectrum channels: q_123 / q_233 / q_134 (GPUMD PR #1517)
# ===========================================================================

# name -> (term table, l_max used to build the moments, number of s-moments).
# q_123 / q_233 need L=3 moments (15 of them); q_134 also uses L=4 (24).
CHANNELS = {
    "q_123": (Q123_TERMS, 3, 15),
    "q_233": (Q233_TERMS, 3, 15),
    "q_134": (Q134_TERMS, 4, 24),
}


def _rand_rotation(rng):
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _moments(dirs, weights, l_max=3):
    x, y, z = (torch.tensor(dirs[:, i]) for i in range(3))
    blm = ops.angular_basis(x, y, z, l_max).numpy()
    return weights @ blm


def _eval(s, terms):
    return float(ops._eval_extra(torch.tensor(s).view(1, 1, -1), terms)[0, 0])


# --- GPUMD PR #1517 reference polynomials (verbatim from find_q) -------------

def _gpumd_q123(s):
    C = C4B_123
    return (C[6]*(s[12]*s[2]*s[4] - s[11]*s[2]*s[5] + s[1]*s[11]*s[4] + s[1]*s[12]*s[5])
            + C[5]*(s[0]*s[11]*s[6] + s[0]*s[12]*s[7])
            + C[3]*(s[14]*s[2]*s[6] - s[13]*s[2]*s[7] + s[1]*s[13]*s[6] + s[1]*s[14]*s[7])
            + C[4]*(s[10]*s[0]*s[5] + s[0]*s[4]*s[9])
            + C[1]*(s[10]*s[2]*s[3] + s[0]*s[3]*s[8] + s[1]*s[3]*s[9])
            + C[0]*(s[10]*s[2]*s[6] - s[10]*s[1]*s[7] - s[2]*s[7]*s[9] - s[1]*s[6]*s[9])
            + C[2]*(-s[2]*s[5]*s[8] - s[1]*s[4]*s[8]))


def _gpumd_q233(s):
    C = C4B_233
    return (C[0]*(s[3]*s[8]*s[8])
            + C[1]*(s[10]*s[10]*s[3] + s[3]*s[9]*s[9])
            + C[2]*(-s[10]*s[10]*s[6] + s[6]*s[9]*s[9])
            + C[3]*(s[4]*s[8]*s[9] + s[10]*s[5]*s[8])
            + C[4]*(-s[13]*s[13]*s[3] - s[14]*s[14]*s[3])
            + C[5]*(-s[14]*s[7]*s[9] - s[13]*s[6]*s[9] - s[10]*s[14]*s[6] + s[10]*s[13]*s[7])
            + C[6]*(s[10]*s[7]*s[9])
            + C[7]*(-s[11]*s[6]*s[8] - s[12]*s[7]*s[8])
            + C[8]*(s[11]*s[4]*s[9] + s[12]*s[5]*s[9] + s[10]*s[12]*s[4] - s[10]*s[11]*s[5])
            + C[9]*(s[12]*s[14]*s[4] + s[11]*s[14]*s[5] + s[13]*s[11]*s[4] - s[13]*s[12]*s[5]))


def _gpumd_q134(s):
    C = C4B_134
    return (C[0]*(-s[10]*s[15]*s[2] - s[1]*s[15]*s[9])
            + C[1]*(s[0]*s[15]*s[8])
            + C[2]*(-s[1]*s[13]*s[18] - s[1]*s[14]*s[19] - s[2]*s[14]*s[18] + s[2]*s[13]*s[19])
            + C[3]*(-s[10]*s[18]*s[2] + s[1]*s[10]*s[19] + s[1]*s[18]*s[9] + s[2]*s[19]*s[9])
            + C[4]*(s[1]*s[16]*s[8] + s[2]*s[17]*s[8])
            + C[5]*(s[0]*s[10]*s[17] + s[0]*s[16]*s[9] - s[1]*s[11]*s[16] - s[1]*s[12]*s[17]
                    - s[2]*s[12]*s[16] + s[2]*s[11]*s[17])
            + C[6]*(s[1]*s[13]*s[22] + s[1]*s[14]*s[23] - s[2]*s[14]*s[22] + s[2]*s[13]*s[23])
            + C[7]*(s[0]*s[11]*s[18] + s[0]*s[12]*s[19])
            + C[8]*(s[0]*s[13]*s[20] + s[0]*s[14]*s[21])
            + C[9]*(s[1]*s[11]*s[20] + s[1]*s[12]*s[21] - s[2]*s[12]*s[20] + s[2]*s[11]*s[21]))


@pytest.mark.parametrize("name", list(CHANNELS))
def test_rotational_invariance(name):
    terms, l_max, _ = CHANNELS[name]
    rng = np.random.default_rng(2026)
    worst = 0.0
    for _ in range(150):
        dirs = rng.standard_normal((10, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        w = rng.standard_normal(10)
        R = _rand_rotation(rng)
        q0 = _eval(_moments(dirs, w, l_max), terms)
        q1 = _eval(_moments(dirs @ R.T, w, l_max), terms)
        worst = max(worst, abs(q1 - q0) / (abs(q0) + 1e-12))
    assert worst < 1e-10, f"{name}: rotational variance {worst:.2e}"


@pytest.mark.parametrize("name,ref", [("q_123", _gpumd_q123), ("q_233", _gpumd_q233),
                                      ("q_134", _gpumd_q134)])
def test_matches_gpumd_polynomial(name, ref):
    """torchnep term tables are bit-identical to GPUMD's find_q polynomial."""
    terms, _, n_s = CHANNELS[name]
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(2000):
        s = rng.standard_normal(n_s)
        worst = max(worst, abs(_eval(s, terms) - ref(s)))
    assert worst < 1e-12, f"{name}: max |torchnep - GPUMD| = {worst:.2e}"


@pytest.mark.parametrize("name", list(CHANNELS))
def test_extra_grad_matches_autograd(name):
    terms, _, n_s = CHANNELS[name]
    torch.manual_seed(7)
    s = torch.randn(3, 4, n_s, dtype=DTYPE, requires_grad=True)
    q = ops._eval_extra(s, terms)
    grad_auto, = torch.autograd.grad(q.sum(), s)
    grad_ana = ops._extra_grad(s.detach(), terms)
    assert torch.allclose(grad_ana, grad_auto, atol=1e-12)


def _build_model(lmax, seed=0):
    torch.manual_seed(seed)
    cfg = {"num_types": 2, "type_names": ["Si", "O"],
           "cutoff_radial": 6.0, "cutoff_angular": 5.0,
           "n_max_radial": 3, "n_max_angular": 3,
           "basis_size_radial": 6, "basis_size_angular": 6,
           "l_max": lmax, "neuron": 16}
    m = NEPModel(cfg).to(DTYPE)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.1)
        m.q_scaler.uniform_(0.5, 1.5)
    return m


def _random_batch(N=40, seed=0):
    rng = np.random.default_rng(seed)
    cell = np.eye(3) * 9.0
    pos = rng.random((N, 3)) * 9.0
    species = rng.integers(0, 2, N)
    pi, pj, rij = build_neighbor_list_np(pos, cell, 6.0)
    dij = np.linalg.norm(rij, axis=1)
    at = torch.tensor(species, dtype=torch.long)
    rc_r, rc_a, br, ba = 6.0, 5.0, 6, 6
    rm, am = dij < rc_r, dij < rc_a
    pir, pjr, rr = (torch.tensor(pi[rm]), torch.tensor(pj[rm]),
                    torch.tensor(rij[rm], dtype=DTYPE))
    pia, pja, ra = (torch.tensor(pi[am]), torch.tensor(pj[am]),
                    torch.tensor(rij[am], dtype=DTYPE))
    dr, da = torch.norm(rr, dim=-1), torch.norm(ra, dim=-1)
    fkr, fkpr = ops.chebyshev_basis_and_deriv(dr, rc_r, br)
    fka, fkpa = ops.chebyshev_basis_and_deriv(da, rc_a, ba)
    d12r, d12a = 1.0 / dr.clamp(min=1e-10), 1.0 / da.clamp(min=1e-10)
    blm = ops.angular_basis(ra[:, 0] * d12a, ra[:, 1] * d12a, ra[:, 2] * d12a, 4)
    batch = {"N": N, "num_structures": 1, "atom_types": at,
             "struct_idx": torch.zeros(N, dtype=torch.long),
             "pair_i_rad": pir, "pair_j_rad": pjr, "rij_rad": rr,
             "fk_rad": fkr, "fkp_rad": fkpr, "d12inv_rad": d12r,
             "pair_i_ang": pia, "pair_j_ang": pja, "rij_ang": ra,
             "fk_ang": fka, "fkp_ang": fkpa, "d12inv_ang": d12a, "blm": blm}
    return batch, (pi, pj, rij, dij, at)


def test_analytical_force_vs_autograd():
    """All channels on: analytical force/virial == autograd-on-rij."""
    # 7 fields: L_3b, q_222, q_1111, q_112, q_123, q_233, q_134
    m = _build_model([4, 1, 0, 1, 1, 1, 1])
    batch, (pi, pj, rij, dij, at) = _random_batch()
    N = batch["N"]
    with torch.enable_grad():
        r = m.compute_properties_cached(batch, need_forces=True,
                                        need_virial=True, backend="loop")
    f_ana = r["forces"].detach().numpy()
    v_ana = r["virial"].detach().numpy()

    rm, am = dij < 6.0, dij < 5.0
    rr = torch.tensor(rij[rm], dtype=DTYPE, requires_grad=True)
    ra = torch.tensor(rij[am], dtype=DTYPE, requires_grad=True)
    pir, pjr = torch.tensor(pi[rm]), torch.tensor(pj[rm])
    pia, pja = torch.tensor(pi[am]), torch.tensor(pj[am])
    Ei = m.forward(rr, ra, pir, pjr, pia, pja, at, N, backend="loop")
    g = torch.autograd.grad(Ei.sum(), [rr, ra])
    f_auto, v_auto = ops.accumulate_forces_virial(
        N, pir, pjr, rr.detach(), g[0], pia, pja, ra.detach(), g[1],
        DTYPE, torch.device("cpu"))
    assert np.abs(f_ana - f_auto.numpy()).max() < 1e-9
    assert np.abs(v_ana - v_auto.numpy()).max() < 1e-9


def test_nep_txt_round_trip(tmp_path):
    m = _build_model([4, 1, 0, 1, 1, 1, 1])
    p = tmp_path / "nep.txt"
    m.save_nep_txt(str(p), max_NN_radial=100, max_NN_angular=60)
    text = p.read_text()
    # 7-field GPUMD form. Field 2 uses the legacy ``has_q_222 ? 2 : 0``
    # encoding so older GPUMD builds still load it.
    assert "l_max 4 2 0 1 1 1 1" in text
    calc = NEPCalculator(str(p), dtype=DTYPE)
    assert (calc.has_q_123, calc.has_q_233, calc.has_q_134) == (1, 1, 1)
    assert calc.dim == m.dim
    rng = np.random.default_rng(1)
    pos = rng.random((12, 3)) * 8.0
    species = ["Si" if i % 2 else "O" for i in range(12)]
    res = calc.compute(species=species, positions=pos, cell=np.eye(3) * 8.0)
    assert np.isfinite(res["energy"].numpy()).all()
    assert np.isfinite(res["forces"].numpy()).all()


def test_lmax_guard():
    """q_123 / q_233 need l_max_3b >= 3 (they use L=3 moments)."""
    with pytest.raises(ValueError):
        # 6-field PR #1519 layout: q_123 at field 5
        _build_model([2, 1, 0, 0, 1, 0])  # l_max_3b=2, q_123 on


def test_lmax_guard_q134():
    """q_134 needs l_max_3b >= 4 (it uses L=4 moments)."""
    with pytest.raises(ValueError):
        # q_134 at field 7; l_max_3b=3 is insufficient
        _build_model([3, 1, 0, 0, 0, 0, 1])


def test_off_by_default():
    # 4-field GPUMD-core only — q_123/q_233/q_134 default to 0
    m = _build_model([4, 1, 0, 1])
    assert (m.has_q_123, m.has_q_233, m.has_q_134) == (0, 0, 0)
    assert m.dim_angular_123 == 0 and m.dim_angular_233 == 0
    assert m.dim_angular_134 == 0


def test_q1111_redundancy_warns():
    """has_q_1111=1 still works (backward compat) but warns it's redundant."""
    with pytest.warns(UserWarning, match="has_q_1111.*redundant"):
        _build_model([4, 1, 1, 0])  # q_1111 on
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _build_model([4, 1, 0, 0])


# ===========================================================================
# End-to-end rotational invariance — every supported higher-body channel
# ===========================================================================

# Each supported higher-body flag and the single-flag l_max config enabling it
# (fields: L_3b, q_222, q_1111, q_112, q_123, q_233, q_134). An angular
# descriptor must be invariant under a global rotation of the neighbourhood;
# a wrong term or coefficient in any channel breaks that, so this is a strong
# self-contained correctness check for all six channels through the real
# predict pipeline (NEPCalculator.get_descriptor).
HIGHER_BODY = {
    "q_222":  [4, 1, 0, 0],
    "q_1111": [4, 0, 1, 0],
    "q_112":  [4, 0, 0, 1],
    "q_123":  [4, 0, 0, 0, 1, 0, 0],
    "q_233":  [4, 0, 0, 0, 0, 1, 0],
    "q_134":  [4, 0, 0, 0, 0, 0, 1],
}


def _cluster(seed=0, n=14, box=30.0):
    """Finite cluster centred in a large box (no periodic neighbours within
    the cutoff), so a global rotation about the centre leaves the neighbour
    set unchanged."""
    rng = np.random.default_rng(seed)
    centre = np.array([box / 2] * 3)
    pos = centre + rng.normal(0.0, 2.5, (n, 3))
    species = ["Si" if i % 2 else "O" for i in range(n)]
    return pos, species, np.eye(3) * box


@pytest.mark.parametrize("name", list(HIGHER_BODY))
def test_descriptor_rotational_invariance(name, tmp_path):
    """Each higher-body channel yields a rotationally-invariant descriptor
    through the real predict pipeline."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # q_1111 redundancy notice
        m = _build_model(HIGHER_BODY[name])
        p = tmp_path / "nep.txt"
        m.save_nep_txt(str(p), max_NN_radial=200, max_NN_angular=200)
    calc = NEPCalculator(str(p), dtype=DTYPE)

    pos, species, cell = _cluster()
    centre = np.diag(cell) / 2
    R = _rand_rotation(np.random.default_rng(123))
    pos_rot = (pos - centre) @ R.T + centre

    d0 = calc.get_descriptor(species, pos, cell)
    d1 = calc.get_descriptor(species, pos_rot, cell)
    worst = float(np.abs(d0 - d1).max())
    assert worst < 1e-9, f"{name}: descriptor not rotation-invariant ({worst:.2e})"
