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

"""End-to-end parity with GPUMD, plus gradient self-consistency.

Every fixture in ``_common.FIXTURES`` ships a ``data/<name>.gpumd.npz`` blob
baked from GPUMD's ``nep`` binary in prediction mode (see ``bake_fixtures.py``).

Three checks, each over (fixture * device * dtype):

  test_forward_vs_gpumd
      Per-atom E, F, V (6 comp.) and the scaled descriptor match the frozen
      GPUMD reference for every frame. The CrCoNi fixture is multi-frame:
      compressed/rattled frames push pairs to/below the ZBL inner cutoff,
      so the repulsive ZBL branch (forces up to ~120 eV/A) is covered here.
      GPUMD runs float32 and writes %g, so float32 torchnep is the tightest
      comparison; both dtypes agree to ~1e-5.

  test_analytical_vs_autograd
      The closed-form analytical force/virial path matches autograd-on-rij in
      the same context (formula re-ordering only). Run on the original and a
      strongly-compressed frame so the analytical ZBL gradient is exercised.

  test_train_vs_predict
      The training path (NEPModel.compute_properties_cached) matches the
      predict path (NEPCalculator.compute_batch) for identical weights. The
      two dispatch to different kernels under different autograd contexts, so
      the float64 bound is looser (~1e-6) — accumulation order, not a bug.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from torchnep import ops
from torchnep.data import read_xyz, build_neighbor_list_np
from torchnep.nep import NEPCalculator
from torchnep.model import NEPModel
from _common import (DTYPE_MAP, NP_DTYPE_MAP, FIXTURES, devices, dtypes,
                     load_reference)


# Single-frame neighbor-list + batch builders. These mirror what the training
# pipeline produces and what NEPCalculator.compute_batch consumes, but build a
# single hand-made batch — only the parity tests below need them, so they live
# here rather than in the shipped package.

def _preprocess_for_prediction(frames, calc, np_dtype):
    """Build neighbor lists for a list of frames. Returns structure dicts
    with the same schema the training pipeline produces."""
    rc_rad, rc_ang = calc.rc_radial, calc.rc_angular
    max_rc = max(rc_rad, rc_ang)

    structures = []
    for frame in frames:
        positions = np.asarray(frame["positions"], dtype=np_dtype)
        cell = np.asarray(frame["cell"], dtype=np_dtype)
        atom_types = np.array(
            [calc.type_names.index(s) for s in frame["species"]],
            dtype=np.int64)
        pair_i, pair_j, rij = build_neighbor_list_np(positions, cell, max_rc)
        dij = np.linalg.norm(rij, axis=1)
        structures.append({
            "natoms": frame["natoms"],
            "atom_types": atom_types,
            "pair_i_rad": pair_i[dij < rc_rad],
            "pair_j_rad": pair_j[dij < rc_rad],
            "rij_rad":    rij[dij < rc_rad],
            "pair_i_ang": pair_i[dij < rc_ang],
            "pair_j_ang": pair_j[dij < rc_ang],
            "rij_ang":    rij[dij < rc_ang],
            "energy": frame.get("energy"),
            "forces": frame.get("forces"),
            "virial": frame.get("virial"),
        })
    return structures


def _build_batch(structures, indices, calc, dtype, device):
    """Collate a list of structure indices into a GPU batch with cached basis,
    matching the dict shape NEPCalculator.compute_batch expects."""
    rc_rad, rc_ang = calc.rc_radial, calc.rc_angular
    basis_r, basis_a = calc.basis_size_radial, calc.basis_size_angular
    l_max_3b, num_lm = calc.l_max_3b, calc.num_lm

    natoms_list = [structures[i]["natoms"] for i in indices]
    N_total = sum(natoms_list)
    B = len(indices)
    offsets = [0]
    for n in natoms_list:
        offsets.append(offsets[-1] + n)

    atom_types = torch.tensor(
        np.concatenate([structures[i]["atom_types"] for i in indices]),
        dtype=torch.long, device=device)
    struct_idx = torch.cat([
        torch.full((natoms_list[k],), k, dtype=torch.long, device=device)
        for k in range(B)])

    def _cat_int(key):
        return torch.tensor(
            np.concatenate([structures[indices[k]][key] + offsets[k]
                            for k in range(B)]).astype(np.int64),
            dtype=torch.long, device=device)

    def _cat_rij(key):
        return torch.tensor(
            np.concatenate([structures[indices[k]][key] for k in range(B)]),
            dtype=dtype, device=device)

    pi_r, pj_r = _cat_int("pair_i_rad"), _cat_int("pair_j_rad")
    rij_r = _cat_rij("rij_rad")
    pi_a, pj_a = _cat_int("pair_i_ang"), _cat_int("pair_j_ang")
    rij_a = _cat_rij("rij_ang")

    dr = torch.norm(rij_r, dim=-1)
    fk_r, fkp_r = ops.chebyshev_basis_and_deriv(dr, rc_rad, basis_r)
    d12inv_r = 1.0 / dr.clamp(min=1e-10)

    if rij_a.shape[0] > 0:
        da = torch.norm(rij_a, dim=-1)
        fk_a, fkp_a = ops.chebyshev_basis_and_deriv(da, rc_ang, basis_a)
        d12inv_a = 1.0 / da.clamp(min=1e-10)
        blm = ops.angular_basis(rij_a[:, 0] * d12inv_a,
                                rij_a[:, 1] * d12inv_a,
                                rij_a[:, 2] * d12inv_a, l_max_3b)
    else:
        fk_a = torch.zeros(0, basis_a + 1, dtype=dtype, device=device)
        fkp_a = torch.zeros(0, basis_a + 1, dtype=dtype, device=device)
        d12inv_a = torch.zeros(0, dtype=dtype, device=device)
        blm = torch.zeros(0, num_lm, dtype=dtype, device=device)

    return {
        "N": N_total, "num_structures": B,
        "atom_types": atom_types, "struct_idx": struct_idx,
        "pair_i_rad": pi_r, "pair_j_rad": pj_r, "rij_rad": rij_r,
        "fk_rad": fk_r, "fkp_rad": fkp_r, "d12inv_rad": d12inv_r,
        "pair_i_ang": pi_a, "pair_j_ang": pj_a, "rij_ang": rij_a,
        "fk_ang": fk_a, "fkp_ang": fkp_a, "d12inv_ang": d12inv_a,
        "blm": blm,
    }


# Tolerance vs the GPUMD fixture. GPUMD computes in float32 and writes %g
# (6 sig figs), so the floor is relative ~1e-5; near-zero / cancelling
# components (e.g. forces from large opposing ZBL terms) are floored by atol.
# Both torchnep dtypes agree with GPUMD at this level.
RTOL, ATOL = 1e-5, 2e-4
# (A) analytical vs autograd — same context, formula re-ordering only, so the
# bound is relative (forces reach ~120 eV/A on the ZBL frames). (B) train vs
# predict — different autograd contexts dispatch to different kernels with
# subtly different accumulation order, hence the looser float64 bound.
TOL_ANA = {"float64": 1e-9, "float32": 1e-4}   # relative
TOL_TVP = {"float64": 5e-6, "float32": 5e-3}   # absolute


def _ids(seq, prefix):
    return [f"{prefix}={x}" for x in seq]


_FX = pytest.mark.parametrize(
    "fixture", FIXTURES, ids=_ids([f["name"] for f in FIXTURES], "fx"))
_DEV = pytest.mark.parametrize("device", devices(), ids=_ids(devices(), "dev"))
_DT = pytest.mark.parametrize("dtype_s", dtypes(), ids=_ids(dtypes(), "dt"))


@_FX
@_DEV
@_DT
def test_forward_vs_gpumd(fixture, device, dtype_s):
    """E/F/V/Descriptor agree with the baked GPUMD reference, all frames."""
    ref = load_reference(fixture["ref"])
    frames = read_xyz(str(fixture["xyz"]))

    torch_dtype = DTYPE_MAP[dtype_s]
    np_dtype = NP_DTYPE_MAP[dtype_s]
    calc = NEPCalculator(str(fixture["nep"]), dtype=torch_dtype, device=device)

    # Build pre-cached batches frame-by-frame to keep memory bounded.
    structures = _preprocess_for_prediction(frames, calc, np_dtype)
    natoms = np.asarray([s["natoms"] for s in structures], dtype=np.int64)
    N_total = int(natoms.sum())

    E_pa = np.empty(len(frames), dtype=np.float64)
    V_pa = np.empty((len(frames), 6), dtype=np.float64)
    F = np.empty((N_total, 3), dtype=np.float64)
    D = np.empty((N_total, calc.dim), dtype=np.float64)

    off = 0
    for i, s in enumerate(structures):
        batch = _build_batch([s], [0], calc, torch_dtype, torch.device(device))
        with torch.no_grad():
            r = calc.compute_batch(batch)
        n = s["natoms"]
        E_pa[i] = r["Etot"].item() / n
        # Virial: sum per-atom 9-vector, fold to GPUMD's 6 entries, divide by Na.
        v9 = r["virial"].sum(0).cpu().numpy()
        # GPUMD order (xx, yy, zz, xy, yz, zx) from the row-major 3*3 sum.
        V_pa[i] = np.array([v9[0], v9[4], v9[8],
                            v9[1], v9[5], v9[6]]) / n
        F[off:off + n] = r["forces"].cpu().numpy()
        D[off:off + n] = r["descriptor"].cpu().numpy()
        off += n

    checks = [("E", E_pa, ref["E_per_atom"]), ("F", F, ref["F"]),
              ("V", V_pa, ref["V_per_atom"]), ("D", D, ref["D_per_atom"])]
    for name, pred, rf in checks:
        if rf is None:
            continue
        d = float(np.abs(pred - rf).max())
        print(f"\n[{name} {fixture['name']:8s} {device:4s} {dtype_s:7s}] max|d|={d:.2e}")
        assert np.allclose(pred, rf, rtol=RTOL, atol=ATOL), (
            f"[{name} {fixture['name']} {device} {dtype_s}] max|d|={d:.2e} "
            f"exceeds rtol={RTOL}, atol={ATOL}")


# Original frame (index 0) and the rattled, strongly-compressed last frame
# (deepest into the ZBL repulsion) — covers both regimes for the gradient path.
_FRAME_IDX = [0, -1]


@_FX
@_DEV
@_DT
@pytest.mark.parametrize("frame_idx", _FRAME_IDX, ids=_ids(_FRAME_IDX, "frame"))
def test_analytical_vs_autograd(fixture, device, dtype_s, frame_idx):
    """Analytical force / virial path matches autograd on pair vectors."""
    torch_dtype = DTYPE_MAP[dtype_s]
    np_dtype = NP_DTYPE_MAP[dtype_s]
    calc = NEPCalculator(str(fixture["nep"]), dtype=torch_dtype, device=device)
    frame = read_xyz(str(fixture["xyz"]))[frame_idx]

    structures = _preprocess_for_prediction([frame], calc, np_dtype)
    batch = _build_batch(structures, [0], calc, torch_dtype, torch.device(device))
    with torch.no_grad():
        r_ana = calc.compute_batch(batch)
    f_ana = r_ana["forces"].cpu().double().numpy()
    v_ana = r_ana["virial"].cpu().double().numpy()

    r_grad = calc.compute(
        species=list(frame["species"]),
        positions=np.asarray(frame["positions"], dtype=np_dtype),
        cell=np.asarray(frame["cell"], dtype=np_dtype),
    )
    f_grad = r_grad["forces"].cpu().double().numpy()
    v_grad = r_grad["virial"].cpu().double().numpy()

    # Relative bound: ZBL forces reach ~120 eV/A, so scale by the magnitude.
    scaleF = np.abs(f_grad).max() + 1.0
    scaleV = np.abs(v_grad).max() + 1.0
    dF = float(np.abs(f_ana - f_grad).max()) / scaleF
    dV = float(np.abs(v_ana - v_grad).max()) / scaleV

    tol = TOL_ANA[dtype_s]
    msg = (f"[{fixture['name']:8s} {device:4s} {dtype_s:7s} frame={frame_idx}]  "
           f"relF={dF:.2e}  relV={dV:.2e}")
    print("\n" + msg)
    assert dF < tol, f"{msg}  (tol={tol})"
    assert dV < tol, f"{msg}  (tol={tol})"


def _model_from_calc(calc: NEPCalculator, device) -> NEPModel:
    """Mirror a calculator's trained weights into a fresh NEPModel so the
    training-path and predict-path forwards use identical parameters."""
    config = {
        "type_names":          list(calc.type_names),
        "num_types":           calc.num_types,
        "cutoff_radial":       calc.rc_radial,
        "cutoff_angular":      calc.rc_angular,
        "n_max_radial":        calc.n_max_radial,
        "n_max_angular":       calc.n_max_angular,
        "basis_size_radial":   calc.basis_size_radial,
        "basis_size_angular":  calc.basis_size_angular,
        "l_max":               [calc.l_max_3b, calc.has_q_222, calc.has_q_1111,
                                calc.has_q_112],
        "neuron":              calc.num_neurons,
    }
    if calc.has_zbl:
        config["zbl"] = calc.zbl_rc_outer
        if calc.zbl_typewise_factor is not None:
            config["typewise_cutoff_zbl_factor"] = calc.zbl_typewise_factor

    model = NEPModel(config).to(calc.dtype).to(device)
    for t in range(calc.num_types):
        model.fitting_nets[t].w0.data.copy_(calc.w0[t].T)
        model.fitting_nets[t].b0.data.copy_(calc.b0[t])
        model.fitting_nets[t].w1.data.copy_(calc.w1[t])
    model.b1.data.copy_(calc.b1)
    model.c_param_2.data.copy_(calc.c2)
    if calc.l_max_3b > 0:
        model.c_param_3.data.copy_(calc.c3)
    model.q_scaler.data.copy_(calc.q_scaler)
    return model


@_FX
@_DEV
@_DT
def test_train_vs_predict(fixture, device, dtype_s):
    """Training-path forward agrees with the predict-path forward."""
    torch_dtype = DTYPE_MAP[dtype_s]
    np_dtype = NP_DTYPE_MAP[dtype_s]
    calc = NEPCalculator(str(fixture["nep"]), dtype=torch_dtype, device=device)
    model = _model_from_calc(calc, device)
    frame = read_xyz(str(fixture["xyz"]))[0]

    structures = _preprocess_for_prediction([frame], calc, np_dtype)
    batch = _build_batch(structures, [0], calc, torch_dtype, torch.device(device))

    with torch.enable_grad():
        r_train = model.compute_properties_cached(
            batch, need_forces=True, need_virial=True, backend="loop")
    f_train = r_train["forces"].detach().cpu().double().numpy()
    v_train = r_train["virial"].detach().cpu().double().numpy()

    with torch.no_grad():
        r_pred = calc.compute_batch(batch)
    f_pred = r_pred["forces"].cpu().double().numpy()
    v_pred = r_pred["virial"].cpu().double().numpy()

    dF = float(np.abs(f_train - f_pred).max())
    dV = float(np.abs(v_train - v_pred).max())

    tol = TOL_TVP[dtype_s]
    msg = (f"[{fixture['name']:8s} {device:4s} {dtype_s:7s}]  "
           f"|dF|={dF:.2e}  |dV|={dV:.2e}")
    print("\n" + msg)
    assert dF < tol, f"{msg}  (F tol={tol})"
    assert dV < tol, f"{msg}  (V tol={tol})"
