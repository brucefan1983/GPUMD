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
NEP training with PyTorch — single-GPU / CPU entry point.

Use ``train_nep`` launched with plain ``python`` for one-GPU, CPU, or Mac
workloads. For multi-GPU training, use ``train_nep_sharded`` launched with
``torchrun`` — it shards the dataset across ranks so each GPU only holds
1/world_size of the structures, enabling datasets much larger than any one
card's memory.

    # single GPU / CPU / Mac
    python run_train.py                        # calls train_nep(...)

    # multi-GPU, single node
    torchrun --standalone --nproc_per_node=N run_train.py   # train_nep_sharded

    # multi-node (via SLURM)
    see example/run_multi_node.sbatch
"""

import contextlib
import os
import platform
import time
import torch
import numpy as np
from datetime import datetime
from typing import List, Dict
from torch.optim.swa_utils import AveragedModel

from .model import NEPModel, slim_model
from .data import read_xyz, parse_nep_in, build_neighbor_list_np
from . import ops
from . import __version__
from .predict import predict_from_store


# ---------------------------------------------------------------------------
# Banner & environment info
# ---------------------------------------------------------------------------

_BANNER = r"""
████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗███╗   ██╗███████╗██████╗
╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║████╗  ██║██╔════╝██╔══██╗
   ██║   ██║   ██║██████╔╝██║     ███████║██╔██╗ ██║█████╗  ██████╔╝
   ██║   ██║   ██║██╔══██╗██║     ██╔══██║██║╚██╗██║██╔══╝  ██╔═══╝
   ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║██║ ╚████║███████╗██║
   ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝
"""

_AUTHOR = "Yongchao Wu, yongchao.wu@aalto.fi"


def _backend_info(dev: torch.device, world_size: int = 1) -> List[str]:
    """Describe compute backend for the startup banner.

    ``world_size`` is used by ``train_nep_sharded`` to surface DDP info;
    single-GPU ``train_nep`` keeps the default of 1.

    PyTorch-ROCm exposes AMD GPUs through the ``torch.cuda`` namespace, so
    the cuda branch also handles ROCm. Intel GPUs use ``torch.xpu``.
    """
    lines = []
    if dev.type == "cuda":
        n_visible = torch.cuda.device_count()
        is_rocm = getattr(torch.version, "hip", None) is not None
        tag = "ROCm" if is_rocm else "CUDA"
        if world_size > 1:
            lines.append(f"Backend  : {tag} (DDP, {world_size} processes)")
        else:
            lines.append(f"Backend  : {tag}")
        names = {torch.cuda.get_device_name(i) for i in range(n_visible)}
        name_str = ", ".join(sorted(names))
        total_gb = sum(torch.cuda.get_device_properties(i).total_memory
                       for i in range(n_visible)) / 1e9
        lines.append(f"Devices  : {n_visible} x {name_str}  ({total_gb:.1f} GB total)")
    elif dev.type == "xpu":
        n_visible = torch.xpu.device_count()
        lines.append(f"Backend  : XPU (Intel){' (DDP, ' + str(world_size) + ' processes)' if world_size > 1 else ''}")
        names = {torch.xpu.get_device_name(i) for i in range(n_visible)}
        lines.append(f"Devices  : {n_visible} x {', '.join(sorted(names))}")
    elif dev.type == "mps":
        lines.append("Backend  : MPS (Apple Silicon)")
    else:
        lines.append(f"Backend  : CPU ({platform.processor() or platform.machine()})")
    lines.append(f"PyTorch  : {torch.__version__}")
    return lines


def _default_device() -> str:
    """Select best available device: CUDA/ROCm -> XPU -> MPS -> CPU.

    PyTorch-ROCm routes AMD GPUs through the cuda namespace, so the cuda
    probe catches both. Other torch backends (e.g. XPU for Intel) are
    detected if their namespace is present and reports an available device.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def format_config_summary(config: dict) -> List[str]:
    """Render the parsed nep.in config as printable lines.

    Each value is prefixed with ``(input)`` if it appeared in nep.in or
    ``(default)`` if ``parse_nep_in`` filled it in. Helps users verify the
    parameters their run is actually using.

    Reads ``config["_explicit"]`` (the set of keys present in nep.in before
    defaults were applied); falls back to "(input)" for everything if that
    marker is absent.
    """
    explicit = config.get("_explicit", None)
    def tag(*keys):
        # If any of the canonical keys for a grouped option is explicit
        # (e.g. user wrote `cutoff 6 4`, which sets cutoff_radial AND
        # cutoff_angular), mark the whole row as (input).
        if explicit is None:
            return "(input)  "
        return "(input)  " if any(k in explicit for k in keys) else "(default)"

    lines = []
    lines.append("Model architecture")
    lines.append("------------------")
    lines.append(f"  {tag('type_names', 'num_types'):10}  types        "
                 f"{config['num_types']}  {' '.join(config['type_names'])}")
    lines.append(f"  {tag('cutoff_radial', 'cutoff_angular'):10}  cutoff       "
                 f"{config['cutoff_radial']} {config['cutoff_angular']}")
    lines.append(f"  {tag('n_max_radial', 'n_max_angular'):10}  n_max        "
                 f"{config['n_max_radial']} {config['n_max_angular']}")
    lines.append(f"  {tag('basis_size_radial', 'basis_size_angular'):10}  "
                 f"basis_size   "
                 f"{config['basis_size_radial']} {config['basis_size_angular']}")
    lm_pad = (config['l_max'] + [0] * 7)[:7]
    lines.append(f"  {tag('l_max'):10}  l_max        "
                 f"{' '.join(str(x) for x in lm_pad)}\n"
                 f"  {'':10}  (L_3b, q_222, q_1111, q_112, q_123, q_233, q_134)")
    lines.append(f"  {tag('neuron'):10}  neuron       {config['neuron']}")
    if config.get("zbl") is not None:
        zbl_extra = ""
        if config.get("typewise_cutoff_zbl_factor") is not None:
            zbl_extra = f"  typewise factor {config['typewise_cutoff_zbl_factor']}"
        lines.append(f"  {tag('zbl'):10}  zbl          {config['zbl']}{zbl_extra}")

    lines.append("")
    lines.append("Training schedule (Stage 1)")
    lines.append("---------------------------")
    lines.append(f"  {tag('num_epochs'):10}  epoch        {config['num_epochs']}")
    lines.append(f"  {tag('batch_size'):10}  batch        {config['batch_size']}")
    lines.append(f"  {tag('lr'):10}  lr           {config['lr']}")
    lines.append(f"  {tag('stop_lr'):10}  stop_lr      {config['stop_lr']}")
    lines.append(f"  {tag('lr_scheduler'):10}  lr_scheduler {config['lr_scheduler']}")
    # `scheduler_patience` serves both modes: for plateau it is the number
    # of non-improving epochs before LR reduction; for step it is the LR
    # step interval (StepLR's `step_size`).
    patience_label = ("step_size" if config['lr_scheduler'] == 'step'
                      else "patience")
    lines.append(f"  {tag('scheduler_patience'):10}  {patience_label:11}  "
                 f"{config['scheduler_patience']}")
    lines.append(f"  {tag('scheduler_factor'):10}  factor       "
                 f"{config['scheduler_factor']}")
    lines.append(f"  {tag('max_grad_norm'):10}  max_grad     {config['max_grad_norm']}")
    lines.append(f"  {tag('lambda_e'):10}  lambda_e     {config['lambda_e']}")
    lines.append(f"  {tag('lambda_f'):10}  lambda_f     {config['lambda_f']}")
    lines.append(f"  {tag('lambda_v'):10}  lambda_v     {config['lambda_v']}")
    if config.get("lambda_1", 0.0) or config.get("lambda_2", 0.0):
        lines.append(f"  {tag('lambda_1'):10}  lambda_1     {config['lambda_1']}")
        lines.append(f"  {tag('lambda_2'):10}  lambda_2     {config['lambda_2']}")

    if config.get("stage2"):
        lines.append("")
        lines.append("Training schedule (Stage 2)")
        lines.append("---------------------------")
        ss = config.get("start_stage2")
        ss_str = f"{ss}" if ss is not None else f"auto (0.5 * {config['num_epochs']})"
        lines.append(f"  {tag('start_stage2'):10}  start_stage2 {ss_str}")
        lines.append(f"  {tag('stage2_lr'):10}  stage2_lr    {config['stage2_lr']}")
        # The stage 2 scheduler can override stage 1's patience / factor.
        # Fall back to the stage 1 values when not explicitly set.
        s2p = config.get('stage2_scheduler_patience', config['scheduler_patience'])
        s2f = config.get('stage2_scheduler_factor',   config['scheduler_factor'])
        lines.append(f"  {tag('stage2_scheduler_patience'):10}  "
                     f"stage2_scheduler_patience {s2p}")
        lines.append(f"  {tag('stage2_scheduler_factor'):10}  "
                     f"stage2_scheduler_factor   {s2f}")
        lines.append(f"  {tag('stage2_pref_e'):10}  stage2_lambda_e {config['stage2_pref_e']}")
        lines.append(f"  {tag('stage2_pref_f'):10}  stage2_lambda_f {config['stage2_pref_f']}")
        lines.append(f"  {tag('stage2_pref_v'):10}  stage2_lambda_v {config['stage2_pref_v']}")

    return lines




# ---------------------------------------------------------------------------
# GPU data store — all data pre-loaded to device
# ---------------------------------------------------------------------------

def _basis_chunk_size(device, dtype, basis_size_angular, num_lm, l_max_3b,
                      min_chunk=1 << 16, max_chunk=1 << 23):
    """Pairs-per-chunk for the cached-basis precompute in ``GPUDataStore``.

    The basis is built in chunks so the transient working set is one chunk
    instead of the whole shard — this lowers the construction-time GPU memory
    peak (which otherwise sits well above the steady training footprint and
    needlessly caps how big a shard each rank can hold). Results are unchanged:
    the Chebyshev/angular bases are per-pair elementwise, so chunking is
    bit-identical to the one-shot path.

    The chunk is sized so one chunk's transient stays within a small fixed
    budget (``TARGET`` below, ~512 MB) — the whole point is to keep the peak
    just above the steady footprint, not to go as fast as possible. On CUDA the
    budget is additionally clamped to a fraction of the memory still *free*
    after the persistent basis buffers are allocated (queried via
    ``mem_get_info``), so a memory-tight card shrinks the chunk further rather
    than OOM-ing. On CPU only the fixed budget applies. Bigger chunks make
    ``__init__`` faster; they never affect training speed (training reads the
    same split-view layout either way).
    """
    elem = 8 if dtype == torch.float64 else 4
    # Generous per-pair transient estimate: Chebyshev scratch (~14 vectors),
    # angular z/Re/Im powers + the blm list and its torch.stack copy
    # (~2*num_lm + 4*(l+1)), plus margin. Over-estimating only shrinks the
    # chunk, which is safe.
    per_pair = elem * (2 * (basis_size_angular + 1) + 2 * num_lm
                       + 4 * (l_max_3b + 1) + 40)
    TARGET = 512 * 1024 * 1024  # ~512 MB transient budget per chunk
    budget = TARGET
    if device.type == "cuda":
        try:
            free, _ = torch.cuda.mem_get_info(device)
            # Never let the transient eat more than half of what's free, so a
            # tight card shrinks the chunk instead of OOM-ing at build time.
            budget = min(TARGET, int(free * 0.5))
        except Exception:
            budget = min(TARGET, 256 * 1024 * 1024)
    chunk = budget // max(per_pair, 1)
    return max(min_chunk, min(max_chunk, chunk))


class GPUDataStore:
    """Pre-loads all structure data to GPU for zero-copy batch collation.

    When ``config`` is given, also caches Chebyshev basis functions and
    angular basis on GPU so training never recomputes them.
    """

    def __init__(self, structures: List[Dict], device: torch.device,
                 dtype: torch.dtype, config: dict = None):
        self.device = device
        self.dtype = dtype
        self.n = len(structures)
        self.has_cached_basis = config is not None

        n_rad = np.array([len(s["pair_i_rad"]) for s in structures], dtype=np.int64)
        n_ang = np.array([len(s["pair_i_ang"]) for s in structures], dtype=np.int64)
        self.natoms = [int(s["natoms"]) for s in structures]

        # preprocess_structures already returns arrays with the right dtype
        # (int64 for indices, float32 for rij). Skip the defensive astype —
        # it was creating an extra copy of every per-frame array.
        at_cat   = np.concatenate([s["atom_types"]  for s in structures])
        pi_r_cat = np.concatenate([s["pair_i_rad"]  for s in structures])
        pj_r_cat = np.concatenate([s["pair_j_rad"]  for s in structures])
        rij_r_cat = np.concatenate([s["rij_rad"]    for s in structures])
        pi_a_cat = np.concatenate([s["pair_i_ang"]  for s in structures])
        pj_a_cat = np.concatenate([s["pair_j_ang"]  for s in structures])
        rij_a_cat = np.concatenate([s["rij_ang"]    for s in structures])

        at_all   = torch.from_numpy(at_cat).to(device=device, non_blocking=True)

        pi_r_all = torch.from_numpy(pi_r_cat).to(device=device, non_blocking=True)
        pj_r_all = torch.from_numpy(pj_r_cat).to(device=device, non_blocking=True)
        rij_r_all = torch.from_numpy(rij_r_cat).to(device=device, dtype=dtype,
                                                    non_blocking=True)
        pi_a_all = torch.from_numpy(pi_a_cat).to(device=device, non_blocking=True)
        pj_a_all = torch.from_numpy(pj_a_cat).to(device=device, non_blocking=True)
        rij_a_all = torch.from_numpy(rij_a_cat).to(device=device, dtype=dtype,
                                                    non_blocking=True)

        self.energy = [float(s["energy"]) if "energy" in s else 0.0
                       for s in structures]
        self.has_energy_flag = ["energy" in s for s in structures]
        self.has_forces_flag = ["forces" in s for s in structures]
        self.has_virial_flag = ["virial" in s for s in structures]

        f_parts = []
        for s in structures:
            if "forces" in s:
                f_parts.append(np.asarray(s["forces"]).reshape(-1, 3))
            else:
                f_parts.append(np.zeros((s["natoms"], 3), dtype=np.float32))
        f_cat = np.concatenate(f_parts).astype(np.float32 if dtype == torch.float32
                                               else np.float64, copy=False)
        f_all = torch.from_numpy(f_cat).to(device=device, dtype=dtype,
                                            non_blocking=True)

        v_parts = []
        for s in structures:
            if "virial" in s:
                v = np.asarray(s["virial"]).reshape(-1)
                if v.shape[0] == 6:
                    v9 = np.array([v[0], v[3], v[5],
                                   v[3], v[1], v[4],
                                   v[5], v[4], v[2]])
                    v_parts.append(v9)
                else:
                    v_parts.append(v[:9])
            else:
                v_parts.append(np.zeros(9))
        v_cat = np.stack(v_parts).astype(np.float32 if dtype == torch.float32
                                         else np.float64, copy=False)
        v_all = torch.from_numpy(v_cat).to(device=device, dtype=dtype,
                                            non_blocking=True)

        # Per-frame cell volume (A**3) — needed for stress RMSE. Same order as
        # frames, so a batch slice follows the same indexing as .energy etc.
        vol_cat = np.asarray([s.get("volume", 0.0) for s in structures],
                             dtype=np.float32 if dtype == torch.float32
                             else np.float64)
        self.volumes = torch.from_numpy(vol_cat).to(device=device, dtype=dtype,
                                                     non_blocking=True)

        if config is not None:
            # Build the cached basis in pair-chunks, writing into preallocated
            # buffers. The transient working set is then one chunk, not the
            # whole shard — this caps the construction-time GPU memory peak so
            # it stays close to the steady training footprint. Chebyshev and
            # angular bases are per-pair elementwise, so this is bit-identical
            # to computing them in one shot; only __init__ does more work, the
            # training step (which reads the split views below) is unchanged.
            rc_r = config["cutoff_radial"]
            rc_a = config["cutoff_angular"]
            bs_r = config["basis_size_radial"]
            bs_a = config["basis_size_angular"]
            l3 = config["l_max"][0]
            num_lm = sum(2 * ll + 1 for ll in range(1, l3 + 1)) if l3 >= 1 else 0

            P_r = rij_r_all.shape[0]
            fk_r_all = torch.empty(P_r, bs_r + 1, dtype=dtype, device=device)
            fkp_r_all = torch.empty(P_r, bs_r + 1, dtype=dtype, device=device)
            d12inv_r_all = torch.empty(P_r, dtype=dtype, device=device)

            P_a = rij_a_all.shape[0]
            fk_a_all = torch.empty(P_a, bs_a + 1, dtype=dtype, device=device)
            fkp_a_all = torch.empty(P_a, bs_a + 1, dtype=dtype, device=device)
            d12inv_a_all = torch.empty(P_a, dtype=dtype, device=device)
            blm_all = torch.empty(P_a, num_lm, dtype=dtype, device=device)

            # Chunk sized against memory still free *after* the buffers above.
            chunk = _basis_chunk_size(device, dtype, bs_a, num_lm, l3)

            for st in range(0, P_r, chunk):
                en = min(st + chunk, P_r)
                dr = torch.norm(rij_r_all[st:en], dim=-1)
                fk, fkp = ops.chebyshev_basis_and_deriv(dr, rc_r, bs_r)
                fk_r_all[st:en] = fk
                fkp_r_all[st:en] = fkp
                d12inv_r_all[st:en] = 1.0 / dr

            for st in range(0, P_a, chunk):
                en = min(st + chunk, P_a)
                rij = rij_a_all[st:en]
                da = torch.norm(rij, dim=-1)
                fk, fkp = ops.chebyshev_basis_and_deriv(da, rc_a, bs_a)
                fk_a_all[st:en] = fk
                fkp_a_all[st:en] = fkp
                dinv = 1.0 / da
                d12inv_a_all[st:en] = dinv
                if num_lm > 0:
                    blm_all[st:en] = ops.angular_basis(
                        rij[:, 0] * dinv, rij[:, 1] * dinv,
                        rij[:, 2] * dinv, l3)

        nr_list = n_rad.tolist()
        na_list = n_ang.tolist()
        nat_list = [int(x) for x in self.natoms]

        self.atom_types = list(torch.split(at_all, nat_list))
        self.pi_rad = list(torch.split(pi_r_all, nr_list))
        self.pj_rad = list(torch.split(pj_r_all, nr_list))
        self.rij_rad = list(torch.split(rij_r_all, nr_list))
        self.pi_ang = list(torch.split(pi_a_all, na_list))
        self.pj_ang = list(torch.split(pj_a_all, na_list))
        self.rij_ang = list(torch.split(rij_a_all, na_list))
        self.forces = list(torch.split(f_all, nat_list))
        self.virial = list(torch.unbind(v_all, dim=0))

        if config is not None:
            self.fk_rad = list(torch.split(fk_r_all, nr_list))
            self.fkp_rad = list(torch.split(fkp_r_all, nr_list))
            self.d12inv_rad = list(torch.split(d12inv_r_all, nr_list))
            self.fk_ang = list(torch.split(fk_a_all, na_list))
            self.fkp_ang = list(torch.split(fkp_a_all, na_list))
            self.d12inv_ang = list(torch.split(d12inv_a_all, na_list))
            self.blm = list(torch.split(blm_all, na_list))

        self.n_energy = sum(self.has_energy_flag)
        self.n_forces = sum(self.has_forces_flag)
        self.n_virial = sum(self.has_virial_flag)
        self.has_forces = self.n_forces > 0
        self.has_virial = self.n_virial > 0

    def collate(self, indices: List[int]) -> Dict:
        """Fast GPU-side batch collation. No CPU->GPU transfer."""
        offsets = [0]
        for i in indices:
            offsets.append(offsets[-1] + self.natoms[i])
        N_total = offsets[-1]
        B = len(indices)

        at_list = [self.atom_types[i] for i in indices]
        atom_types = torch.cat(at_list)

        struct_idx = torch.cat([
            torch.full((self.natoms[i],), k, dtype=torch.long,
                       device=self.device)
            for k, i in enumerate(indices)
        ])

        pi_r = torch.cat([self.pi_rad[i] + offsets[k]
                          for k, i in enumerate(indices)])
        pj_r = torch.cat([self.pj_rad[i] + offsets[k]
                          for k, i in enumerate(indices)])
        rij_r = torch.cat([self.rij_rad[i] for i in indices])
        pi_a = torch.cat([self.pi_ang[i] + offsets[k]
                          for k, i in enumerate(indices)])
        pj_a = torch.cat([self.pj_ang[i] + offsets[k]
                          for k, i in enumerate(indices)])
        rij_a = torch.cat([self.rij_ang[i] for i in indices])

        energy = torch.tensor([self.energy[i] for i in indices],
                              dtype=self.dtype, device=self.device)
        natoms = torch.tensor([self.natoms[i] for i in indices],
                              dtype=self.dtype, device=self.device)

        volumes = self.volumes[torch.as_tensor(indices, device=self.device,
                                                dtype=torch.long)]

        batch = {
            "N": N_total, "num_structures": B,
            "atom_types": atom_types, "struct_idx": struct_idx,
            "pair_i_rad": pi_r, "pair_j_rad": pj_r, "rij_rad": rij_r,
            "pair_i_ang": pi_a, "pair_j_ang": pj_a, "rij_ang": rij_a,
            "energy": energy, "natoms": natoms, "volumes": volumes,
        }

        batch["energy_mask"] = torch.tensor(
            [self.has_energy_flag[i] for i in indices],
            dtype=torch.bool, device=self.device)

        batch["forces"] = torch.cat([self.forces[i] for i in indices])
        force_flags = [self.has_forces_flag[i] for i in indices]
        batch["force_mask"] = torch.cat([
            torch.full((self.natoms[indices[k]],), force_flags[k],
                       dtype=torch.bool, device=self.device)
            for k in range(B)
        ])

        batch["virial"] = torch.stack([self.virial[i] for i in indices])
        batch["virial_mask"] = torch.tensor(
            [self.has_virial_flag[i] for i in indices],
            dtype=torch.bool, device=self.device)

        if self.has_cached_basis:
            batch["fk_rad"] = torch.cat([self.fk_rad[i] for i in indices])
            batch["fkp_rad"] = torch.cat([self.fkp_rad[i] for i in indices])
            batch["d12inv_rad"] = torch.cat([self.d12inv_rad[i] for i in indices])
            batch["fk_ang"] = torch.cat([self.fk_ang[i] for i in indices])
            batch["fkp_ang"] = torch.cat([self.fkp_ang[i] for i in indices])
            batch["d12inv_ang"] = torch.cat([self.d12inv_ang[i] for i in indices])
            batch["blm"] = torch.cat([self.blm[i] for i in indices])

        return batch


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess_one_frame(args):
    """Worker: build neighbor lists for a single frame. Picklable for mp.Pool."""
    frame, rc_rad, rc_ang, max_rc, type_names, dtype = args
    positions = frame["positions"].astype(dtype)
    cell = frame["cell"].astype(dtype)
    atom_types = np.array([type_names.index(s) for s in frame["species"]],
                          dtype=np.int64)
    pair_i, pair_j, rij = build_neighbor_list_np(positions, cell, max_rc)
    dij = np.linalg.norm(rij, axis=1)
    rad_mask = dij < rc_rad
    ang_mask = dij < rc_ang
    s = {
        "natoms": frame["natoms"],
        "atom_types": atom_types,
        "volume": float(abs(np.linalg.det(cell))),   # A**3, used for stress RMSE
        "pair_i_rad": pair_i[rad_mask], "pair_j_rad": pair_j[rad_mask],
        "rij_rad": rij[rad_mask].astype(dtype),
        "pair_i_ang": pair_i[ang_mask], "pair_j_ang": pair_j[ang_mask],
        "rij_ang": rij[ang_mask].astype(dtype),
    }
    if "energy" in frame:
        s["energy"] = frame["energy"]
    if "forces" in frame:
        s["forces"] = frame["forces"].astype(dtype)
    if "virial" in frame:
        s["virial"] = frame["virial"].astype(dtype)
    return s


def preprocess_structures(frames, config, dtype=np.float32, n_workers=None):
    """Build neighbor lists for all frames, parallelized across CPU cores.

    Per-frame work is embarrassingly parallel. Worker behavior:

    - **Worker count** defaults to ``cpu_count() // LOCAL_WORLD_SIZE`` so
      DDP ranks on the same node don't oversubscribe CPU cores. Override
      with ``TORCHNEP_PREPROC_WORKERS`` env var.

    - **Start method** defaults to ``fork`` (fastest). Workers do pure numpy
      (neighbor list + type lookup) and never touch CUDA, so fork is safe
      even after the parent has initialized CUDA — same pattern used by
      PyTorch's ``DataLoader(num_workers>0)``. Override with
      ``TORCHNEP_MP_START_METHOD=spawn`` on systems where fork-after-CUDA
      behaves pathologically (rare).

    Disable pooling entirely with ``n_workers=1`` (useful for debugging).
    """
    rc_rad = config["cutoff_radial"]
    rc_ang = config["cutoff_angular"]
    type_names = config["type_names"]
    max_rc = max(rc_rad, rc_ang)

    if n_workers is None:
        cpu_total = os.cpu_count() or 1
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        n_workers = max(1, cpu_total // local_world)
        n_workers = int(os.environ.get("TORCHNEP_PREPROC_WORKERS", n_workers))

    if n_workers <= 1 or len(frames) < 64:
        return [_preprocess_one_frame((f, rc_rad, rc_ang, max_rc, type_names, dtype))
                for f in frames]

    import multiprocessing as mp
    method = os.environ.get("TORCHNEP_MP_START_METHOD", "fork")
    try:
        ctx = mp.get_context(method)
    except ValueError:
        ctx = mp.get_context("spawn")

    args = [(f, rc_rad, rc_ang, max_rc, type_names, dtype) for f in frames]
    chunksize = max(1, len(frames) // (n_workers * 4))
    with ctx.Pool(n_workers) as pool:
        return pool.map(_preprocess_one_frame, args, chunksize=chunksize)


def compute_max_neighbors(structures):
    """Return (max_NN_radial, max_NN_angular) over all structures."""
    max_rad = max_ang = 0
    for s in structures:
        n = s["natoms"]
        if len(s["pair_i_rad"]) > 0:
            counts = np.bincount(s["pair_i_rad"], minlength=n)
            max_rad = max(max_rad, int(counts.max()))
        if len(s["pair_i_ang"]) > 0:
            counts = np.bincount(s["pair_i_ang"], minlength=n)
            max_ang = max(max_ang, int(counts.max()))
    return max_rad, max_ang


# The 6 unique components of the symmetric virial, picked from the row-major
# 3x3 (length-9) layout in GPUMD order: xx, yy, zz, xy, yz, zx. The virial loss
# must use these 6 — averaging over all 9 would weight the (symmetric)
# off-diagonal pairs twice relative to the diagonal. Matches GPUMD's
# 6-component virial RMSE (lambda_shear=1 by default).
_VIRIAL_6 = [0, 4, 8, 1, 5, 6]


@torch.no_grad()
def compute_q_scaler(model, data_store, batch_size=1000, backend="loop",
                     gpumd_init=False):
    """Compute descriptor min/max across training set.

    Uses the cached-basis path (reusing data_store's precomputed Chebyshev +
    angular basis) — orders of magnitude faster than recomputing from rij.

    ``batch_size`` is a q-scaler-only knob; independent from the training
    batch size because q-scaler has no backward and can tolerate much bigger
    batches (default 1000). Set smaller if GPU memory is tight.
    ``backend`` should match the training backend so the type-pair contraction
    order of operations (and hence the floating-point accumulation) is the
    same as what training will see.

    ``gpumd_init`` selects how the descriptor coefficients are set for this
    pass. False (default): the model's actual (init) coefficients — q_scaler is
    self-consistent with the init. True: all coefficients forced to 1.0,
    exactly matching GPUMD's generation-0 q_scaler (``initial_para``).
    """
    model.eval()
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    q_min = torch.full((model.dim,), float("inf"), dtype=dtype, device=dev)
    q_max = torch.full((model.dim,), float("-inf"), dtype=dtype, device=dev)

    if gpumd_init:
        c2 = torch.ones_like(model.c_param_2)
        c3 = (torch.ones_like(model.c_param_3)
              if model.c_param_3 is not None else None)
    else:
        c2, c3 = model.c_param_2, model.c_param_3

    for start in range(0, data_store.n, batch_size):
        end = min(start + batch_size, data_store.n)
        batch = data_store.collate(list(range(start, end)))
        q = ops.compute_descriptors_cached(
            batch["fk_rad"], batch["fk_ang"], batch["blm"],
            batch["pair_i_rad"], batch["pair_j_rad"],
            batch["pair_i_ang"], batch["pair_j_ang"],
            batch["atom_types"], batch["N"],
            c2, c3,
            model.n_max_radial, model.n_max_angular,
            model.l_max_3b,
            model.has_q_222, model.has_q_1111, model.has_q_112,
            model.num_lm, model._c3b, model._c4b, model._c5b,
            model._c4b2,
            dtype, dev,
            backend=backend,
            has_q_123=model.has_q_123, has_q_233=model.has_q_233,
            has_q_134=model.has_q_134,
        )
        q_min = torch.min(q_min, q.min(0).values)
        q_max = torch.max(q_max, q.max(0).values)

    model.train()
    return q_min, q_max


@torch.no_grad()
def recompute_b1_shift(raw_model, data_store, batch_size, backend):
    """Set the energy offset ``b1`` to its analytical optimum (GPUMD-style).

    ``b1`` is not a gradient-trained parameter — its gradient is tiny (the
    mean per-atom energy error) so Adam moves it slowly, which on some datasets
    leaves the predicted energies globally shifted. Instead we solve the 1-D
    least-squares offset in closed form each epoch: the optimal shift is the
    (current) mean per-atom residual, and since the predicted energy already
    contains ``b1`` it is an additive correction

        b1 <- b1 + mean_over_energy_structs(E_pred/Na - E_ref/Na).

    This mirrors GPUMD, which recomputes the energy shift every generation so
    the energy loss is offset-free and the overall level cannot drift.
    Returns the new b1 value (float).
    """
    was_training = raw_model.training
    raw_model.eval()
    dev = raw_model.b1.device
    dtype = raw_model.b1.dtype
    num = torch.zeros((), dtype=dtype, device=dev)
    den = 0
    for start in range(0, data_store.n, batch_size):
        end = min(start + batch_size, data_store.n)
        batch = data_store.collate(list(range(start, end)))
        e_mask = batch["energy_mask"]
        if not bool(e_mask.any()):
            continue
        res = raw_model.compute_properties_cached(
            batch, need_forces=False, need_virial=False, backend=backend)
        e_pa_pred = res["Etot"] / batch["natoms"]
        e_pa_ref = batch["energy"] / batch["natoms"]
        diff = (e_pa_pred - e_pa_ref)[e_mask]
        num = num + diff.sum()
        den += int(e_mask.sum().item())
    if den > 0:
        raw_model.b1.add_(num / den)
    if was_training:
        raw_model.train()
    return float(raw_model.b1.item())


# ---------------------------------------------------------------------------
# LR scheduler helpers
# ---------------------------------------------------------------------------

def _make_lr_scheduler(optimizer, mode, factor, patience, min_lr):
    """Build the LR scheduler — "plateau" (default) or "step".

    "plateau" -> ReduceLROnPlateau(factor, patience, min_lr=min_lr).
    "step"    -> StepLR(step_size=patience, gamma=factor); min_lr enforced
                manually after each step() via _scheduler_step (StepLR has
                no min_lr).

    ``patience`` is shared between the two modes: for plateau it is the
    number of non-improving epochs before LR is reduced; for step it is the
    interval between scheduled reductions.
    """
    if mode == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=patience, gamma=factor)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=factor,
        patience=patience, min_lr=min_lr)


def _scheduler_step(scheduler, avg_loss, mode, optimizer, min_lr):
    """Advance the scheduler; for 'step' mode, clamp LR at min_lr manually."""
    if mode == "step":
        scheduler.step()
        for pg in optimizer.param_groups:
            if pg["lr"] < min_lr:
                pg["lr"] = min_lr
    else:
        scheduler.step(avg_loss)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(path, model, optimizer, scheduler, epoch, best_loss,
                     loss_weights=None, in_stage2=False, swa_model=None,
                     best_true_loss=None):
    """Write a full training checkpoint.

    ``scheduler`` must be the ACTIVE one (stage2_scheduler while in stage 2,
    lr_scheduler otherwise); ``in_stage2`` records which it was so the load
    side can restore the state into the right object.
    """
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    m = m.module if hasattr(m, "module") else m
    state = {
        "epoch": epoch,
        "best_loss": best_loss,
        "model_state": m.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "in_stage2": in_stage2,
    }
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    if loss_weights is not None:
        state["loss_weights"] = loss_weights
    if swa_model is not None:
        state["swa_state"] = swa_model.state_dict()
    if best_true_loss is not None:
        state["best_true_loss"] = best_true_loss
    torch.save(state, path)


def _load_checkpoint(path, model, optimizer, lr_scheduler, stage2_scheduler,
                     swa_model, device):
    """Load a checkpoint so the run continues exactly where it stopped.

    The optimizer state carries the lr of the checkpoint moment — nep.in's
    lr is never re-applied on resume. The scheduler state is restored into
    the scheduler that was active when the checkpoint was written
    (``in_stage2`` tag; pre-tag checkpoints fall back to the stage-1
    scheduler, matching the old behaviour). SWA state is restored when both
    sides have it.

    Returns a dict with epoch / best_loss / best_true_loss / loss_weights /
    in_stage2.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    m = m.module if hasattr(m, "module") else m
    model_state = ckpt["model_state"]
    # Checkpoints written by older DDP runs saved the shim's state_dict,
    # whose keys carry a uniform "model." prefix — strip it so they load
    # into a plain NEPModel (new checkpoints always store plain keys).
    if model_state and all(k.startswith("model.") for k in model_state):
        model_state = {k[len("model."):]: v for k, v in model_state.items()}
    m.load_state_dict(model_state)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    in_stage2 = ckpt.get("in_stage2", False)
    target = (stage2_scheduler if (in_stage2 and stage2_scheduler is not None)
              else lr_scheduler)
    if target is not None and "scheduler_state" in ckpt:
        # Scheduler class may have changed between runs (user switched
        # lr_scheduler mode). Tolerate that by skipping the state load —
        # scheduler just starts fresh this run.
        try:
            target.load_state_dict(ckpt["scheduler_state"])
        except (KeyError, ValueError, TypeError):
            pass
    if swa_model is not None and ckpt.get("swa_state") is not None:
        try:
            swa_model.load_state_dict(ckpt["swa_state"])
        except (KeyError, ValueError, RuntimeError):
            pass
    return {
        "epoch": ckpt["epoch"],
        "best_loss": ckpt["best_loss"],
        "best_true_loss": ckpt.get("best_true_loss", float("inf")),
        "loss_weights": ckpt.get("loss_weights"),
        "in_stage2": in_stage2,
    }


def _trim_loss_log(path, last_epoch):
    """Drop loss.out rows past ``last_epoch``.

    A run killed between a checkpoint and the next one has logged epochs the
    resumed run will retrain; without trimming, those epochs would appear
    twice in the file.
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        lines = f.readlines()
    kept = []
    for ln in lines:
        try:
            ep = int(ln.split()[0])
        except (ValueError, IndexError):
            kept.append(ln)  # header
            continue
        if ep <= last_epoch:
            kept.append(ln)
    if len(kept) != len(lines):
        with open(path, "w") as f:
            f.writelines(kept)


def _accumulate_true_loss_sums(data_store, batch_size, raw_model,
                               compute_props, compute_props_cached,
                               use_autograd_forces, backend,
                               has_forces, has_virial, dtype, dev):
    """Frozen-weight squared-error sums over the LOCAL data_store.

    Mirrors the training-epoch accumulation exactly (same masks, same
    per-sample units), but forward-only on one fixed set of weights.
    Returns (sum_le, sum_lf, sum_lv, n_e, n_f, n_v, sum_e_resid) so callers
    can finish the per-sample averaging themselves — the DDP path all-reduces
    these numbers across ranks first, which makes the aggregated loss EXACTLY
    the full-dataset value (a sum of per-shard sums), identical to the
    single-GPU result. ``sum_e_resid`` (Σ signed per-atom energy residual)
    lets the caller solve the exact optimal energy offset b1 for these frozen
    weights in this same pass: δ = sum_e_resid / n_e, and the offset-corrected
    energy MSE is sum_le/n_e − δ².
    """
    sum_le = sum_lf = sum_lv = 0.0
    sum_e_resid = 0.0      # Σ signed per-atom energy residual (for exact b1)
    n_e = n_f = n_v = 0

    was_training = raw_model.training
    raw_model.eval()  # autograd path: .training drives create_graph
    # The analytical path needs no graph at all; the autograd path still
    # needs grad-enabled rij for the force evaluation, so no_grad only
    # wraps the cached (analytical) compute.
    ctx = contextlib.nullcontext() if use_autograd_forces else torch.no_grad()
    try:
        with ctx:
            for start in range(0, data_store.n, batch_size):
                idx = list(range(start, min(start + batch_size,
                                            data_store.n)))
                batch = data_store.collate(idx)
                if use_autograd_forces:
                    result = compute_props(
                        batch["rij_rad"], batch["rij_ang"],
                        batch["pair_i_rad"], batch["pair_j_rad"],
                        batch["pair_i_ang"], batch["pair_j_ang"],
                        batch["atom_types"], batch["N"],
                        batch["struct_idx"], batch["num_structures"],
                        need_forces=has_forces, need_virial=has_virial,
                        backend=backend)
                else:
                    result = compute_props_cached(
                        batch, need_forces=has_forces,
                        need_virial=has_virial, backend=backend)

                e_mask = batch["energy_mask"]
                if e_mask.any():
                    e_pa_pred = result["Etot"] / batch["natoms"]
                    e_pa_ref = batch["energy"] / batch["natoms"]
                    diff_e = e_pa_pred[e_mask] - e_pa_ref[e_mask]
                    sum_le += (diff_e ** 2).sum().item()
                    sum_e_resid += diff_e.sum().item()
                    n_e += int(e_mask.sum().item())

                if has_forces:
                    f_mask = batch["force_mask"]
                    if f_mask.any():
                        f_diff = result["forces"][f_mask] - batch["forces"][f_mask]
                        sum_lf += (f_diff ** 2).mean(dim=1).sum().item()
                        n_f += int(f_mask.sum().item())

                if (has_virial and "virial" in result
                        and batch["virial"].shape[1] == 9):
                    v_mask = batch["virial_mask"]
                    if v_mask.any():
                        v_atom = result["virial"]
                        v_sys = torch.zeros(batch["num_structures"], 9,
                                            dtype=dtype, device=dev)
                        si = batch["struct_idx"].unsqueeze(-1).expand_as(v_atom)
                        v_sys.scatter_add_(0, si, v_atom)
                        na = batch["natoms"][v_mask].unsqueeze(-1)
                        # 6 unique components only (see _VIRIAL_6).
                        v_pred6 = v_sys[:, _VIRIAL_6]
                        v_ref6 = batch["virial"][:, _VIRIAL_6]
                        v_diff = (v_pred6[v_mask] - v_ref6[v_mask]) / na
                        sum_lv += (v_diff ** 2).mean(dim=1).sum().item()
                        n_v += int(v_mask.sum().item())
    finally:
        if was_training:
            raw_model.train()

    return sum_le, sum_lf, sum_lv, n_e, n_f, n_v, sum_e_resid


def _evaluate_true_loss(data_store, batch_size, raw_model,
                        compute_props, compute_props_cached,
                        use_autograd_forces, backend,
                        pref_e, pref_f, pref_v, dtype, dev):
    """Weighted loss + RMSEs of the CURRENT (frozen) weights, full dataset.

    Single-GPU wrapper around ``_accumulate_true_loss_sums`` — used to
    decide whether a candidate epoch really is the best model (see the
    best-save block in the epoch loop).

    Side effect: sets the exact optimal energy offset ``b1`` for these frozen
    weights (δ = sum_e_resid / n_e) so the evaluated loss and the saved model
    agree. This is what keeps nep_best ≤ nep_final: every candidate (the final
    epoch always among them) is judged AND saved with its own exact b1.

    Returns (true_loss, rmse_e, rmse_f, rmse_v) — energy terms offset-corrected.
    """
    has_forces = data_store.has_forces and pref_f > 0
    has_virial = data_store.has_virial and pref_v > 0
    sum_le, sum_lf, sum_lv, n_e, n_f, n_v, sum_e_resid = \
        _accumulate_true_loss_sums(
            data_store, batch_size, raw_model,
            compute_props, compute_props_cached,
            use_autograd_forces, backend, has_forces, has_virial, dtype, dev)
    # Exact optimal b1 for these frozen weights, solved from this same pass.
    delta = sum_e_resid / n_e if n_e > 0 else 0.0
    if n_e > 0:
        with torch.no_grad():
            raw_model.b1.add_(delta)
    # Offset-corrected energy MSE = Var(residual) = E[r²] − δ². Clamp tiny
    # negatives from float round-off.
    mse_e = max(0.0, sum_le / max(n_e, 1) - delta * delta)
    mse_f = sum_lf / max(n_f, 1)
    mse_v = sum_lv / max(n_v, 1)
    true_loss = pref_e * mse_e + pref_f * mse_f + pref_v * mse_v
    return true_loss, np.sqrt(mse_e), np.sqrt(mse_f), np.sqrt(mse_v)


def _quiet_compile_logs():
    """Silence torch.compile's graph-break WARNING spam.

    The analytical compute has intentional data-dependent control flow (per-type
    loops with boolean masks -> aten.nonzero, ZBL .item()) that Dynamo cannot
    trace, so it logs a graph break for each — harmless (it just falls back to
    eager for those ops), but very noisy, especially under DDP (every rank,
    every recompile). Raise the dynamo/inductor loggers to ERROR so genuine
    compile failures still surface. To see the graph breaks again, set
    logging.getLogger("torch._dynamo").setLevel(logging.WARNING).
    """
    import logging
    for name in ("torch._dynamo", "torch._inductor",
                 "torch.fx.experimental.symbolic_shapes"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _clean_warning_format():
    """Show warnings as a one-line ``Category: message`` — drop the
    ``file:line:`` prefix and the source-line echo. The default formatter prints
    the triggering line of code under the message (e.g. ``model =
    NEPModel(...)`` below the redundant-q_1111 notice), which is just noise.
    """
    import warnings
    warnings.formatwarning = (
        lambda message, category, filename, lineno, line=None:
        f"{category.__name__}: {message}\n")


def _maybe_enable_tf32(dev, dtype, log):
    """TF32 tensor-core matmul — OFF by default.

    NEP's matmuls are small (the NN layer and the descriptor contraction) and
    the workload is dominated by gather/scatter/elementwise ops, so TF32 gives
    no measurable speedup here while slightly reducing float32 matmul precision.
    Opt in with ``TORCHNEP_TF32=1`` (may help on A100/H100). Either way we
    silence Inductor's "consider set_float32_matmul_precision('high')" nag. Only
    applies to float32 CUDA; float64 is never touched.
    """
    if dev.type == "cuda" and dtype == torch.float32:
        import warnings
        warnings.filterwarnings("ignore", message=".*TensorFloat32.*")
        if os.environ.get("TORCHNEP_TF32", "0") == "1":
            torch.set_float32_matmul_precision("high")
            log("  TF32 matmul: enabled (TORCHNEP_TF32=1)")


def _compile_check(dev):
    """Decide whether torch.compile can be used; never raises.

    Returns (ok: bool, msg: str). The CUDA backend (TorchInductor) lowers to
    Triton kernels, so Triton must be importable — this is the usual failure
    mode on a cluster compute node. On CPU, Inductor uses the C++ backend and
    needs no Triton. ``msg`` is a short reason when ``ok`` is False, so callers
    can log a warning and fall back to eager instead of crashing mid-training.
    """
    if not hasattr(torch, "compile"):
        return False, "torch.compile is unavailable in this PyTorch build"
    if dev.type == "cuda":
        import importlib.util
        if importlib.util.find_spec("triton") is None:
            return False, ("Triton not found (the CUDA TorchInductor backend "
                           "needs it) — install triton, or set "
                           "use_compile=False to silence this")
    return True, ""


# ---------------------------------------------------------------------------
# Unified training entry point (single GPU or torchrun DDP)
# ---------------------------------------------------------------------------

def train_nep(
    config_file: str,
    data_file: str,
    output_dir: str = ".",
    device: str = None,
    precision: str = "float32",
    backend: str = "auto",
    use_autograd_forces: bool = False,
    use_swa: bool = False,
    use_compile: bool = False,
    print_interval: int = 10,
    restart: bool = True,
    checkpoint_interval: int = 100,
    prediction_interval: int = 20,
    finetune_from: str = None,
    resume_from: str = None,
    recompute_q_scaler: bool = False,
    slim_types: bool = False,
    energy_key: str = "energy",
    use_gpumd_qscaler: bool = True,
):
    """Train a NEP model on a single device (GPU / CPU / MPS).

    Hyperparameters (epoch / batch / lr / lambda_e,f,v / stage2* / …) come
    from ``config_file`` only. See README for the full nep.in reference.

    Launch:  python run_train.py

    Parameters
    ----------
    config_file, data_file, output_dir : paths.
    device : "cuda" | "cpu" | "mps" — auto-detected if omitted.
    precision : "float32" (default) or "float64".
    backend : "auto" | "loop" | "bmm" — see torchnep.ops.resolve_backend.
    use_autograd_forces : True -> autograd-through-rij forces (slower, gold
        standard); False (default) -> analytical chain rule.
    use_swa : True -> maintain an averaged model during stage 2 and save it
        as ``nep_average.txt`` at the end.
    use_compile : torch.compile the analytical compute method (~1.3x faster per
        epoch after a one-time first-epoch compilation cost; needs Triton, which
        ships with the CUDA PyTorch build). Ignored on the autograd force path
        (incompatible with its double-backward). For an extra ~1.3x when memory
        allows, also pass backend="bmm" (faster under compile but higher peak
        memory — see docs/torch_compile.md).
    print_interval : log a line to screen every N epochs (all epochs still
        land in output.log).
    restart : on fresh output_dir, write new log; otherwise resume from
        checkpoint.pt if present.
    checkpoint_interval : save checkpoint.pt every N epochs.
    prediction_interval : every N epochs, run predict_from_store with the
        current nep_best weights and overwrite {energy,force,virial}_train.out
        in output_dir — lets you watch the parity-plot converge live.
        Set to 0 or a negative value to disable.
    finetune_from : path to an existing .pt or nep.txt to load weights from
        (weights only — a NEW training starts from them: epoch 1, nep.in lr,
        fresh optimizer). The q_scaler stored with the weights is kept.
    resume_from : path to a checkpoint (.pt) to CONTINUE from — restores
        model, optimizer (incl. lr), scheduler, epoch and best losses, so
        the run picks up exactly where that checkpoint left off. Use
        ``checkpoint_stage1.pt`` to redo stage 2 with edited stage2_*
        settings. Mutually exclusive with finetune_from; takes precedence
        over the automatic checkpoint.pt pickup.
    recompute_q_scaler : only with finetune_from. Default False — the
        q_scaler stored with the loaded weights is kept (it is part of the
        potential they were trained as). True recomputes it from the new
        dataset, which rescales the NN inputs: the loaded model starts off
        wrong and must re-adapt (see bench_test/restart_rework/t8).
    slim_types : drop element types not present in data_file before training.
    energy_key : name of the comment-line tag read as the reference energy
        (default ``"energy"``). Set to ``"atomization_energy"`` to train
        against atomization energies instead of totals.
    use_gpumd_qscaler : Default True — reproduce GPUMD's initialization:
        descriptor coefficients are re-initialised uniform(-1, 1) and the
        q_scaler is computed with all coefficients = 1.0 (GPUMD's generation-0
        ``initial_para``). False uses the self-consistent q_scaler (computed
        from the model's actual init coefficients). Only applies to fresh
        training (ignored under finetune_from).
    """
    _clean_warning_format()

    # ---- Device ----------------------------------------------------------
    if device is None:
        device = _default_device()
    dev = torch.device(device)
    dtype = torch.float32 if precision == "float32" else torch.float64

    # ---- Logging ---------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    _out_log_file = open(os.path.join(output_dir, "output.log"),
                         "a" if restart else "w")

    def _log(msg=""):
        # flush=True so sbatch / piped stdout doesn't block-buffer log lines
        # (default block buffering hides progress between long-running steps)
        print(msg, flush=True)
        _out_log_file.write(msg + "\n")
        _out_log_file.flush()

    # ---- Banner ----------------------------------------------------------
    total_t0 = time.time()
    _log(_BANNER.rstrip())
    _log(f"   torchnep  v{__version__}   author: {_AUTHOR}")
    _log(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _log("")
    for line in _backend_info(dev):
        _log(line)
    _log(f"Precision: {precision}")
    _maybe_enable_tf32(dev, dtype, _log)
    _log("")

    # ---- Config (all hyperparameters from nep.in) -----------------------
    orig_config = parse_nep_in(config_file)
    for line in format_config_summary(orig_config):
        _log(line)
    _log("")
    config = orig_config
    # Model regularisation coefficients
    lambda_1 = config["lambda_1"]
    lambda_2 = config["lambda_2"]
    # Training schedule + loss weights
    num_epochs         = config["num_epochs"]
    batch_size         = config["batch_size"]
    lr                 = config["lr"]
    stop_lr            = config["stop_lr"]
    scheduler_patience = config["scheduler_patience"]
    scheduler_factor   = config["scheduler_factor"]
    lr_scheduler_mode  = config["lr_scheduler"]     # "plateau" | "step"
    max_grad_norm      = config["max_grad_norm"]
    pref_e             = config["lambda_e"]
    pref_f             = config["lambda_f"]
    pref_v             = config["lambda_v"]
    # Optional stage-2 block
    stage2             = config["stage2"]
    start_stage2       = config.get("start_stage2")  # may be None -> auto 0.5*num_epochs
    stage2_lr          = config["stage2_lr"]
    stage2_pref_e      = config["stage2_pref_e"]
    stage2_pref_f      = config["stage2_pref_f"]
    stage2_pref_v      = config["stage2_pref_v"]
    # Stage 2 can have its own decay schedule (default: same as stage 1).
    # Use this when the two stages span different LR ranges — e.g. stage 1
    # 1e-2 -> 1e-3 (factor 0.794, 10 decays) vs stage 2 1e-3 -> 1e-5
    # (factor 0.631, 10 decays).
    stage2_scheduler_patience = config.get("stage2_scheduler_patience",
                                           scheduler_patience)
    stage2_scheduler_factor   = config.get("stage2_scheduler_factor",
                                           scheduler_factor)

    # ---- Data ------------------------------------------------------------
    _log("Data")
    _log("----")
    frames = read_xyz(data_file, energy_key=energy_key)
    _log(f"  read {len(frames)} structures from {data_file} "
         f"(energy label: {energy_key})")

    # Single-GPU: per-epoch shuffle is done at iteration time via
    # torch.randperm(n_structs). We deliberately do NOT pre-sort by natoms
    # — natoms-homogeneous minibatches are biased gradient estimates and
    # converge significantly worse than i.i.d. shuffled minibatches (and
    # torch.compile dislikes the wider spread of input shapes that sorting
    # produces).
    n_structs = len(frames)

    # slim_types: detect which element types actually appear in the data and
    # narrow config before building neighbor lists / GPUDataStore / model.
    # This makes the entire training run faster, not just the output file.
    _slim_keep = None  # None = no slimming; list = types to keep
    if slim_types:
        seen_species = set(s for f in frames for s in f["species"])
        keep = [t for t in orig_config["type_names"] if t in seen_species]
        removed = [t for t in orig_config["type_names"] if t not in keep]
        if removed:
            _slim_keep = keep
            config = dict(orig_config)
            config["type_names"] = keep
            config["num_types"] = len(keep)
            _log(f"  slim_types: {orig_config['type_names']} -> {keep} "
                 f"(removing: {removed})")
        else:
            _log("  slim_types: all types present in data, nothing to remove")

    t0 = time.time()
    np_dtype = np.float64 if precision == "float64" else np.float32
    structures = preprocess_structures(frames, config, np_dtype)
    _log(f"  built neighbor lists in {time.time() - t0:.1f}s")

    max_NN_rad, max_NN_ang = compute_max_neighbors(structures)

    t0 = time.time()
    data_store = GPUDataStore(structures, dev, dtype, config=config)
    del structures
    if dev.type == "cuda":
        torch.cuda.synchronize()
    _log(f"  loaded to {dev} in {time.time() - t0:.1f}s (cached basis)")
    _log(f"  coverage: {data_store.n_energy} E / "
         f"{data_store.n_forces} F / {data_store.n_virial} V")
    _log("")

    # ---- Model -----------------------------------------------------------
    _log("Model")
    _log("-----")
    model = NEPModel(config).to(dtype).to(dev)

    # use_gpumd_qscaler: reproduce GPUMD's init for fresh training — descriptor
    # coefficients uniform(-1, 1) (GPUMD initialises every parameter this way)
    # paired with the c=1 q_scaler computed below. Skipped under finetune_from,
    # where the loaded trained coefficients must be kept.
    if use_gpumd_qscaler and finetune_from is None:
        with torch.no_grad():
            torch.nn.init.uniform_(model.c_param_2, -1.0, 1.0)
            if model.c_param_3 is not None:
                torch.nn.init.uniform_(model.c_param_3, -1.0, 1.0)
        _log("  use_gpumd_qscaler: descriptor coeffs re-init uniform(-1,1), "
             "q_scaler will use c=1 (GPUMD-consistent)")

    # b1 (global energy offset) is determined analytically each epoch, not by
    # gradient descent (see recompute_b1_shift) — exclude it from the optimizer.
    model.b1.requires_grad_(False)
    trainable_params = [p for n, p in model.named_parameters() if n != "b1"]

    if finetune_from is not None:
        # Load pre-trained weights; skip random b1 init from mean_epa.
        # The stored q_scaler is part of the loaded model and is KEPT (see
        # the q_scaler block below) — recomputing it on the new dataset
        # would rescale the NN inputs the weights were trained against.
        ft_path = finetune_from

        def _load_weights(target, path):
            if path.endswith(".pt"):
                state = torch.load(path, map_location=dev, weights_only=False)
                if "model_state" in state:
                    state = state["model_state"]
                target.load_state_dict(state, strict=True)
            else:
                target.load_weights_from_nep_txt(path)

        if _slim_keep is not None:
            # Load full pre-trained model (orig arch), then slim to current config
            full_model = NEPModel(orig_config).to(dtype).to(dev)
            _load_weights(full_model, ft_path)
            slimmed = slim_model(full_model, config["type_names"])
            model.load_state_dict(slimmed.state_dict())
            del full_model, slimmed
            _log(f"  fine-tuning from {ft_path}  "
                 f"[{orig_config['num_types']} -> {config['num_types']} types]")
        else:
            _load_weights(model, ft_path)
            _log(f"  fine-tuning from {ft_path}")
        _log(f"  {sum(p.numel() for p in model.parameters())} parameters, "
             f"dim={model.dim}, b1={model.b1.item():.4f}")
    else:
        mean_epa = np.mean([data_store.energy[i] / data_store.natoms[i]
                            for i in range(data_store.n)
                            if data_store.has_energy_flag[i]])
        with torch.no_grad():
            model.b1.fill_(-mean_epa)
        _log(f"  {sum(p.numel() for p in model.parameters())} parameters, "
             f"dim={model.dim}, b1 init={model.b1.item():.4f}")

    # Decide whether torch.compile will actually run BEFORE resolving the
    # backend — the backend choice depends on it. Only the analytical path is
    # compiled; the autograd path's create_graph=True double backward is
    # incompatible with compile, so it stays eager.
    compile_on = False
    compile_msg = None
    if use_compile:
        ok, msg = _compile_check(dev)
        if not ok:
            compile_msg = f"  torch.compile: disabled — {msg}"
        elif use_autograd_forces:
            compile_msg = ("  torch.compile: skipped — incompatible with "
                           "autograd double-backward forces")
        else:
            compile_on = True
            compile_msg = "  torch.compile: enabled (analytical compute method)"

    # Resolve "auto" backend. ``backend`` is the eager backend for the one-shot
    # q_scaler pass (num_types-based: loop for few types, bmm for >=8).
    # ``train_backend`` is what the per-batch compute uses — bmm whenever
    # compiling (it fuses best under Inductor). An explicit backend= wins for
    # both; the two backends are numerically identical.
    from .ops import resolve_backend as _resolve_backend
    orig_backend = backend
    backend = _resolve_backend(orig_backend, num_types=model.num_types)
    train_backend = _resolve_backend(
        orig_backend, num_types=model.num_types, use_compile=compile_on)
    force_str = "autograd" if use_autograd_forces else "analytical"
    if train_backend != backend:
        _log(f"  backend: {backend} (q_scaler) / {train_backend} (training), "
             f"forces: {force_str}")
    else:
        _log(f"  backend: {backend}, forces: {force_str}")

    if finetune_from is not None and not recompute_q_scaler:
        # The q_scaler is part of the potential definition: the loaded NN
        # weights were trained against it. Keep it; recomputing from the
        # new data (with the loaded, trained c-params) would rescale every
        # descriptor component and wreck the loaded model.
        _log("  q_scaler: kept from finetune source (not recomputed)")
    else:
        if finetune_from is not None:
            _log("  q_scaler: RECOMPUTED from the new dataset "
                 "(recompute_q_scaler=True) — the loaded weights will "
                 "see rescaled descriptors and must re-adapt")
        t0 = time.time()
        q_min, q_max = compute_q_scaler(model, data_store, backend=backend,
                                        gpumd_init=use_gpumd_qscaler)
        model.set_q_scaler(q_min, q_max)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        _log(f"  q_scaler in {time.time() - t0:.1f}s")

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    # torch.compile: compile the bound compute METHOD, not the module — the
    # training loop calls compute_properties_cached directly (not forward()), so
    # compiling the module would be a no-op. Compiling the bound method shares
    # parameters with raw_model. dynamic=True avoids a recompile per batch shape.
    compute_props = raw_model.compute_properties
    compute_props_cached = raw_model.compute_properties_cached
    if compile_on:
        _quiet_compile_logs()
        compute_props_cached = torch.compile(
            raw_model.compute_properties_cached, dynamic=True)
    if compile_msg is not None:
        _log(compile_msg)

    optimizer = torch.optim.Adam(trainable_params, lr=lr,
                                 weight_decay=lambda_2, amsgrad=True)

    if stage2 and start_stage2 is None:
        start_stage2 = max(1, int(num_epochs * 0.5))

    lr_scheduler = _make_lr_scheduler(
        optimizer, lr_scheduler_mode, scheduler_factor,
        scheduler_patience, stop_lr)

    def _loss_fn(pred, ref):
        return torch.mean((pred - ref) ** 2)

    swa_model = None
    stage2_scheduler = None
    if stage2:
        stage2_scheduler = _make_lr_scheduler(
            optimizer, lr_scheduler_mode, stage2_scheduler_factor,
            stage2_scheduler_patience, stop_lr)
        if use_swa:
            swa_model = AveragedModel(raw_model)

    # Snapshot of current loss weights — saved in checkpoint so a restart can
    # detect that the user edited nep.in and reset best_loss (otherwise the
    # old-scale best_loss would keep the new run from ever saving a new best).
    cur_loss_weights = {
        "lambda_e": pref_e, "lambda_f": pref_f, "lambda_v": pref_v,
        "stage2_pref_e": stage2_pref_e, "stage2_pref_f": stage2_pref_f,
        "stage2_pref_v": stage2_pref_v,
    }

    ckpt_path = os.path.join(output_dir, "checkpoint.pt")
    start_epoch = 1
    best_loss = float("inf")
    best_true_loss = float("inf")
    stage2_lr_applied = False  # tracks whether stage2 lr/reset has fired yet
    if resume_from is not None and finetune_from is not None:
        raise ValueError("resume_from and finetune_from are mutually "
                         "exclusive: resume_from continues a run, "
                         "finetune_from starts a new one from weights")
    resume_ckpt = None
    if resume_from is not None:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"resume_from: {resume_from} not found")
        resume_ckpt = resume_from
    elif restart and os.path.exists(ckpt_path) and finetune_from is None:
        resume_ckpt = ckpt_path
    if resume_ckpt is not None:
        info = _load_checkpoint(resume_ckpt, model, optimizer,
                                lr_scheduler, stage2_scheduler,
                                swa_model, dev)
        start_epoch = info["epoch"] + 1
        best_loss = info["best_loss"]
        best_true_loss = info["best_true_loss"]
        _log(f"Resumed from {resume_ckpt}: epoch {start_epoch - 1}, "
             f"best_loss={best_loss:.4e}")
        # Resume = exact continuation: lr (and the whole optimizer state)
        # comes from the checkpoint moment. nep.in's lr only applies to
        # fresh runs / finetunes — and stage2_lr at a fresh stage-2 entry.
        _log(f"  lr taken from checkpoint: "
             f"{optimizer.param_groups[0]['lr']:.2e} "
             f"(nep.in lr is not applied on resume)")
        saved_loss_weights = info["loss_weights"]
        if saved_loss_weights is not None and saved_loss_weights != cur_loss_weights:
            _log("Loss weights changed since checkpoint was saved — "
                 "resetting best losses so the new scale can establish "
                 "a new best.")
            _log(f"  saved:   {saved_loss_weights}")
            _log(f"  current: {cur_loss_weights}")
            best_loss = float("inf")
            best_true_loss = float("inf")

    n_structs = data_store.n
    # has_forces / has_virial are recomputed per-epoch inside the loop using
    # the CURRENT stage's weights — so a stage-1 weight of 0 can still enable
    # computation in stage 2 (and vice versa). Do NOT latch them here.

    loss_log_path = os.path.join(output_dir, "loss.out")
    if start_epoch > 1:
        # Keep loss.out in 1:1 correspondence with the actual history: rows
        # past the checkpoint epoch belong to epochs the resumed run is
        # about to retrain.
        _trim_loss_log(loss_log_path, start_epoch - 1)
    write_header = (start_epoch == 1 or not os.path.exists(loss_log_path)
                    or os.path.getsize(loss_log_path) == 0)
    loss_log = open(loss_log_path, "w" if start_epoch == 1 else "a")
    if write_header:
        loss_log.write("# epoch  loss  rmse_e(eV/atom)  rmse_f(eV/A)  "
                       "rmse_v(eV/atom)  rmse_stress(GPa)\n")

    # All training hyperparameters (lr/scheduler/loss weights/stage2 ...)
    # already printed by format_config_summary above; here we just announce
    # the runtime epoch range — different from `epoch` in nep.in when
    # resuming from a checkpoint.
    stage2_tag = (f", Stage 2 from epoch {start_stage2} "
                  f"(SWA={'on' if use_swa else 'off'})") if stage2 else ""
    # In the last third of the run a candidate best is verified by a
    # frozen-weight evaluation over the full dataset before nep_best is
    # written (see the best-save block in the loop).
    true_eval_start = (2 * num_epochs) // 3 + 1
    _log("")
    _log(f"Training: epochs {start_epoch}..{num_epochs}{stage2_tag}")
    _log("=" * 72)

    def _save_best():
        raw_model.save_nep_txt(
            os.path.join(output_dir, "nep_best.txt"),
            max_NN_rad, max_NN_ang)

    train_t0 = time.time()

    try:
        for epoch in range(start_epoch, num_epochs + 1):
            t_epoch = time.time()
            model.train()

            # Per-epoch frame-level shuffle (i.i.d. minibatches). Seeded by
            # epoch so reruns are reproducible and resumed runs continue
            # along the same stream.
            g = torch.Generator()
            g.manual_seed(epoch)
            perm = torch.randperm(n_structs, generator=g).tolist()

            sum_le = sum_lf = sum_lv = 0.0
            sum_ls = 0.0                     # (eV/A**3)**2 accumulator for stress
            sum_e_structs = sum_f_atoms = sum_v_structs = 0
            sum_e_resid = 0.0                # Σ(E_pred/Na − E_ref/Na) for b1
            max_gn = 0.0

            in_stage2 = stage2 and epoch >= start_stage2
            if in_stage2:
                cur_pref_e, cur_pref_f, cur_pref_v = (
                    stage2_pref_e, stage2_pref_f, stage2_pref_v)
                # Apply stage 2 lr + reset the first time we hit stage 2 in
                # THIS run — covers both the natural transition and resuming
                # from a stage-1 checkpoint whose start_epoch has already
                # crossed into stage 2 (``epoch == start_stage2`` would miss
                # the second case, leaving the optimizer on stage-1 lr).
                if not stage2_lr_applied:
                    stage2_lr_applied = True
                    # Distinguish a NATURAL stage-1 -> stage-2 crossing in this
                    # run from RESUMING a checkpoint that was already in stage 2.
                    fresh_transition = start_epoch <= start_stage2
                    _log(f"\n{'='*72}")
                    if fresh_transition:
                        # Crossing into stage 2 now: save a FULL end-of-
                        # stage-1 checkpoint (model + optimizer + scheduler
                        # state through epoch-1), so stage 2 can be redone
                        # from this exact point via resume_from after
                        # editing the stage2_* settings in nep.in. Then
                        # apply the stage-2 lr and reset the best losses
                        # (the stage-2 loss weights put them on a
                        # different scale).
                        _save_checkpoint(
                            os.path.join(output_dir, "checkpoint_stage1.pt"),
                            model, optimizer, lr_scheduler, epoch - 1,
                            best_loss, loss_weights=cur_loss_weights,
                            in_stage2=False, swa_model=None,
                            best_true_loss=best_true_loss)
                        _log("Saved end-of-stage-1 checkpoint: "
                             "checkpoint_stage1.pt (redo stage 2 from it "
                             "via resume_from)")
                        for pg in optimizer.param_groups:
                            pg['lr'] = stage2_lr
                        best_loss = float("inf")
                        best_true_loss = float("inf")
                        _log(f"Stage 2 started at epoch {epoch}: "
                             f"E_w={cur_pref_e}, F_w={cur_pref_f}, "
                             f"V_w={cur_pref_v}, lr={stage2_lr:.2e}")
                    else:
                        # Resumed mid-stage-2: keep the checkpoint's restored
                        # (possibly decayed) lr / optimizer / scheduler /
                        # best losses — do NOT re-apply nep.in's stage2_lr
                        # (resume = exact continuation). best losses are
                        # reset only if the loss weights changed (handled
                        # at checkpoint load).
                        cur_lr = optimizer.param_groups[0]['lr']
                        _log(f"Stage 2 resumed at epoch {epoch}: "
                             f"E_w={cur_pref_e}, F_w={cur_pref_f}, "
                             f"V_w={cur_pref_v}, lr={cur_lr:.2e} "
                             f"(kept from checkpoint)")
                    _log(f"{'='*72}")
            else:
                cur_pref_e, cur_pref_f, cur_pref_v = pref_e, pref_f, pref_v

            # Per-epoch compute eligibility: a weight of 0 means "don't compute
            # this channel". This is recomputed every epoch so a stage-1 zero
            # weight doesn't block stage-2 computation (see stage transition
            # above) — and so pref_v=0 really skips virial compute/backward.
            has_forces = data_store.has_forces and cur_pref_f > 0
            has_virial = data_store.has_virial and cur_pref_v > 0

            for start in range(0, n_structs, batch_size):
                idx = perm[start:start + batch_size]
                batch = data_store.collate(idx)

                if use_autograd_forces:
                    result = compute_props(
                        batch["rij_rad"], batch["rij_ang"],
                        batch["pair_i_rad"], batch["pair_j_rad"],
                        batch["pair_i_ang"], batch["pair_j_ang"],
                        batch["atom_types"], batch["N"],
                        batch["struct_idx"], batch["num_structures"],
                        need_forces=has_forces, need_virial=has_virial,
                        backend=backend)
                else:
                    result = compute_props_cached(
                        batch, need_forces=has_forces, need_virial=has_virial,
                        backend=train_backend)

                e_pa_pred = result["Etot"] / batch["natoms"]
                e_pa_ref = batch["energy"] / batch["natoms"]
                e_mask = batch["energy_mask"]
                loss = torch.tensor(0.0, dtype=dtype, device=dev)
                # sum_l* accumulates per-batch MSE so the rmse_* columns in
                # the log are real RMSE. Optimizer sees _loss_fn (MSE) too.
                if e_mask.any():
                    diff_e = e_pa_pred[e_mask] - e_pa_ref[e_mask]
                    loss_e = _loss_fn(e_pa_pred[e_mask], e_pa_ref[e_mask])
                    loss = loss + cur_pref_e * loss_e
                    sum_le += (diff_e ** 2).mean().item() * e_mask.sum().item()
                    # Accumulate the signed residual for the analytical b1
                    # update (folded into this pass — no extra forward).
                    sum_e_resid += diff_e.sum().item()

                if has_forces:
                    f_mask = batch["force_mask"]
                    if f_mask.any():
                        f_pred = result["forces"][f_mask]
                        f_ref = batch["forces"][f_mask]
                        loss_f = _loss_fn(f_pred, f_ref)
                        loss = loss + cur_pref_f * loss_f
                        sum_lf += ((f_pred - f_ref) ** 2).mean().item() * f_mask.sum().item()

                if has_virial and "virial" in result:
                    v_mask = batch["virial_mask"]
                    if v_mask.any():
                        v_atom = result["virial"]
                        v_sys = torch.zeros(batch["num_structures"], 9,
                                            dtype=dtype, device=dev)
                        si = batch["struct_idx"].unsqueeze(-1).expand_as(v_atom)
                        v_sys.scatter_add_(0, si, v_atom)
                        v_ref = batch["virial"]
                        if v_ref.shape[1] == 9:
                            na = batch["natoms"][v_mask].unsqueeze(-1)
                            # 6 unique components only (see _VIRIAL_6).
                            v_pred_pa = v_sys[:, _VIRIAL_6][v_mask] / na
                            v_ref_pa = v_ref[:, _VIRIAL_6][v_mask] / na
                            loss_v = _loss_fn(v_pred_pa, v_ref_pa)
                            loss = loss + cur_pref_v * loss_v
                            v_diff = v_pred_pa - v_ref_pa
                            sum_lv += (v_diff ** 2).mean().item() * v_mask.sum().item()
                            # Stress RMSE (eV/A**3): convert the same diff using
                            # per-frame (natoms/volume). Sign cancels under MSE.
                            scale = (batch["natoms"][v_mask]
                                     / batch["volumes"][v_mask]).unsqueeze(-1)
                            s_diff = v_diff * scale
                            sum_ls += (s_diff ** 2).mean().item() * v_mask.sum().item()

                if lambda_1 > 0:
                    l1 = sum(p.abs().sum() for p in trainable_params)
                    loss = loss + lambda_1 * l1

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if max_grad_norm > 0:
                    gn = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm).item()
                else:
                    gn = torch.sqrt(sum(
                        p.grad.norm()**2 for p in raw_model.parameters()
                        if p.grad is not None)).item()

                if not np.isfinite(gn):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()

                if in_stage2 and swa_model is not None:
                    swa_model.update_parameters(raw_model)

                sum_e_structs += batch["energy_mask"].sum().item()
                sum_f_atoms += batch["force_mask"].sum().item()
                sum_v_structs += batch["virial_mask"].sum().item()
                max_gn = max(max_gn, gn)

            # Analytical b1 (GPUMD-style), folded into the training pass: b1
            # absorbs this epoch's mean per-atom energy residual. Updated AFTER
            # the batch loop but BEFORE the best-model eval / nep_best save, so
            # the saved weights and offset stay consistent. ``b1`` is not a
            # gradient parameter (see the optimizer exclusion above).
            if sum_e_structs > 0:
                with torch.no_grad():
                    raw_model.b1.add_(sum_e_resid / sum_e_structs)

            # Per-sample (not per-batch) averaging so avg_loss is self-
            # consistent with rmse_{e,f,v}: avg_loss == \Sigma pref_X * MSE_X
            # where each MSE_X aggregates over all samples in the epoch.
            from .constants import EV_PER_A3_TO_GPa
            mse_e = sum_le / max(sum_e_structs, 1)
            mse_f = sum_lf / max(sum_f_atoms, 1) if sum_lf > 0 else 0.0
            mse_v = sum_lv / max(sum_v_structs, 1) if sum_lv > 0 else 0.0
            mse_s = sum_ls / max(sum_v_structs, 1) if sum_ls > 0 else 0.0
            avg_loss = (cur_pref_e * mse_e + cur_pref_f * mse_f
                        + cur_pref_v * mse_v)
            # Output units: eV/atom (E, V), eV/A (F), GPa (stress).
            rmse_e = np.sqrt(mse_e)
            rmse_f = np.sqrt(mse_f)
            rmse_v = np.sqrt(mse_v)
            rmse_s_gpa = np.sqrt(mse_s) * EV_PER_A3_TO_GPa
            dt = time.time() - t_epoch

            if in_stage2 and stage2_scheduler is not None:
                _scheduler_step(stage2_scheduler, avg_loss,
                                lr_scheduler_mode, optimizer, stop_lr)
            elif not in_stage2:
                _scheduler_step(lr_scheduler, avg_loss,
                                lr_scheduler_mode, optimizer, stop_lr)

            loss_log.write(f"{epoch} {avg_loss:.6e} {rmse_e:.6f} "
                           f"{rmse_f:.6f} {rmse_v:.6f} {rmse_s_gpa:.4f}\n")
            loss_log.flush()

            stage_str = "[S2] " if in_stage2 else ""
            cur_lr = optimizer.param_groups[0]['lr']
            v_str = (f" | V {rmse_v:.5f} eV/atom | S {rmse_s_gpa:.3f} GPa"
                     if has_virial else "")
            line = (f"{stage_str}Epoch {epoch:4d} | loss {avg_loss:.4e} | "
                    f"E {rmse_e:.5f} eV/atom | F {rmse_f:.5f} eV/A"
                    f"{v_str} | gnorm {max_gn:.1f} | "
                    f"lr {cur_lr:.2e} | {dt:.1f}s")
            if epoch % print_interval == 0 or epoch == 1:
                _log(line)
            else:
                _out_log_file.write(line + "\n")
                _out_log_file.flush()

            # Best-model bookkeeping. avg_loss averages over weights that
            # keep moving within the epoch, so it is a noisy proxy for the
            # end-of-epoch weights actually saved. Early in the run that is
            # fine (the model improves much faster than the noise). In the
            # last third — where best really gets decided — a new avg_loss
            # minimum only NOMINATES the weights: they become nep_best only
            # if a frozen-weight pass over the full dataset (same masks and
            # averaging as the screen numbers) beats the best true loss so
            # far. The final epoch is always evaluated, so nep_best can
            # never end up worse than nep_final.
            new_min = avg_loss < best_loss
            if new_min:
                best_loss = avg_loss
            if epoch < true_eval_start:
                if new_min:
                    _save_best()
            elif new_min or epoch == num_epochs:
                t_loss, _te, _tf, _tv = _evaluate_true_loss(
                    data_store, batch_size, raw_model,
                    compute_props, compute_props_cached,
                    use_autograd_forces, train_backend,
                    cur_pref_e, cur_pref_f, cur_pref_v, dtype, dev)
                if t_loss < best_true_loss:
                    best_true_loss = t_loss
                    _save_best()

            if epoch % checkpoint_interval == 0 or epoch == num_epochs:
                _save_checkpoint(
                    ckpt_path, model, optimizer,
                    stage2_scheduler if in_stage2 else lr_scheduler,
                    epoch, best_loss, loss_weights=cur_loss_weights,
                    in_stage2=in_stage2,
                    swa_model=swa_model if in_stage2 else None,
                    best_true_loss=best_true_loss)

            # Interim predict — overwrites the same output files, so users can
            # refresh the parity plot live. Runs on the CURRENT-epoch weights
            # (not nep_best) so the predict loss matches what was just logged
            # for this epoch: it should fall between this epoch's and the next
            # epoch's displayed loss (current weights = end-of-epoch, whereas
            # the screen average covers weights that were still improving
            # throughout the epoch).
            # Skip on the final epoch — the end-of-training predict (below)
            # immediately overwrites these files with the final-epoch result.
            if (prediction_interval > 0
                    and epoch % prediction_interval == 0
                    and epoch != num_epochs):
                # Silent interim predict — reuses data_store's preprocessed
                # neighbor lists + basis (no xyz re-read, no recompute).
                predict_from_store(raw_model, data_store, output_dir,
                                   batch_size=batch_size, backend=backend,
                                   verbose=False)
    finally:
        if loss_log is not None:
            loss_log.close()

    # Final-epoch model (what the current weights actually are). b1 is already
    # exact for these weights: the final epoch (epoch == num_epochs) always
    # runs the best-model eval, which solves and sets the optimal b1. Do NOT
    # recompute it here — that would give nep_final a different (lower-loss) b1
    # than the value the best-model comparison used, which could make nep_final
    # beat the saved nep_best.
    raw_model.save_nep_txt(os.path.join(output_dir, "nep_final.txt"),
                           max_NN_rad, max_NN_ang)
    # SWA-averaged model (only when user opted in and stage 2 ran).
    if swa_model is not None:
        swa_state = swa_model.module.state_dict()
        # Keep a copy of the final-epoch weights so we can restore them
        # after saving SWA — the end-of-training predict below must see
        # final weights, not SWA-averaged ones.
        final_state = {k: v.clone() for k, v in raw_model.state_dict().items()}
        raw_model.load_state_dict(swa_state)
        raw_model.save_nep_txt(os.path.join(output_dir, "nep_average.txt"),
                               max_NN_rad, max_NN_ang)
        raw_model.load_state_dict(final_state)
        _log("SWA model saved to nep_average.txt")

    train_time = time.time() - train_t0
    h, rem = divmod(train_time, 3600)
    m_, s = divmod(rem, 60)
    _log(f"\nDone. Best loss: {best_loss:.6e}")
    _log(f"Training time: {int(h):02d}:{int(m_):02d}:{s:04.1f}")

    # End-of-training predict reuses the in-memory data_store (no xyz re-read)
    # and the final-epoch weights in raw_model (no model-file round-trip).
    _log("\nRunning prediction on training set (final-epoch model)...")
    pred_t0 = time.time()
    predict_from_store(raw_model, data_store, output_dir,
                       batch_size=batch_size, backend=backend,
                       verbose=False)
    _log(f"  Prediction time: {time.time() - pred_t0:.1f}s")

    total_time = time.time() - total_t0
    h, rem = divmod(total_time, 3600)
    m_, s = divmod(rem, 60)
    _log(f"\nTotal time (data + train + predict): "
         f"{int(h):02d}:{int(m_):02d}:{s:04.1f}")
    _log(f"Output: {output_dir}/")
    _out_log_file.close()
