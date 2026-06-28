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
Data-sharded distributed NEP training.

Each rank loads only 1/N of the training structures, so data-store memory
scales as 1/N instead of being replicated.  Gradients are all-reduced by
DDP; q_scaler statistics and per-epoch metrics are all-reduced explicitly.

Usage (typical multi-GPU launch via torchrun):

    torchrun --nproc_per_node=N run_train.py

where run_train.py calls ``train_nep_sharded(...)`` instead of ``train_nep``.

Backend is chosen automatically: NCCL when every rank has its own CUDA
device, otherwise gloo (covers CPU-only testing and the GPU-sharing case).
Single-rank (N=1) runs but offers nothing over ``train_nep`` in that case.
"""

import os
import atexit
import time
import warnings
import torch
import torch.distributed as dist
import numpy as np
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel

import torch.nn as nn
from .model import NEPModel
from .data import read_xyz, parse_nep_in
from . import ops
from . import __version__
from .predict import predict_from_store_sharded
from .model import slim_model


_PG_ATEXIT_DONE = False


def _register_pg_atexit():
    """Register a one-time atexit hook to destroy the process group on exit.

    We deliberately do NOT destroy the group at the end of train_nep_sharded so
    the same process can run several trainings back to back; cleanup happens
    once, at interpreter shutdown, which also avoids the "process group has NOT
    been destroyed" warning a bare leak would print.
    """
    global _PG_ATEXIT_DONE
    if _PG_ATEXIT_DONE:
        return
    _PG_ATEXIT_DONE = True

    def _cleanup():
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    atexit.register(_cleanup)


class _NEPDDPShim(nn.Module):
    """Thin wrapper whose forward calls compute_properties{_cached}.

    Why this exists: DDP's gradient all-reduce is armed inside
    ``DistributedDataParallel.forward`` (it calls
    ``reducer.prepare_for_backward`` on the output). Calling
    ``self.module.compute_properties_cached(...)`` directly — even though the
    parameters live in the module — bypasses DDP's forward and therefore the
    reducer. Each rank then keeps its local gradient; weights drift per-rank
    and only rank 0's overfit copy is saved. Putting the compute call inside
    this shim's ``forward`` puts it on the DDP path.
    """

    def __init__(self, model: NEPModel, use_compile: bool = False):
        super().__init__()
        self.model = model
        # torch.compile is applied to the bound analytical compute method, not
        # to ``model`` — the heavy compute lives in compute_properties_cached,
        # NOT in NEPModel.forward(), so compiling the module would be a no-op.
        # The compiled call stays INSIDE this shim's forward, so it remains on
        # DDP's forward path and the reducer still arms backward all-reduce; the
        # compiled region does not span the DDP boundary. The autograd path is
        # left eager (its create_graph=True double backward is incompatible with
        # torch.compile's donated-buffer optimisation). See train.py for the
        # single-device counterpart.
        if use_compile and hasattr(torch, "compile"):
            self._compute_cached = torch.compile(
                model.compute_properties_cached, dynamic=True)
        else:
            self._compute_cached = model.compute_properties_cached

    def forward(self, batch, use_autograd_forces: bool,
                need_forces: bool, need_virial: bool, backend: str):
        if use_autograd_forces:
            return self.model.compute_properties(
                batch["rij_rad"], batch["rij_ang"],
                batch["pair_i_rad"], batch["pair_j_rad"],
                batch["pair_i_ang"], batch["pair_j_ang"],
                batch["atom_types"], batch["N"],
                batch["struct_idx"], batch["num_structures"],
                need_forces=need_forces, need_virial=need_virial,
                backend=backend)
        return self._compute_cached(
            batch, need_forces=need_forces, need_virial=need_virial,
            backend=backend)

from .train import (
    _BANNER, _AUTHOR,
    _backend_info, GPUDataStore,
    format_config_summary,
    preprocess_structures,
    _save_checkpoint, _load_checkpoint,
    _trim_loss_log, _accumulate_true_loss_sums,
    _make_lr_scheduler, _scheduler_step,
    _compile_check, _quiet_compile_logs, _maybe_enable_tf32,
    _clean_warning_format, _VIRIAL_6,
)


# ---------------------------------------------------------------------------
# Sharded q_scaler
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_q_scaler_sharded(model, data_store, batch_size=1000,
                               backend="loop"):
    """Compute descriptor min/max over the local shard, then all-reduce.

    Uses cached basis from data_store (no Chebyshev recompute). batch_size
    here is q-scaler-only (independent from training batch), default 1000.
    """
    model.eval()
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    q_min = torch.full((model.dim,), float("inf"), dtype=dtype, device=dev)
    q_max = torch.full((model.dim,), float("-inf"), dtype=dtype, device=dev)

    for start in range(0, data_store.n, batch_size):
        end = min(start + batch_size, data_store.n)
        batch = data_store.collate(list(range(start, end)))
        q = ops.compute_descriptors_cached(
            batch["fk_rad"], batch["fk_ang"], batch["blm"],
            batch["pair_i_rad"], batch["pair_j_rad"],
            batch["pair_i_ang"], batch["pair_j_ang"],
            batch["atom_types"], batch["N"],
            model.c_param_2, model.c_param_3,
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

    dist.all_reduce(q_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(q_max, op=dist.ReduceOp.MAX)

    model.train()
    return q_min, q_max


# ---------------------------------------------------------------------------
# Sharded training entry point
# ---------------------------------------------------------------------------

def train_nep_sharded(
    config_file: str,
    data_file: str,
    output_dir: str = ".",
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
):
    """Data-sharded NEP training.  Launch via torchrun (or any launcher that
    sets RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT).

    All hyperparameters (epoch / batch / lr / lambda_e,f,v / stage2* / ...)
    come from ``config_file`` — see README for the nep.in reference.

    Each rank loads structures[rank::world_size] only, so data-store memory
    scales as 1/world_size.  Gradients are all-reduced by DDP; q_scaler and
    epoch metrics are all-reduced explicitly.

        torchrun --nproc_per_node=N run_train.py

    Parameters mirror ``train_nep`` (same restart semantics: resume is an
    exact continuation — lr/optimizer/scheduler/SWA/best gates from the
    checkpoint, nep.in's lr never re-applied on resume; ``resume_from``
    continues from a specific checkpoint such as ``checkpoint_stage1.pt``;
    ``finetune_from`` keeps the source model's q_scaler unless
    ``recompute_q_scaler=True``). Runtime-exclusive differences are the DDP
    launch, CUDA/gloo backend auto-select, and distributed aggregation of
    q_scaler / epoch metrics / the frozen-weight best evaluation.
    """
    _clean_warning_format()

    # ---- Distributed init ------------------------------------------------
    # Wrap local_rank around the number of visible GPUs — lets several
    # processes share one GPU (useful for locally simulating multi-rank DDP).
    # NCCL refuses to share a GPU across ranks, so fall back to gloo (slower
    # but correct) when world_size > available GPUs.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cuda_available = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if cuda_available else 0
    if cuda_available:
        gpu_id = local_rank % max(1, n_gpus)
        torch.cuda.set_device(gpu_id)
        dev = torch.device(f"cuda:{gpu_id}")
    else:
        gpu_id = None
        dev = torch.device("cpu")
    if not dist.is_initialized():
        world_size_env = int(os.environ.get("WORLD_SIZE", 1))
        ddp_backend = "nccl" if cuda_available and world_size_env <= n_gpus else "gloo"
        # Pass device_id so NCCL can bind the rank to its CUDA device
        # deterministically (silences "Guessing device ID based on global
        # rank" and the collective-context warnings, and prevents hangs on
        # heterogeneous rank->GPU mappings). gloo ignores device_id.
        dist.init_process_group(backend=ddp_backend,
                                device_id=dev if cuda_available else None)
        # Tear the group down once, at process exit — NOT at the end of this
        # function. That lets a single script call train_nep_sharded() more than
        # once (e.g. compile vs no-compile back to back): re-initialising a
        # torchrun process group after destroying it desynchronises the ranks
        # and fails NCCL eager-connect ("remote process exited / Connection
        # refused"). See the end-of-function barrier.
        _register_pg_atexit()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    is_main = rank == 0
    dtype = torch.float32 if precision == "float32" else torch.float64

    # Only rank 0 logs. All app output already routes through _log (is_main-
    # guarded), but Python warnings (e.g. NEPModel's q_1111 redundancy notice)
    # fire on every rank — silence them on non-main ranks so the console shows
    # one copy, not world_size copies.
    if not is_main:
        warnings.filterwarnings("ignore")

    # ---- Logging (rank 0 only) -------------------------------------------
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
    dist.barrier()

    _out_log_file = (open(os.path.join(output_dir, "output.log"),
                          "a" if restart else "w")
                     if is_main else None)

    def _log(msg=""):
        if not is_main:
            return
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
    for line in _backend_info(dev, world_size):
        _log(line)
    _log(f"Precision: {precision}")
    _maybe_enable_tf32(dev, dtype, _log)   # sets the flag on every rank
    _log(f"Mode     : data-sharded DDP ({world_size} ranks, "
         f"each holds 1/{world_size} of structures)")
    _log("")

    # ---- Config (all hyperparameters from nep.in) -----------------------
    orig_config = parse_nep_in(config_file)
    if is_main:
        for line in format_config_summary(orig_config):
            _log(line)
        _log("")
    config = orig_config
    lambda_1 = config["lambda_1"]
    lambda_2 = config["lambda_2"]
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
    stage2             = config["stage2"]
    start_stage2       = config.get("start_stage2")
    stage2_lr          = config["stage2_lr"]
    stage2_pref_e      = config["stage2_pref_e"]
    stage2_pref_f      = config["stage2_pref_f"]
    stage2_pref_v      = config["stage2_pref_v"]

    # ---- Data: each rank loads 1/world_size of structures ----------------
    _log("Data")
    _log("----")
    frames = read_xyz(data_file, energy_key=energy_key)
    n_total = len(frames)
    _log(f"  read {n_total} structures from {data_file} "
         f"(energy label: {energy_key})")

    # slim_types: all ranks agree on which types to keep (deterministic scan)
    _slim_keep = None
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

    # Random sharding: shuffle globally with a fixed seed (identical on
    # every rank -> all ranks agree on the partition), then give rank r the
    # r-th equal slice. To keep step counts identical across ranks (DDP
    # collectives require lock-step iteration) AND keep every frame in the
    # training/predict set, we pad the perm with duplicates from the head
    # until it divides evenly. At most W-1 frames are seen twice in an
    # epoch — global counts in the loss normalisation already account for
    # this (the duplicated frames contribute their squared error twice in
    # both numerator and denominator), and the predict scatter writes
    # identical values into the duplicated slots, so output is loss-fair
    # and complete.
    shuffle_g = torch.Generator()
    shuffle_g.manual_seed(0)
    global_perm = torch.randperm(n_total, generator=shuffle_g).tolist()
    n_local = (n_total + world_size - 1) // world_size  # ceil
    pad = n_local * world_size - n_total
    if pad:
        global_perm = global_perm + global_perm[:pad]
    local_global_idx = global_perm[rank * n_local : (rank + 1) * n_local]
    local_frames = [frames[i] for i in local_global_idx]
    pad_note = (f", {pad} frame(s) duplicated for even split"
                if pad else "")
    _log(f"  sharded across {world_size} ranks: "
         f"{n_local} frames per rank{pad_note}")

    t0 = time.time()
    np_dtype = np.float64 if precision == "float64" else np.float32
    structures = preprocess_structures(local_frames, config, np_dtype)
    _log(f"  built neighbor lists (local shard) in {time.time() - t0:.1f}s")

    # max_NN: local max then all-reduce so rank-0 has the global value
    def _compute_max_neighbors_local(structures):
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

    local_max_rad, local_max_ang = _compute_max_neighbors_local(structures)
    nn_t = torch.tensor([local_max_rad, local_max_ang], dtype=torch.long,
                        device=dev)
    dist.all_reduce(nn_t, op=dist.ReduceOp.MAX)
    max_NN_rad, max_NN_ang = int(nn_t[0].item()), int(nn_t[1].item())

    t0 = time.time()
    data_store = GPUDataStore(structures, dev, dtype, config=config)
    del structures
    if cuda_available:
        torch.cuda.synchronize()
    _log(f"  loaded to {dev} in {time.time() - t0:.1f}s (cached basis)")

    # Aggregate data counts across all ranks for the banner
    counts_t = torch.tensor(
        [data_store.n, data_store.n_energy,
         data_store.n_forces, data_store.n_virial],
        dtype=torch.long, device=dev)
    dist.all_reduce(counts_t)
    g_n, g_ne, g_nf, g_nv = counts_t.tolist()
    _log(f"  coverage (global): {g_ne} E / {g_nf} F / {g_nv} V")
    _log("")

    # ---- Model -----------------------------------------------------------
    _log("Model")
    _log("-----")
    model = NEPModel(config).to(dtype).to(dev)

    if finetune_from is not None:
        def _load_weights(target, path):
            if path.endswith(".pt"):
                state = torch.load(path, map_location=dev, weights_only=False)
                if "model_state" in state:
                    state = state["model_state"]
                target.load_state_dict(state, strict=True)
            else:
                target.load_weights_from_nep_txt(path)

        if _slim_keep is not None:
            full_model = NEPModel(orig_config).to(dtype).to(dev)
            _load_weights(full_model, finetune_from)
            slimmed = slim_model(full_model, config["type_names"])
            model.load_state_dict(slimmed.state_dict())
            del full_model, slimmed
            _log(f"  fine-tuning from {finetune_from}  "
                 f"[{orig_config['num_types']} -> {config['num_types']} types]")
        else:
            _load_weights(model, finetune_from)
            _log(f"  fine-tuning from {finetune_from}")
        _log(f"  {sum(p.numel() for p in model.parameters())} parameters, "
             f"dim={model.dim}, b1={model.b1.item():.4f}")
    else:
        # mean_epa: weighted average across all ranks
        local_epa_sum = sum(
            data_store.energy[i] / data_store.natoms[i]
            for i in range(data_store.n) if data_store.has_energy_flag[i]
        )
        local_n_e = float(data_store.n_energy)
        epa_t = torch.tensor([local_epa_sum, local_n_e], device=dev, dtype=torch.float64)
        dist.all_reduce(epa_t)
        mean_epa = float(epa_t[0] / epa_t[1]) if epa_t[1] > 0 else 0.0
        with torch.no_grad():
            model.b1.fill_(-mean_epa)
        _log(f"  {sum(p.numel() for p in model.parameters())} parameters, "
             f"dim={model.dim}, b1 init={model.b1.item():.4f}")

    # has_forces / has_virial: OR across ranks
    flags_t = torch.tensor(
        [int(data_store.has_forces), int(data_store.has_virial)],
        dtype=torch.long, device=dev)
    dist.all_reduce(flags_t, op=dist.ReduceOp.MAX)
    global_has_forces = bool(flags_t[0].item())
    global_has_virial = bool(flags_t[1].item())

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

    # ``backend`` is the eager backend for the one-shot q_scaler pass
    # (num_types-based); ``train_backend`` is what the per-batch compute uses —
    # bmm whenever compiling. Explicit backend= wins for both.
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
        # weights were trained against it. Keep it (same as train_nep).
        _log("  q_scaler: kept from finetune source (not recomputed)")
    else:
        if finetune_from is not None:
            _log("  q_scaler: RECOMPUTED from the new dataset "
                 "(recompute_q_scaler=True) — the loaded weights will "
                 "see rescaled descriptors and must re-adapt")
        # q_scaler: local shard -> all_reduce
        t_qs = time.time()
        q_min, q_max = _compute_q_scaler_sharded(model, data_store,
                                                 backend=backend)
        model.set_q_scaler(q_min, q_max)
        if cuda_available:
            torch.cuda.synchronize()
        _log(f"  q_scaler in {time.time() - t_qs:.1f}s "
             f"(all-reduce across shards)")

    # Wrap in a shim whose forward calls compute_properties{_cached} — this
    # keeps the force/virial compute on DDP's forward path so the reducer can
    # arm backward all-reduce. See _NEPDDPShim docstring. torch.compile is
    # applied INSIDE the shim (to the bound analytical method), not to the
    # module — compiling the module only wraps NEPModel.forward(), which the
    # training loop never calls, so it would be a no-op (the bug this fixes).
    if compile_msg is not None:
        _log(compile_msg)
    if compile_on:
        _quiet_compile_logs()
    shim = _NEPDDPShim(model, use_compile=compile_on)
    # All per-type nets are always touched in compute_properties_cached (dummy
    # pass for types absent in a given batch) so DDP sees every parameter in
    # every step — no need for find_unused_parameters, and no implicit grad
    # dilution for rare types.
    model = DDP(shim,
                device_ids=[gpu_id] if cuda_available else None,
                find_unused_parameters=False)
    # raw_model: unwrap DDP -> shim -> NEPModel (compile now lives on a bound
    # method inside the shim, so model.model is the plain NEPModel).
    _shim = model.module
    raw_model = _shim.model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
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
        # Stage 2 may have its own decay schedule (same fallback as train.py)
        stage2_scheduler = _make_lr_scheduler(
            optimizer, lr_scheduler_mode,
            config.get("stage2_scheduler_factor", scheduler_factor),
            config.get("stage2_scheduler_patience", scheduler_patience),
            stop_lr)
        if use_swa and is_main:
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
        # Every rank loads the same file; model weights go into raw_model
        # (the plain NEPModel — keeps checkpoints interchangeable with
        # single-GPU runs). swa_model is None on non-main ranks, so SWA
        # state is restored on rank 0 only — consistent with rank-0-only
        # SWA updates.
        info = _load_checkpoint(resume_ckpt, raw_model, optimizer,
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

    n_local = data_store.n
    # has_forces / has_virial are recomputed per-epoch inside the loop using
    # the CURRENT stage's weights — so a stage-1 weight of 0 can still enable
    # computation in stage 2 (and vice versa). Do NOT latch them here.

    loss_log = None
    if is_main:
        loss_log_path = os.path.join(output_dir, "loss.out")
        if start_epoch > 1:
            # Keep loss.out in 1:1 correspondence with the actual history.
            _trim_loss_log(loss_log_path, start_epoch - 1)
        write_header = (start_epoch == 1
                        or not os.path.exists(loss_log_path)
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
    _log("")
    _log(f"Training: epochs {start_epoch}..{num_epochs}{stage2_tag}")
    _log("=" * 72)

    # In the last third of the run a candidate best is verified by a
    # frozen-weight evaluation (each rank evaluates its shard, sums are
    # all-reduced) before nep_best is written — same criterion as train_nep.
    true_eval_start = (2 * num_epochs) // 3 + 1

    def _save_best():
        raw_model.save_nep_txt(
            os.path.join(output_dir, "nep_best.txt"),
            max_NN_rad, max_NN_ang)

    train_t0 = time.time()

    try:
        for epoch in range(start_epoch, num_epochs + 1):
            t_epoch = time.time()
            model.train()

            # Per-epoch local frame shuffle. Each rank independently
            # permutes its own n_local frames (different seeds across ranks
            # so each rank sees a fresh order). Step count = ceil(n_local /
            # batch_size) is identical across ranks because n_local is —
            # so DDP collectives stay in lock-step.
            g = torch.Generator()
            g.manual_seed(epoch * world_size + rank)
            perm = torch.randperm(n_local, generator=g).tolist()

            sum_le = sum_lf = sum_lv = sum_ls = 0.0  # sum_ls is in (eV/A**3)**2
            sum_e_structs = sum_f_atoms = sum_v_structs = 0
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
                        # Crossing into stage 2 now: rank 0 writes a FULL
                        # end-of-stage-1 checkpoint (model + optimizer +
                        # scheduler through epoch-1) so stage 2 can be
                        # redone from this exact point via resume_from.
                        # Weights/optimizer are identical across ranks
                        # (DDP), so rank 0's copy is THE state.
                        if is_main:
                            _save_checkpoint(
                                os.path.join(output_dir,
                                             "checkpoint_stage1.pt"),
                                raw_model, optimizer, lr_scheduler,
                                epoch - 1, best_loss,
                                loss_weights=cur_loss_weights,
                                in_stage2=False, swa_model=None,
                                best_true_loss=best_true_loss)
                            _log("Saved end-of-stage-1 checkpoint: "
                                 "checkpoint_stage1.pt (redo stage 2 from "
                                 "it via resume_from)")
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
                        # (resume = exact continuation).
                        cur_lr = optimizer.param_groups[0]['lr']
                        _log(f"Stage 2 resumed at epoch {epoch}: "
                             f"E_w={cur_pref_e}, F_w={cur_pref_f}, "
                             f"V_w={cur_pref_v}, lr={cur_lr:.2e} "
                             f"(kept from checkpoint)")
                    _log(f"{'='*72}")
            else:
                cur_pref_e, cur_pref_f, cur_pref_v = pref_e, pref_f, pref_v

            # Per-epoch compute eligibility: a weight of 0 means "don't compute
            # this channel". Recomputed every epoch so a stage-1 zero weight
            # doesn't block stage-2 computation — and so pref_v=0 really skips
            # virial compute/backward.
            has_forces = global_has_forces and cur_pref_f > 0
            has_virial = global_has_virial and cur_pref_v > 0

            for start in range(0, n_local, batch_size):
                idx = perm[start:start + batch_size]
                batch = data_store.collate(idx)

                # Go through DDP wrapper (not raw_model.compute_*) so the
                # reducer arms backward all-reduce for this step.
                result = model(batch, use_autograd_forces,
                               has_forces, has_virial, train_backend)

                e_pa_pred = result["Etot"] / batch["natoms"]
                e_pa_ref = batch["energy"] / batch["natoms"]
                e_mask = batch["energy_mask"]
                f_mask = batch["force_mask"] if has_forces else None
                v_mask = batch["virial_mask"] if has_virial else None

                # --- DDP-correct normalisation --------------------------
                # DDP averages gradients by world_size. If each rank used a
                # plain local .mean() loss, the implicit per-rank n_local in
                # the denominator leaks into the global gradient (only
                # equivalent to single-card global mean when all n_local are
                # equal — which is NOT the case here because frames have
                # different atom counts).
                # Fix: each rank computes SUM-of-squared-errors, and we divide
                # by the GLOBAL count (all-reduced per batch). The * world_size
                # factor cancels DDP's /world_size averaging — giving a true
                # global-mean loss regardless of how atoms are sharded.
                counts = torch.tensor([
                    float(e_mask.sum().item()),
                    float(f_mask.sum().item()) if f_mask is not None else 0.0,
                    float(v_mask.sum().item()) if v_mask is not None else 0.0,
                ], device=dev, dtype=torch.float64)
                dist.all_reduce(counts)
                n_e_g = max(counts[0].item(), 1.0)
                n_f_g = max(counts[1].item(), 1.0)
                n_v_g = max(counts[2].item(), 1.0)
                ws = float(world_size)

                loss = torch.tensor(0.0, dtype=dtype, device=dev)

                if e_mask.any():
                    diff_e = e_pa_pred[e_mask] - e_pa_ref[e_mask]
                    sum_sq_e = (diff_e ** 2).sum()
                    loss = loss + cur_pref_e * sum_sq_e * ws / n_e_g
                    sum_le += sum_sq_e.item()  # global sum-of-squared-errors

                if f_mask is not None and f_mask.any():
                    f_pred = result["forces"][f_mask]
                    f_ref = batch["forces"][f_mask]
                    sum_sq_f = ((f_pred - f_ref) ** 2).sum()
                    # 3 components per atom -> divide by (3 * n_f_g)
                    loss = loss + cur_pref_f * sum_sq_f * ws / (3.0 * n_f_g)
                    sum_lf += (sum_sq_f.item() / 3.0)

                if v_mask is not None and v_mask.any() and "virial" in result:
                    v_atom = result["virial"]
                    v_sys = torch.zeros(batch["num_structures"], 9,
                                        dtype=dtype, device=dev)
                    si = batch["struct_idx"].unsqueeze(-1).expand_as(v_atom)
                    v_sys.scatter_add_(0, si, v_atom)
                    v_ref = batch["virial"]
                    if v_ref.shape[1] == 9:
                        na = batch["natoms"][v_mask].unsqueeze(-1)
                        # 6 unique components only (see _VIRIAL_6); all 9 would
                        # weight the symmetric off-diagonals twice.
                        v_pred_pa = v_sys[:, _VIRIAL_6][v_mask] / na
                        v_ref_pa = v_ref[:, _VIRIAL_6][v_mask] / na
                        v_diff = v_pred_pa - v_ref_pa
                        sum_sq_v = (v_diff ** 2).sum()
                        # 6 components per frame -> divide by (6 * n_v_g)
                        loss = loss + cur_pref_v * sum_sq_v * ws / (6.0 * n_v_g)
                        sum_lv += (sum_sq_v.item() / 6.0)
                        # Stress (eV/A**3) = virial_total / V. Sign cancels in MSE.
                        scale = (batch["natoms"][v_mask]
                                 / batch["volumes"][v_mask]).unsqueeze(-1)
                        sum_sq_s = ((v_diff * scale) ** 2).sum()
                        sum_ls += (sum_sq_s.item() / 6.0)

                if lambda_1 > 0:
                    l1 = sum(p.abs().sum() for p in model.parameters())
                    loss = loss + lambda_1 * l1

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if max_grad_norm > 0:
                    gn = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm).item()
                else:
                    gn = torch.sqrt(sum(
                        p.grad.norm() ** 2 for p in raw_model.parameters()
                        if p.grad is not None)).item()

                if not np.isfinite(gn):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()

                if in_stage2 and swa_model is not None and is_main:
                    swa_model.update_parameters(raw_model)

                sum_e_structs += batch["energy_mask"].sum().item()
                sum_f_atoms += batch["force_mask"].sum().item()
                sum_v_structs += batch["virial_mask"].sum().item()
                max_gn = max(max_gn, gn)

            metrics = torch.tensor(
                [sum_le, sum_lf, sum_lv, sum_ls,
                 float(sum_e_structs), float(sum_f_atoms),
                 float(sum_v_structs)],
                device=dev)
            dist.all_reduce(metrics)
            gn_t = torch.tensor(max_gn, device=dev)
            dist.all_reduce(gn_t, op=dist.ReduceOp.MAX)
            (sum_le, sum_lf, sum_lv, sum_ls,
             sum_e_structs, sum_f_atoms, sum_v_structs) = metrics.tolist()
            max_gn = gn_t.item()

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
            rmse_e = np.sqrt(mse_e)                           # eV/atom
            rmse_f = np.sqrt(mse_f)                           # eV/A
            rmse_v = np.sqrt(mse_v)                           # eV/atom
            rmse_s_gpa = np.sqrt(mse_s) * EV_PER_A3_TO_GPa    # GPa
            dt = time.time() - t_epoch

            if in_stage2 and stage2_scheduler is not None:
                _scheduler_step(stage2_scheduler, avg_loss,
                                lr_scheduler_mode, optimizer, stop_lr)
            elif not in_stage2:
                _scheduler_step(lr_scheduler, avg_loss,
                                lr_scheduler_mode, optimizer, stop_lr)

            if is_main:
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

            # ---- best-model bookkeeping (COLLECTIVE — outside is_main) ----
            # avg_loss comes from the all-reduced metrics, so every rank
            # sees the same value and takes the same branch — required
            # because the frozen evaluation is itself a collective (each
            # rank evaluates its local shard, the six error sums are
            # all-reduced, and the aggregated loss is bit-identical across
            # ranks). Only rank 0 touches files. Early in the run avg_loss
            # decides directly; in the last third a new minimum only
            # nominates the weights, and the final epoch is always
            # evaluated, so nep_best can never end up worse than nep_final.
            new_min = avg_loss < best_loss
            if new_min:
                best_loss = avg_loss
            if epoch < true_eval_start:
                if new_min and is_main:
                    _save_best()
            elif new_min or epoch == num_epochs:
                local_sums = _accumulate_true_loss_sums(
                    data_store, batch_size, raw_model,
                    raw_model.compute_properties, _shim._compute_cached,
                    use_autograd_forces, train_backend,
                    has_forces, has_virial, dtype, dev)
                sums_t = torch.tensor(local_sums, device=dev,
                                      dtype=torch.float64)
                dist.all_reduce(sums_t)
                s_le, s_lf, s_lv, n_e, n_f, n_v = sums_t.tolist()
                t_loss = (cur_pref_e * s_le / max(n_e, 1.0)
                          + cur_pref_f * s_lf / max(n_f, 1.0)
                          + cur_pref_v * s_lv / max(n_v, 1.0))
                if t_loss < best_true_loss:
                    best_true_loss = t_loss
                    if is_main:
                        _save_best()

            if is_main and (epoch % checkpoint_interval == 0
                            or epoch == num_epochs):
                _save_checkpoint(
                    ckpt_path, raw_model, optimizer,
                    stage2_scheduler if in_stage2 else lr_scheduler,
                    epoch, best_loss, loss_weights=cur_loss_weights,
                    in_stage2=in_stage2,
                    swa_model=swa_model if in_stage2 else None,
                    best_true_loss=best_true_loss)

            # Interim predict — uses the CURRENT-epoch weights (not nep_best)
            # so the predict loss matches the line just logged for this epoch:
            # it should fall between this epoch's and the next epoch's
            # displayed loss. Each rank predicts its own data_store shard;
            # arrays are gathered onto rank 0, which writes the output files
            # in input-xyz order. No xyz re-read, no temp model file, no
            # neighbor-list rebuild.
            # Skip on the final epoch — the end-of-training predict runs
            # right after and would overwrite this output anyway.
            if (prediction_interval > 0
                    and epoch % prediction_interval == 0
                    and epoch != num_epochs):
                predict_from_store_sharded(
                    raw_model, data_store, local_global_idx,
                    n_total_frames=n_total,
                    output_dir=output_dir,
                    batch_size=batch_size, backend=backend,
                    verbose=False)
    finally:
        if is_main and loss_log is not None:
            loss_log.close()

    if is_main:
        raw_model.save_nep_txt(os.path.join(output_dir, "nep_final.txt"),
                               max_NN_rad, max_NN_ang)
        if swa_model is not None:
            swa_state = swa_model.module.state_dict()
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

        _log("\nRunning prediction on training set (final-epoch model)...")

    # End-of-training predict: every rank still holds its data_store shard,
    # so we reuse it — no xyz re-read, no neighbor-list rebuild, no model-
    # file round-trip. Each rank predicts its own shard; rank 0 gathers the
    # per-frame arrays via all_gather_object and writes the output files.
    pred_t0 = time.time()
    predict_from_store_sharded(
        raw_model, data_store, local_global_idx,
        n_total_frames=n_total,
        output_dir=output_dir,
        batch_size=batch_size, backend=backend,
        verbose=is_main)
    if is_main:
        _log(f"  Prediction time: {time.time() - pred_t0:.1f}s")

    # data_store is no longer needed — free it now that predict is done.
    del data_store
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    if is_main:
        total_time = time.time() - total_t0
        h, rem = divmod(total_time, 3600)
        m_, s = divmod(rem, 60)
        _log(f"\nTotal time (data + train + predict): "
             f"{int(h):02d}:{int(m_):02d}:{s:04.1f}")
        _log(f"Output: {output_dir}/")
        _out_log_file.close()

    # Resync all ranks before returning. The end-of-training prediction writes
    # are rank-0 only, so non-main ranks finish well ahead; without this barrier
    # a *second* train_nep_sharded() call in the same script would start its
    # collectives while rank 0 is still writing, desynchronising the group.
    # The group itself is kept alive (destroyed once at process exit, see
    # _register_pg_atexit) so the re-init that previously crashed never happens.
    if dist.is_initialized():
        dist.barrier()
