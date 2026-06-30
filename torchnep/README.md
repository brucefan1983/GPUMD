# TorchNEP

A pure PyTorch implementation of the [NEP4](https://gpumd.org/theory/nep.html) (Neuroevolution Potential) training framework.

## Features

- **GPUMD-compatible** — output `nep.txt` files load directly into GPUMD for MD simulation
- **Two-stage training** — Stage 1: force-focused; Stage 2: energy-focused
- **Multi-GPU training** — distributed data parallel (DDP) across on one node or multiple nodes
- **Fine-tuning** — load any `nep.txt` or `checkpoint.pt` to do fine-tuning; optionally slim the model to only the element types present in the new dataset
- **ZBL** — Universal ZBL repulsive potential with optional typewise cutoffs

---

## Installation

TorchNEP needs only `torch >= 2.0` and `numpy`, but neither is installed automatically — install the PyTorch build that matches your CUDA/CPU setup first (see the [official guide](https://pytorch.org/get-started/locally/); numpy comes with it).

Then install TorchNEP with:

```bash
pip install torchnep -U
```

or install from source code:

```bash
cd GPUMD/torchnep
pip install .
```

---

## Training data (extended-XYZ)

TorchNEP reads extended-XYZ files. The parser is strict — the rules below are
enforced, and violations raise on load.

### Comment line tags

- `Lattice="ax ay az bx by bz cx cy cz"` — **mandatory**. Nine floats in Å
  giving the three lattice vectors as rows. Every frame is treated as fully
  periodic, so `pbc=...` is ignored. For isolated clusters/molecules or a
  non-periodic direction, use a vacuum box wider than the NEP cutoff.
- `energy=<value>`— optional, eV. System energy.
- `virial="vxx vxy vxz vyx vyy vyz vzx vzy vzz"` — optional, eV. Must
  have exactly 9 components. Positive values denote compressed states,
  negative denote stretched states (GPUMD convention).
- `stress="sxx sxy sxz syx syy syz szx szy szz"` — optional, eV/Å³.
  Must have exactly 9 components. Positive = stretched, negative =
  compressed — opposite sign to virial. If both `virial` and `stress` are
  present, `virial` wins.

### Per-atom columns

The `Properties=...` schema declares column layout. TorchNEP reads only
three fields and silently ignores everything else (e.g. `Z:I:1`):

- `species:S:1` — chemical symbol (case-sensitive; must match the
  `type` list in `nep.in`).
- `pos:R:3` — Cartesian position in Å.
- `force:R:3` or `forces:R:3` — reference force in eV/Å (optional).

---

## Training Parameters

### Model architecture (GPUMD-compatible)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `type` | required | `N name1 name2 ...` — number and names of element types |
| `cutoff` | `8 4` | Radial and angular cutoff (Å) |
| `n_max` | `6 6` | Radial and angular expansion orders |
| `basis_size` | `6 6` | Chebyshev basis size per channel (radial / angular)|
| `l_max` | `4 1 0` | `L_3b q_222 q_1111 q_112 q_123 q_233 q_134` — max L of 3-body terms (1–8) plus up to six boolean flags (matching GPUMD) enabling each higher-body invariant|
| `neuron` | `30` | Neurons in the (single) hidden layer |
| `zbl` | — | ZBL outer cutoff (Å); enables short-range repulsion |
| `use_typewise_cutoff_zbl` | — | Scale ZBL cutoffs by covalent radii |

### Training hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epoch` | `600` | Total training epochs |
| `batch` | `32` | Structures per gradient step |
| `lr` | `0.01` | Initial learning rate |
| `stop_lr` | `1e-6` | Minimum learning rate (scheduler floor) |
| `lambda_e` | `0.01` | Energy loss weight |
| `lambda_f` | `1.0` | Force loss weight |
| `lambda_v` | `0.01` | Virial loss weight |
| `lambda_1` | `0.0` | L1 regularisation |
| `lambda_2` | `0.0` | L2 regularisation (weight decay) |
| `max_grad_norm` | `10.0` | Gradient clipping threshold |
| `lr_scheduler` | `plateau` | LR schedule — `plateau` (ReduceLROnPlateau) or `step` (StepLR). Stage 1 and Stage 2 share this mode |
| `scheduler_patience` | `15` | For `plateau`: epochs without improvement before LR reduction. For `step`: epoch interval between LR reductions |
| `scheduler_factor` | `0.7` | LR reduction factor — multiplied on each decay in both modes |
| `stage2` | `0` | Enable Stage 2 (`1` = on) |
| `start_stage2` | 50 % of epochs | Epoch to switch to Stage 2 |
| `stage2_lr` | `1e-3` | Stage 2 learning rate |
| `stage2_scheduler_patience` | `scheduler_patience` | Stage 2 scheduler patience (overrides Stage 1's; same semantics) |
| `stage2_scheduler_factor` | `scheduler_factor` | Stage 2 LR decay factor (overrides Stage 1's)|
| `stage2_lambda_e` | `1.0` | Stage 2 energy weight |
| `stage2_lambda_f` | `0.05` | Stage 2 force weight |
| `stage2_lambda_v` | `0.1` | Stage 2 virial weight |

### Runtime arguments

Everything that is not about hyperparameter *values* lives on the Python
function (`train_nep` / `train_nep_sharded`):

| Argument | Default | What it controls |
|---|---|---|
| `device` | auto | `"cuda"` / `"xpu"` / `"mps"` / `"cpu"`; any other stream-based PyTorch accelerator should also work if passed explicitly |
| `precision` | `"float32"` | dtype for training + store, `"float32"` or `"float64"` |
| `backend` | `"auto"` | `"loop"`, `"bmm"`, or `"auto"` |
| `use_autograd_forces` | `False` | autograd-through-rij |
| `use_swa` | `False` | maintain SWA-averaged model and save `nep_average.txt` |
| `use_compile` | `False` | `torch.compile` the analytical compute (faster epochs after a one-time compile; needs Triton; ignored on the autograd path) |
| `print_interval` | `10` | log to screen every N epochs |
| `checkpoint_interval` | `100` | save `checkpoint.pt` every N epochs |
| `prediction_interval` | `20` | every N epochs run predict with the current-epoch weights and overwrite `{energy,force,virial}_train.out` |
| `restart` | `True` | resume from `checkpoint.pt` if present |
| `finetune_from` | `None` | load weights from a `.pt` or `nep.txt` and start a NEW training from them |
| `resume_from` | `None` | path to a checkpoint to CONTINUE from (e.g. `checkpoint_stage1.pt` to redo Stage 2); takes precedence over the automatic `checkpoint.pt` pickup |
| `recompute_q_scaler` | `False` | only with `finetune_from`: recompute the descriptor scaler on the new data instead of keeping the source model's |
| `slim_types` | `False` | drop element types absent from the dataset |
| `energy_key` | `"energy"` | comment-line tag read as reference energy (e.g. `"atomization_energy"`) |
| `use_gpumd_qscaler` | `True` | use GPUMD's `q_scaler` (coeffs `c=1`); fresh training only |

---

## Output Files

| File | Contents |
|------|----------|
| `nep_best.txt` | Best model |
| `nep_final.txt`    | Model at the last epoch (used for the end-of-training predict) |
| `nep_average.txt` | SWA-averaged model (only with `use_swa=True`) |
| `checkpoint.pt`    | Full training state |
| `checkpoint_stage1.pt` | Full end-of-Stage-1 checkpoint |
| `output.log`       | Full console log |
| `loss.out`         | Per-epoch: epoch, loss, RMSE_E (eV/atom), RMSE_F (eV/Å), RMSE_V, RMSE_stress (GPa), gnorm |
| `energy_train.out` | Per-frame predicted vs reference E/atom (eV/atom) |
| `force_train.out` | Per-atom predicted vs reference Fx Fy Fz (eV/Å) |
| `virial_train.out` | Per-frame predicted vs reference virial xx yy zz xy yz zx (eV/atom) |
| `stress_train.out` | Per-frame predicted vs reference stress (GPa) |

---

## Launch training

### Single GPU / CPU / MPS — `train_nep`

```python
# run_train.py
from torchnep import train_nep
train_nep("nep.in", "train.xyz", output_dir="output")
```

```bash
python run_train.py
```

### Multi-GPU, single node — `train_nep_sharded`

Each rank loads only `1/N` of the structures, so total GPU memory for the data store scales as `1/N`.

```python
# run_train.py
from torchnep import train_nep_sharded
train_nep_sharded("nep.in", "train.xyz", output_dir="output")
```

```bash
torchrun --standalone --nproc_per_node=4 run_train.py    # 4 GPUs on this node
```

### Multi-GPU, multi-node (SLURM) — `train_nep_sharded`

For M nodes × N GPUs each, the key SLURM directives are:

```bash
#SBATCH --nodes=2                  # M nodes
#SBATCH --ntasks-per-node=1        # 1 srun task per node; torchrun fans out to all GPUs
#SBATCH --gpus-per-node=4          # N GPUs per node
#SBATCH --cpus-per-task=16         # CPU cores per node

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

srun --nodes=$SLURM_NNODES --ntasks-per-node=1 bash -c "
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=\$SLURM_GPUS_ON_NODE \
    --node_rank=\$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    run_train.py
"
```

---

## Restart and Resume

Two ways to resume:

```python
# 1) automatic: looks for checkpoint.pt in output_dir (restart=True default)
train_nep("nep.in", "train.xyz", output_dir="output")

# 2) explicit: continue from a specific checkpoint
train_nep("nep.in", "train.xyz", output_dir="output",
          resume_from="output/checkpoint_stage1.pt")
```

### What you can safely change on restart

| Parameter | Safe to change? | Notes |
|-----------|----------------|-------|
| `epoch` | Yes | Extend training by increasing this |
| `lambda_e` / `lambda_f` / `lambda_v` | Yes | New weights take effect next epoch. |
| `stage2_lambda_e` / `stage2_lambda_f` / `stage2_lambda_v` | Yes | Same auto-reset rule. |
| `batch` | Yes | — |
| `stage2`, `start_stage2` | Yes | Add Stage 2 to a run that did not have it, or push it later |
| `stage2_lr` | Only at the transition | Applied **once**, when training first crosses Stage 1 → Stage 2. If you resume from a checkpoint that was *already* in Stage 2, the checkpoint's current (possibly-decayed) LR is kept — editing `stage2_lr` then has no effect. To re-enter Stage 2 with a new LR, `resume_from=".../checkpoint_stage1.pt"`. |
| `lr_scheduler` (`plateau` ↔ `step`) | Yes | Scheduler state from the old mode is incompatible and silently discarded; the new scheduler starts fresh from the current LR |
| `scheduler_patience` / `scheduler_factor` | Yes | Applied immediately |
| `stage2_scheduler_patience` / `stage2_scheduler_factor` | Yes | Applied immediately to the Stage 2 scheduler |
| `lr` (Stage 1) | **No** | Resume keeps the checkpoint's LR |
| Architecture (`neuron`, `cutoff`, `n_max`, `basis_size`, `l_max`, `type`) | **No** | Dimensions are fixed in the saved weights |

---

## Fine-Tuning

Fine-tuning starts from a pre-trained model's weights instead of random initialisation. The architecture (`nep.in` parameters) must match the source model, but the new dataset's element types may be a subset of the original.

### Basic fine-tuning

```python
train_nep(
    "nep.in",
    "new_data.xyz",
    output_dir="finetune_output",
    finetune_from="pretrained/nep.txt",   # or a "pretrained/checkpoint.pt"
    slim_types=True,
)
```

`finetune_from` accepts:
- `nep.txt` — GPUMD text format (works with models trained by GPUMD or TorchNEP)
- `checkpoint.pt` — full checkpoint (weights are extracted automatically)

If the new dataset contains fewer element types than the original model, `slim_types=True` removes the unused types **before training begins**, shrinking the model and speeding up training.

### Standalone model slimming

```python
from torchnep.model import NEPModel, slim_model
from torchnep.data import parse_nep_in

config = parse_nep_in("nep.in")
model = NEPModel(config)
model.load_weights_from_nep_txt("nep.txt")

slimmed = slim_model(model, ["Cr", "Ni"])
slimmed.save_nep_txt("nep_slim.txt", max_NN_radial, max_NN_angular)
```

---

## Prediction

### Single-structure prediction

```python
from torchnep.nep import NEPCalculator
import numpy as np

calc = NEPCalculator("nep.txt")
result = calc.compute(
    species=["Cr", "Cr", "Ni"],
    positions=np.array([[0,0,0],[1.5,0,0],[3,0,0]]),
    cell=np.eye(3) * 6.0,
)
print(result["energy"])         # (N,) per-atom energy (eV); sum for total
print(result["forces"])         # (N, 3) forces (eV/Å)
print(result["virial"])         # (N, 9) per-atom virial (eV)

# Split the NEP (neural-network) part from the ZBL repulsive part:
result = calc.compute(..., return_components=True)
print(result["energy_nep"], result["energy_zbl"])   # sum == result["energy"]
```

### ASE calculator

If ASE is installed, any ASE workflow (relaxation, MD, EOS, …) can drive a trained model:

```python
from ase.io import read
from torchnep.ase_calculator import NEP

atoms = read("POSCAR")
atoms.calc = NEP("nep.txt", dtype='float32', device='cuda')
print(atoms.get_potential_energy())   # eV
print(atoms.get_forces())             # (N, 3) eV/Å
print(atoms.get_stress())             # Voigt 6-vector eV/Å³ (periodic cells)

# NEP / ZBL / total breakdown of energy, forces, and stress:
parts = atoms.calc.get_components()
print(parts["nep"]["energy"], parts["zbl"]["energy"], parts["total"]["energy"])
```

### Full-dataset prediction

Runs batched GPU inference on an entire `.xyz` file and writes GPUMD-compatible output files.

```python
from torchnep import predict_dataset

predict_dataset(
    "nep.txt",
    "test.xyz",
    output_dir="results",
    dtype="float32",       # float32 or float64
    batch_size=500,
    output_descriptor=0,   # 0=off, 1=per-frame mean, 2=per-atom (matches GPUMD)
)
# writes energy_train.out, force_train.out, virial_train.out,
# stress_train.out, and (when output_descriptor != 0) descriptor.out
```

---

## Source layout

The `torchnep/` package is organised as follows:

| File | Role |
|------|------|
| `__init__.py` | Public API — re-exports the three entry points `train_nep`, `train_nep_sharded`, `predict_dataset` |
| `data.py` | I/O and parsing — reads extended-XYZ frames and `nep.in`, plus the NumPy brute-force neighbor builder used for training |
| `neighbor.py` | PyTorch linked-cell (cell-list) neighbor search, O(N) for the large structures of an ASE-driven MD run |
| `model.py` | Trainable NEP4 model (`NEPModel`) as an `nn.Module`, per-type fitting nets, ZBL, and `slim_model` |
| `ops.py` | Core differentiable kernels — Chebyshev/angular basis, descriptors, ANN evaluation, ZBL; pure-PyTorch `loop`/`bmm` backends |
| `nep.py` | `NEPCalculator` — loads a `nep.txt` and computes energy/forces/virial/descriptors for single structures |
| `predict.py` | Batched full-dataset inference (`predict_dataset`), writing GPUMD-compatible `*_train.out` files |
| `train.py` | Single-GPU/CPU training (`train_nep`): data store, two-stage loop, schedulers, checkpoint/restart, periodic predict |
| `train_sharded.py` | Data-sharded multi-GPU/multi-node training (`train_nep_sharded`) via DDP |
| `ase_calculator.py` | ASE `Calculator` wrapper (`NEP`) for relaxation, MD, EOS, phonons, … |
| `constants.py` | Shared constants — element table, covalent radii, NEP polynomial coefficients |

---

## Citation

If you use TorchNEP in your research, please cite the following paper:


```bibtex
@misc{wu2026torchne,
      title={TorchNEP: Ultra-Efficient and Accurate Training of Neuroevolution Potentials}, 
      author={Yong-Chao Wu and Xiaoya Chang and Tero Mäkinen and Amin Esfandiarpour and Jian-Li Shao and Tapio Ala-Nissila and Zheyong Fan and Mikko Alava},
      year={2026},
      eprint={2606.19557},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2606.19557}, 
}
```