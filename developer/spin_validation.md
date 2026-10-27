# Spin training regression checks

Run from the repository root. Use Python with NumPy and pytest, a C++ compiler,
and CUDA for GPU checks. Set `RESULTS` to an existing absolute directory outside
the checkout. `NEP_COMMAND` selects the training executable; the default is
`src/nep`. GPU commands require allocated devices in `CUDA_VISIBLE_DEVICES`.

## CPU checks

```sh
python -m pytest --noconftest -q tests_pytest/test_spin_model_format.py \
  tests_pytest/test_spin_validation.py
```

These check the current model codec, finite numerical comparisons and rotation
response geometry. Both C++ probes use strict C++14. No GPU is required.
`--noconftest` avoids the ordinary NEP suite's ASE/Calorine fixtures.

## Build and single-GPU checks

Build on the target system. This example targets V100; change `CUDA_ARCH` for
other GPUs. The make fragment links the probes against the same NEP objects and
libraries as the executable. Use a fresh build after changing compiler flags.

```sh
make -C src -j8 -f makefile -f ../tests_pytest/spin_probes.mk \
  CUDA_ARCH=-arch=sm_70 SPIN_TEST_BIN="$RESULTS/probes" nep spin-probes
python tests_pytest/validate_main_nep_spin3_training.py --check-jit
python tests_pytest/validate_spin_training_lifecycle.py
RESTART_PROBE="$RESULTS/probes/restart_parity" \
  python tests_pytest/validate_restart_scaler.py
RESPONSE_FITNESS_PROBE="$RESULTS/probes/response_fitness" \
  python tests_pytest/validate_response_file_runtime.py
python tests_pytest/validate_spin_mforce_split.py --devices 1 \
  --oracle "$RESULTS/probes/spin_mforce_split" --output "$RESULTS/split-single"
```

Expose one allocated GPU for this section. `--check-jit` requires Linux/CUDA with
`nvcc` available and successful structural and Spin layout specialization; a
fallback is a failure. Omit that flag for native-only checks.

The main validator checks model IO, defaults, fixed-parameter energy, force,
magnetic force and virial against 24 frozen FP64 cases. Universal ZBL remains
supported; flexible ZBL is rejected. Seven native/JIT layout pairs cover
compression channels 1/2/3/9, `spin_order` 1/2/3, `spin_l_max` 0/1/2 and
`spin_soc` 0/1. Both Fe and Ge have nonzero spins in those layout comparisons.
They compare energy, force, magnetic force and virial at identical parameters.
Lifecycle checks cover prediction/restart, imported scalers and masking excluded
types without changing energy or force labels. The restart probe separately
checks SNES mean/sigma restoration, the exported elite, scaler initialization,
energy references and variable-type assignment. Magnetic and response probes
check projections, zero spins, weighting, loss normalization and data isolation.

## Four-GPU checks

Expose four allocated GPUs. These commands explicitly reproduce the
native/JIT × 1/2/4 GPU training matrix and magnetic-loss decomposition:

```sh
python tests_pytest/validate_spin_training_multigpu.py --compile both \
  --output "$RESULTS/training-matrix"
python tests_pytest/validate_spin_mforce_split.py --devices 1 2 4 \
  --oracle "$RESULTS/probes/spin_mforce_split" --output "$RESULTS/split-multigpu"
```

The matrix compares restart parameters, energy references, loss components and
saved-model predictions. Batch sizes 1 and 3 distinguish the centered single-frame
negative control from an actual energy-fitting batch. Energy error is recomputed
from predictions and original labels. These short runs check software contracts,
not predictive accuracy on unseen data. The ordinary `tests_regression` suite is
a separate regression gate.

## Frozen reference data

See [fixture provenance and reproduction](../tests_pytest/fixtures/nep_spin3/README.md)
for fixed independent FP64 sources and regeneration commands. SHA256 binds each
oracle to its model. Never generate expected values with the candidate CUDA
implementation to make a failing check pass. Models use `model nep_spin`,
`spin_version 1` and named parameter/scaler blocks; restart uses `nep_restart`.

Keep build logs, source/binary hashes, timing and numerical reports outside the
source tree.
