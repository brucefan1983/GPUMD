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

These check the public model codec and finite numerical comparisons. The codec
check uses strict C++14. No GPU is required.
`--noconftest` avoids the ordinary NEP suite's ASE/Calorine fixtures.

## Build and single-GPU checks

Build on the target system. This example targets V100; change `CUDA_ARCH` for
other GPUs. Use a fresh build after changing compiler flags. All GPU checks
run the actual executable through its input files and exported results.

```sh
make -C src -j8 CUDA_ARCH=-arch=sm_70 nep
python tests_pytest/validate_main_nep_spin3_training.py --check-jit
python tests_pytest/validate_spin_training_lifecycle.py
python tests_pytest/validate_response_file_runtime.py
python tests_pytest/validate_spin_mforce_split.py --devices 1 \
  --output "$RESULTS/split-single"
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
types without changing energy or force labels. Frozen-parameter restart checks
compare predictions before and after resuming through the CLI. Response checks
cover input rejection, group leakage, disabled response, equal group weights and
isolation from unused labels. Magnetic errors are independently recomputed from
exported predictions, including parallel/perpendicular components. Tests do not
inspect private model buffers or optimizer state.

## Four-GPU checks

Expose four allocated GPUs. These commands explicitly reproduce the
native/JIT × 1/2/4 GPU training matrix and magnetic-loss decomposition:

```sh
python tests_pytest/validate_spin_training_multigpu.py --compile both \
  --output "$RESULTS/training-matrix"
python tests_pytest/validate_spin_mforce_split.py --devices 1 2 4 \
  --output "$RESULTS/split-multigpu"
```

The matrix compares restart parameters, energy references, loss components and
saved-model predictions. Batch sizes 1 and 3 distinguish the centered single-frame
negative control from an actual energy-fitting batch. Energy error is recomputed
from predictions and original labels. These short runs check software contracts,
not predictive accuracy on unseen data. The ordinary `tests_regression` suite is
a separate regression gate. Use baseline and candidate NEP executables built
with the same flags, including `-DDEBUG` as required by that suite:

```sh
python tests_regression/run_regression.py --repo-root "$PWD" \
  --baseline-nep "$BASELINE_NEP" --candidate-nep "$NEP_COMMAND" \
  --suite full --program nep
```

This selects the NEP cases; it does not validate MD relations. See the
[regression runner instructions](../tests_regression/README.md) for build and
report details.

## Frozen reference data

See [fixture provenance and reproduction](../tests_pytest/fixtures/nep_spin3/README.md)
for fixed independent FP64 sources and regeneration commands. SHA256 binds each
oracle to its model. Never generate expected values with the candidate CUDA
implementation to make a failing check pass. Models use `model nep_spin`,
`spin_version 1` and named parameter/scaler blocks; restart uses `nep_restart`.

Keep build logs, source/binary hashes, timing and numerical reports outside the
source tree.
