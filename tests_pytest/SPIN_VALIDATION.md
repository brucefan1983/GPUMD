# Spin validation

Run commands from the repository root. Use Python with NumPy and pytest, a
C++ compiler for the CPU checks, and CUDA for the GPU checks below. The CPU
commands use `--noconftest` to avoid loading the ordinary suite's ASE/Calorine
fixtures. GPU checks run the public executables through their input and output
files.

## Build and environment

Select an architecture supported by the GPU and toolkit; `sm_89` is an example:

```sh
make -C src -j4 CUDA_ARCH=-arch=sm_89 gpumd nep
export GPUMD_COMMAND="$PWD/src/gpumd"
export NEP_COMMAND="$PWD/src/nep"
export RESULTS="$(mktemp -d /tmp/gpumd-spin-validation.XXXXXX)"
```

Use a fresh build after changing compiler flags. Keep source and binary hashes,
toolchain details, GPU allocation, logs, and numerical reports with the results
outside the source tree. Use a new results directory for each validation run.

## CPU checks

```sh
python3 -m pytest --noconftest -q \
  tests_pytest/test_spin_model_format.py tests_pytest/test_spin_validation.py
```

These check the public model codec and finite numerical comparisons without a
GPU. The codec check compiles with strict C++14.

## Single-GPU checks

Expose one allocated device through `CUDA_VISIBLE_DEVICES`. For native/JIT
checks, use Linux/CUDA with `nvcc` available. `--require-jit` and `--check-jit`
require successful specialization; fallback to the generic path fails these
checks. Omit those flags only when intentionally checking the native path.

### Model evaluation and external fields

```sh
python3 tests_pytest/validate_spin_model.py \
  --workdir "$RESULTS/model" --require-jit
python3 tests_pytest/validate_mfield_runtime.py
```

The model validator compares energy, atomic force, magnetic force, and virial
with frozen independent FP64 references. It also checks geometry, excluded-spin
masking, energy references, and model I/O. See
[fixture provenance](fixtures/nep_spin/README.md) for reference generation and
source/model hashes. Never regenerate expected values with the candidate CUDA
implementation to make a failing check pass.

The field validator checks Zeeman energy and magnetic force, group and region
selection, boundary crossing, invalid inputs, additive fields, and field
clearing/redeclaration across runs.

### Training, prediction, and restart

```sh
python3 tests_pytest/validate_spin_training.py --check-jit
python3 tests_pytest/validate_spin_training_lifecycle.py
python3 tests_pytest/validate_spin_full_virial.py --check-jit
python3 tests_pytest/validate_response_file_runtime.py
python3 tests_pytest/validate_spin_mforce_split.py --devices 1 \
  --output "$RESULTS/split-single"
```

These checks cover input validation, training/prediction and restart,
native/JIT agreement at fixed parameters, descriptor scaling, magnetic losses,
and response supervision. They check software behavior rather than convergence
or predictive accuracy on unseen data. Keep model, seed, training budget, and
GPU settings fixed when comparing revisions. For the separate ANN precision
check, see [the pytest README](README.md#spin-training-precision).

### Integration and minimization

```sh
python3 -m pytest --noconftest -q \
  tests_pytest/test_spin_tspin.py tests_pytest/test_spin_restart.py \
  tests_pytest/test_spin_sib.py tests_pytest/test_spin_fire.py \
  tests_pytest/test_spin_force_modifiers.py
```

TSPIN is compared with an independent one-step formula; SIB with independent
Cayley and thermal-increment references. Restart checks cover physical-state
round trips, new NVT/NPT runs, and output-independent run boundaries. FIRE
checks cover spin, atomic, and cell relaxation. Field and force-modifier checks
cover minimization/MD separation, corrector behavior, and cleanup across runs.

## Four-GPU checks

Expose four allocated devices through `CUDA_VISIBLE_DEVICES`. These commands
exercise one-, two-, and four-GPU configurations within that allocation:

```sh
python3 tests_pytest/validate_spin_multigpu.py \
  --binary "$GPUMD_COMMAND" \
  --model tests_pytest/fixtures/nep_spin/c3l2s2_fege.nep \
  --output "$RESULTS/runtime"
python3 tests_pytest/validate_spin_glsd_runtime.py \
  --binary "$GPUMD_COMMAND" --output "$RESULTS/glsd"
python3 tests_pytest/validate_spin_integrators_multigpu.py \
  --binary "$GPUMD_COMMAND" --output "$RESULTS/integrators"
python3 tests_pytest/validate_spin_training_multigpu.py --compile both \
  --output "$RESULTS/training-matrix"
python3 tests_pytest/validate_spin_mforce_split.py --devices 1 2 4 \
  --output "$RESULTS/split-multigpu"
```

The runtime comparison covers fixed-spin energies, forces, magnetic forces,
and virials for different geometries, cutoffs, partition directions, fields,
energy references, and motion across partitions. It rejects nonfinite results
and uses componentwise tolerances. Add `--reference /path/to/reference/gpumd`
to compare against a separately built reference on one GPU; otherwise, the
candidate executable supplies the one-GPU reference.

GLSD checks analytic trajectories, radial relaxation, physical-state restart,
noise seeding, and GPU-count parity. The integrator comparison checks TSPIN/SIB
trajectories and magnetic endpoints. The training matrix compares native/JIT
results, restart parameters, energy references, losses, and saved-model
predictions. These checks do not establish equilibrium sampling of an
arbitrary trained potential or measure performance.

## Ordinary GPUMD and NEP regression

Spin checks supplement the ordinary regression suite. Use independently built
accepted baseline executables and candidates from the same source snapshot as
the spin checks. Match build flags, including `-DDEBUG` for NEP training:

```sh
python3 tests_regression/run_regression.py --repo-root "$PWD" \
  --baseline "$BASELINE_GPUMD" --candidate "$GPUMD_COMMAND" \
  --baseline-nep "$BASELINE_NEP" --candidate-nep "$NEP_COMMAND" --suite full
```

Set `BASELINE_GPUMD` and `BASELINE_NEP` to the accepted executable paths before
running. For a focused NEP-only diagnostic run, omit the GPUMD executable
arguments and add `--program nep`; that selection does not validate MD
relations. Follow the [regression instructions](../tests_regression/README.md)
for baseline builds, the full acceptance contract, and reports.
