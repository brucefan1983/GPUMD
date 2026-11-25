# GPUMD pytest suite

Automated tests for `gpumd` and `nep`, including numerical consistency, parser,
output, training, spin and replica checks. These complement the executable
comparisons in [tests_regression](../tests_regression/README.md).

## Prerequisites

- A GPU-equipped machine for executable-level tests. Parser and host-side checks
  can run without a GPU.
- Python packages: `ase`, `calorine`, `numpy`, `pytest`, `hypothesis`. Install with:

  ```bash
  pip install ase calorine numpy pytest hypothesis
  ```

## Building `gpumd` and `nep`

```bash
cd src
make
```

This produces `src/gpumd` and `src/nep`. See `doc/installation.rst` at the repo root for full
build prerequisites.

### How the suite finds the executables

`conftest.py` looks for each executable at `<repo root>/src/gpumd` and `<repo root>/src/nep`
first — where the build above puts them, so a normal build needs no extra setup — and falls back
to `<repo root>/gpumd` and `<repo root>/nep` if nothing exists at the default location (e.g. a
manually placed copy or symlink).

Training and prediction tests use `nep`; simulation tests use `gpumd`. Standalone
validation scripts document their executable options in `--help` or their module
docstrings.

For spin validator setup and executable selection, see
[Spin validation](SPIN_VALIDATION.md).

## Running the tests

From this directory (`tests_pytest/`):

```bash
pytest -q                 # everything
pytest -m "not slow" -q   # excludes MD-conservation-scale tests (the expensive tier)
pytest -m fast -q         # single-point-evaluation / quick-command subset only
```

Two extra flags control fixture regeneration, both deliberate/manual (never automatic):

```bash
pytest --update-golden     # regenerate the frozen reference files under fixtures/golden/
pytest --dump-fixtures     # write representative run.in/model.xyz pairs into
                            # fixtures/sanitizer_inputs/, then exit without running tests --
                            # consumed by run_sanitizer_checks.sh (compute-sanitizer, run
                            # manually and separately, not part of the pytest run)
```

## Fixture layout

```
fixtures/
  models/       # NEP/qNEP/TNEP model files (nep.txt format)
  structures/   # pre-rattled structure files (extended XYZ)
  golden/       # frozen reference outputs (energy/force/virial/BEC/TNEP arrays)
  training/     # small labelled training set for the `nep` training runs, plus the script
                # that generates it
```

`fixtures/nep_spin/` contains spin models and independent reference data;
`fixtures/replica/` contains the replica fixtures.

`fixtures/sanitizer_inputs/` is generated on demand via `--dump-fixtures` above
and ignored by Git. Python and pytest caches are also ignored. These generated
files can be removed when their diagnostics are no longer needed; keep the
tracked models, structures and golden reference files.

The `generate_*.py` and `generate_*.cpp` tools reproduce fixtures or generated
headers. The `validate_*.py` and `validate_*.cu` tools provide additional runtime,
multi-GPU and kernel checks; some are also imported by pytest tests. They are
maintained test tools and are not all run by `pytest` discovery. Their additional
dependencies are documented in the scripts.

## Replica dynamics

`test_replica.py` holds 20 integration tests of replica dynamics, i.e. `gpumd` runs with a
`multi_replica` command, marked `slow`. They need only the Python standard library and `pytest`
(the packages above are needed because pytest loads `conftest.py`):

```bash
pytest test_replica.py -q
```

The file can also be run directly, without `conftest.py`, from the repository root:

```bash
python3 tests_pytest/test_replica.py
```

Both DEBUG and release builds are supported; fixed initial seeds are not required. The tests
use the first device selected by `CUDA_VISIBLE_DEVICES`, or device `0` if that variable is unset;
multiple replicas share that GPU. `GPUMD` selects another executable:

```bash
GPUMD=/path/to/gpumd CUDA_VISIBLE_DEVICES=1 pytest test_replica.py -q
```

The tests check:

- exchange attempts at every interval for two replicas;
- alternating neighbor pairs, accepted swaps, and label mappings for 4/8 replicas;
- initial PRD correlation, unaccelerated correlation exits, discrete event clocks,
  and the lowest-index winner for simultaneous exits;
- continuation of REMD temperature mappings and exchange schedules between exchanges,
  with batching forced off and on, and restoration of PRD phases and physical clocks;
- the standard restart velocity layout and older files with `vel_natural`, both
  restoring known velocities and writing the standard layout on completion;
- rejection of unsupported restart versions and changed exchange/dephasing settings;
- qNEP Ewald/PPPM selection, inline comments, default solver, and restart validation,
  including a changed PPPM spacing;
- two concurrent qNEP replicas on the large-box path (2000-atom PbTe supercell) with
  Ewald and PPPM: repeated runs from the same checkpoint agree (with PPPM, whose charge
  spreading uses floating-point atomics, within `1e-6`), and split continuation
  preserves valid checkpoint states and step counts;
- the initial qNEP energy (Ewald and PPPM, small-box and large-box path) and the
  initial NEP energy for a model.xyz with only some species of the model, both equal
  to those of an ordinary `gpumd` run;
- single-segment dephasing syntax and the retry limit;
- rejection of the unsupported `verbose` and `verbose_output` options in both modes;
- batched integration (forced with `GPUMD_REPLICA_BATCH=1`) reproducing the per-slot
  path (`GPUMD_REPLICA_BATCH=0`) bit for bit for REMD and PRD, on a 2000-atom triclinic
  PbTe supercell that uses the large-box NEP path;
- the automatic batching choice across replica sizes and total atom counts, the
  forced-off setting, and rejection of a `batch` keyword in `run.in`.

The four-carbon fixture in `fixtures/replica/` has a zero-energy NEP and is used to obtain
predictable exchange and event-clock behavior. It is not a physical model. Force-bearing
NEP/qNEP tests reuse the tracked PbTe examples. Continuation tests check restored states,
step counts, and method behavior; they do not require a segmented run to reproduce an
uninterrupted trajectory, exchange outcomes, or event times. RNG states are saved and
restored, but this does not establish a bitwise restart guarantee. Batch comparisons
start from the same checkpoint and retain their bitwise checks.

Tests run in isolated temporary directories and normally delete them afterwards.
Set `REPLICA_TEST_WORKDIR` to retain test directories, including outputs and
the last `run.in` and `run.log` for each case:

```bash
REPLICA_TEST_WORKDIR=/tmp/replica-checks \
  python3 tests_pytest/test_replica.py
```

A missing executable, unavailable GPU, process timeout (120 seconds per run),
or failed assertion fails the test. The suite does not modify CI and
does not cover multiple GPUs, long-time sampling accuracy, or performance.

The NPT extension of REMD uses the same `multi_replica remd` entry and is covered
by `test_remd_npt.py`.
Run `python3 tests_pytest/test_remd_npt.py` with `GPUMD` set to the candidate binary.
It checks that NPT uses the isotropic `npt_scr` syntax of ordinary MD, rejects
the other pressure-control forms and invalid pressure ladders or grids, and defaults
to the `npt_scr` pressure without a ladder. Exchange pairs and acceptance ratios in
`remd_npt_exchange.out` are compared with the documented schedule and an independent
formula on rectangular, sheared, and two-condition grids. The suite includes
strained small/large-box force and full per-atom virial comparisons against ordinary
GPUMD, nonzero-virial pressure checks after BDP and accepted temperature exchange,
nontrivial acceptance probabilities, rejection of corrupted SCR states, and repeated
continuation from the same checkpoint with box/mapping/SCR/RNG checks.
Set `GPUMD_BASELINE` to a matching upstream executable to also check that its
NVT format-5 checkpoint can be continued by the candidate without changing the
existing metadata layout or replica dynamics.
Set `REPLICA_SAMPLING=1` for ideal-gas sampling; `REPLICA_SAMPLING_STEPS` changes
its length (at least 1000000 steps). Ordinary MD reference snapshots use zero MD
time step and `dump_xyz ... precision double mass velocity force potential virial`.

## Spin validation

See [SPIN_VALIDATION.md](SPIN_VALIDATION.md) for CPU checks, single-GPU model
evaluation and training, integration/restart checks, and four-GPU comparisons.
It also documents how to run the ordinary GPUMD/NEP regression alongside spin checks.

## Spin training precision

From the repository root, with CUDA `nvcc` on `PATH` and an appropriate GPU
selected in `CUDA_VISIBLE_DEVICES`:

```bash
python tests_pytest/validate_spin_ann_precision.py --arch sm_89
```

Choose `--arch` for the target GPU. This check compares the spin-private ANN
accumulator with an independent host long-double oracle, including small energy
differences and finite-difference derivatives for its supported single hidden layer.
For training/restart/prediction and native/JIT checks, follow
[Spin validation](SPIN_VALIDATION.md#training-prediction-and-restart). ANN-only precision is not a
bound on geometry or descriptor rounding in a full potential evaluation.

## Spin generated headers

Regenerate with NumPy, Torch, e3nn and clang-format 17 on `PATH`:

```bash
python tests_pytest/generate_spin_tables.py
c++ -std=c++14 -Isrc tests_pytest/generate_spin_md_layouts.cpp -o /path/to/generator
/path/to/generator | clang-format --style=file > src/utilities/nep_spin/spin_md_layouts.cuh
```

Both outputs use the project license header and repository formatting.
