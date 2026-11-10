# GPUMD pytest suite

An automated pytest-based test suite for `gpumd`/`nep`, living alongside the existing manual
`tests/` directory (which this suite does not touch or replace). See
`gpumd_pytest_suite_spec.md` at the repo root for the full design rationale, fixture
architecture, and numeric-validation strategy; this file only covers setup and how to run things.

## Prerequisites

- A GPU-equipped machine — every test in this suite drives the real `gpumd` binary.
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

`nep` is used only by `test_nep_model_consistency.py`; every other module drives `gpumd`.

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

`fixtures/sanitizer_inputs/` is generated on demand via `--dump-fixtures` above, not checked in.

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
- uninterrupted versus split continuation from the same checkpoint, including
  every replica's thermostat RNG state;
- rejection of unsupported restart versions and changed exchange/dephasing settings;
- qNEP Ewald/PPPM selection, inline comments, default solver, and restart validation,
  including a changed PPPM spacing;
- two concurrent qNEP replicas on the large-box path (2000-atom PbTe supercell) with
  Ewald and PPPM: a repeated run and a run split into resumed segments are identical
  (with PPPM, whose charge spreading uses floating-point atomics, within `1e-6`);
- the initial qNEP energy (Ewald and PPPM, small-box and large-box path) and the
  initial NEP energy for a model.xyz with only some species of the model, both equal
  to those of an ordinary `gpumd` run;
- single-segment dephasing syntax and the retry limit;
- rejection of the unsupported `verbose` and `verbose_output` options in both modes;
- batched integration (forced with `GPUMD_REPLICA_BATCH=1`) reproducing the per-slot
  path (`GPUMD_REPLICA_BATCH=0`) bit for bit for REMD and PRD, on a 2000-atom triclinic
  PbTe supercell that uses the large-box NEP path;
- the automatic batching choice (on for 4 x 2000 atoms, off for 12 x 2000 atoms or when
  forced off) and rejection of a `batch` keyword in `run.in`;
- an uninterrupted REMD run and the same run split into two resumed segments giving
  identical restart files and exchange records, with batching forced off and on.

The four-carbon fixture in `fixtures/replica/` has a zero-energy NEP and is used to obtain
predictable exchange and event-clock behavior. It is not a physical model. Force-bearing
NEP/qNEP tests reuse the tracked PbTe examples. Restart metadata is compared
exactly. Restart atom coordinates, masses, and velocities use an absolute
tolerance of `1e-10` for text serialization and velocity-unit roundoff.

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
