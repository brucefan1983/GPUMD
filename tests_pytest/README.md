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
