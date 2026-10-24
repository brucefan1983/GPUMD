# GPUMD pytest suite

An automated pytest suite for `gpumd` and `nep`.

## Prerequisites

Every test except those in `test_parsing.py` drives the real `gpumd` binary, so the suite needs
a GPU-equipped machine.

```bash
pip install ase calorine numpy pytest hypothesis netCDF4 MDAnalysis
```

`netCDF4` and `MDAnalysis` are read by the `dump_netcdf` tests. Without them those tests skip
and the run still reports success.

## Building `gpumd` and `nep`

```bash
cd src
make
```

This produces `src/gpumd` and `src/nep`. Add `-DUSE_NETCDF` to cover the `dump_netcdf` tests,
which otherwise skip; see `doc/installation.rst` for the NetCDF setup and the rest of the build
prerequisites.

`conftest.py` takes each executable from `<repo root>/src/` and falls back to `<repo root>/`.
`nep` is used only by `test_nep_model_consistency.py`.

## Running the tests

From this directory:

```bash
pytest -q                 # everything
pytest -m "not slow" -q   # excludes the MD-conservation tests, which are the expensive tier
pytest -m fast -q         # single-point evaluations and quick commands only
```

Two flags regenerate fixtures:

```bash
pytest --update-golden    # rewrite the reference files under fixtures/golden/
pytest --dump-fixtures    # write run.in/model.xyz pairs into fixtures/sanitizer_inputs/, then
                          # exit; read by run_sanitizer_checks.sh, which drives
                          # compute-sanitizer separately from the pytest run
```

## Fixture layout

```
fixtures/
  models/       # NEP, qNEP and TNEP model files in nep.txt format
  structures/   # pre-rattled structures in extended XYZ
  golden/       # frozen reference outputs
  training/     # small labelled training set for the nep runs, and the script that writes it
```

`fixtures/sanitizer_inputs/` is written by `--dump-fixtures` and is not committed.
