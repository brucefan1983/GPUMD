# GPUMD simulation regression tests

This directory contains the 231 `gpumd` cases, their fixtures, and all 25
cross-case relations. MD simulations using NEP potentials belong here;
training and prediction with the `nep` executable belong in `../nep/`.

Run from the repository root using the shared entry point:

```bash
python3 tests_regression/run_regression.py \
  --repo-root "$PWD" \
  --baseline "$PWD/src_old/gpumd" \
  --candidate "$PWD/src/gpumd" \
  --suite full --program gpumd --device 0
```

This selects the simulation part of the full suite. Acceptance of the complete
regression package also requires the NEP cases; use the combined command in
the [main README](../README.md) to run all 255 cases and 25 relations.

Paths beginning with `package:` in this directory's `manifest.json` are
relative to `tests_regression/`. Public potentials and examples use `repo:`.
See [CONTRACT_SUITES.md](CONTRACT_SUITES.md) and [PPPM.md](PPPM.md) for checks
involving several cases and PPPM-specific checks.
