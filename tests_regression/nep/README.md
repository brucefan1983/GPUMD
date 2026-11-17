# NEP training and prediction regression tests

This directory contains the 24 cases executed by `nep` and their fixtures.
Its manifest declares the training and prediction inputs as `nep.in` files.
MD simulations using NEP potentials belong in `../gpumd/`.

Run from the repository root using the shared entry point:

```bash
python3 tests_regression/run_regression.py \
  --repo-root "$PWD" \
  --baseline-nep "$PWD/src_old/nep" \
  --candidate-nep "$PWD/src/nep" \
  --suite full --program nep --device 0
```

Use matching builds with `-DDEBUG` and keep each executable in its corresponding
compiled source tree so runtime-specialized kernels use the matching sources.
Paths beginning with `package:` are relative to `tests_regression/`; `repo:`
paths refer to the supplied repository root.

This selection has no cross-case relations of its own. Relations belonging to
the GPUMD cases are reported as skipped. Use the combined command in the
[main README](../README.md) for acceptance of all 255 cases and 25 relations.
