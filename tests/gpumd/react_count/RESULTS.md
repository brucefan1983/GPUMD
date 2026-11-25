# compute_react_count regression record

Environment: CUDA 12.8, RTX 5090, C++14 Make build and C++17 CMake build.

## Short-cell core regression

```sh
tests/gpumd/react_count/run.sh
```

Result: PASS.  25 H2O, 50 O-H bonds, 80 four-coordinate diamond atoms, and
the append check all pass.

## Full combustion comparison

The 1500-frame combustion trajectory from ReaxTools was analyzed by the
host-only core with `perception reax` and compared with the ReaxTools
reference:

```text
species: 14363 got keys, 14363 ref keys, 0 mismatches
bond:    6407 got keys,  6407 ref keys, 0 mismatches
coord:   15000 got keys, 15000 ref keys, 0 mismatches
events:  11813 got keys, 11813 ref keys, 0 mismatches
RESULT: PASS
```

The GPUMD live test on the same 1500-atom model compares GPU output with the
core output for both `reax` and `mutual`; all four output files are identical.
