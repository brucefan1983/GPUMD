# compute_react_count core regression

This CPU-only test compiles the production core with
`REACT_COUNT_CORE_ONLY` and checks two cases that exercise the short
periodic-cell path:

- 25 independent water molecules in a 20 x 20 x 4.2 Angstrom box with
  `perception mutual`: 25 H2O and 50 O-H bonds;
- 80 diamond atoms in a 10 x 1 x 1 conventional-cell box with the default
  `reax` perception: all 80 carbons are four-coordinate.

Run:

```sh
./run.sh
```

The test needs only a C++14 compiler and Python 3.  The full 1500-frame
combustion comparison against ReaxTools is maintained in the ReaxTools
repository under `test/combustion_react_count_gpumd`.
