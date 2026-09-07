# Replica regression tests

Build and run from the repository root with a CUDA toolkit and a visible GPU:

```sh
make -C src/main_replica -j4
python3 tests/replica/test_replica.py
```

The tests use Python's standard library. Set `GPUMD_REPLICA` to the path of an
alternative executable. Each simulation runs in its own temporary directory.

Coverage includes discrete PRD clocks and coincident exits with event intervals
1 and 4, initial correlation and its restart state, REMD/PRD split-run
consistency, qNEP inline comments, and Ewald/PPPM restart compatibility.
Ballistic event fixtures use a zero-energy NEP to control the exit step; they
test scheduling and clock accounting, not physical transition rates.

Replica restart format 4 records the effective `kspace` method (`none` for
ordinary NEP). Earlier formats are rejected because they do not identify the
electrostatic solver and PRD used different clock and event-ordering semantics.
Start a fresh run when upgrading from those formats; changing the version
number in an old checkpoint does not convert its state.
