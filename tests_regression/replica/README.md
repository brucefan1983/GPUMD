# Replica regression tests

These 12 basic integration tests exercise the standalone `gpumd_replica`
executable. They run independently of the parent directory's 221-case
GPUMD/NEP differential suite and do not require a historical executable.

From the repository root, build with the ordinary `make` command and run:

```bash
make -C src -j
python3 tests_regression/replica/test_replica.py
```

Requirements are Python 3.9 or newer (standard library only) and one CUDA GPU.
Both DEBUG and release builds are supported; fixed initial seeds are not
required. The script uses the first device selected by `CUDA_VISIBLE_DEVICES`,
or device `0` if that variable is unset. Multiple replicas share that GPU.
To select another executable and GPU:

```bash
GPUMD_REPLICA=/path/to/gpumd_replica CUDA_VISIBLE_DEVICES=1 \
  python3 tests_regression/replica/test_replica.py
```

The tests check:

- exchange attempts at every interval for two replicas;
- alternating neighbor pairs, accepted swaps, and label mappings for 4/8 replicas;
- initial PRD correlation, unaccelerated correlation exits, discrete event clocks,
  and the lowest-index winner for simultaneous exits;
- uninterrupted versus split continuation from the same checkpoint, including
  every replica's thermostat RNG state;
- rejection of unsupported restart versions and changed exchange/dephasing settings;
- qNEP Ewald/PPPM selection, inline comments, default solver, and restart validation;
- single-segment dephasing syntax and the retry limit;
- rejection of the unsupported `verbose` and `verbose_output` options in both modes.

The four-carbon fixture has a zero-energy NEP and is used to obtain predictable
exchange and event-clock behavior. It is not a physical model. Force-bearing
NEP/qNEP tests reuse the tracked PbTe examples. Restart metadata is compared
exactly. Restart atom coordinates, masses, and velocities use an absolute
tolerance of `1e-10` for text serialization and velocity-unit roundoff.

Tests run in isolated temporary directories and normally delete them afterwards.
Set `REPLICA_TEST_WORKDIR` to retain test directories, including outputs and
the last `run.in` and `run.log` for each case:

```bash
REPLICA_TEST_WORKDIR=/tmp/replica-checks \
  python3 tests_regression/replica/test_replica.py
```

A missing executable, unavailable GPU, process timeout (120 seconds per run),
or failed assertion fails an explicit invocation. General test discovery skips
this GPU suite unless `GPUMD_REPLICA` is set. The suite does not modify CI and
does not cover multiple GPUs, long-time sampling accuracy, or performance.
