# NEP partition tests on one GPU

Opt-in release checks for ordinary NEP versus the existing `-DZHEYONG` path:
one physical GPU, one versus three NEP partitions. Python 3.9+ standard library
only; Linux with CUDA or HIP. No pytest or NumPy is needed.

## Run

Build both executables from the same sources with matching backend,
architecture and optimization options, differing only in `-DZHEYONG`:

| Executable | Compile options |
|---|---|
| `src/gpumd` | `-DDEBUG` |
| `src_multigpu/gpumd` | `-DDEBUG -DZHEYONG` |

`src_multigpu` can be a copy of `src`. Keep its sources in sync when applying
patches, and clean/rebuild after changing compile flags. The runner prints
both executable paths and SHA256 hashes, then verifies the expected NEP path
from each process log. It does not build or modify either executable.

From the repository root, run correctness followed by performance:

```sh
python3 tests_regression/multigpu/run_tests.py
```

Run either part separately:

```sh
python3 tests_regression/multigpu/run_tests.py --mode correctness
python3 tests_regression/multigpu/run_tests.py --mode performance
```

For AMD, add `--backend hip`. Use `--device 0` to explicitly select one device;
the default is the first inherited CUDA/HIP visible device, or device 0.
Optional `--single` and `--zheyong` override the executable paths.
`--timeout` sets the per-process limit in seconds (default 900).

All inputs, outputs and logs live in a temporary directory, removed on success,
failure or interruption. Nothing is written to the repository. Results appear
only on screen: per-case correctness, numerical failure details or the last
25 process-log lines on execution failure, and the performance table.

## Correctness

Each row shows independent checks of the ordinary and partitioned paths,
followed by their numerical comparison. A complete pass is **20/20**: 18 cases
and one repeatability check for each path. Missing output, non-finite values,
wrong partition counts, errors or skipped comparisons cannot pass.

| Cases | Coverage |
|---|---|
| `axis_x`, `axis_y`, `axis_z`, `auto` | Explicit/automatic partition direction; shuffled atom order |
| `triclinic`, `partial_pbc` | Skewed box and nonperiodic y |
| `alloy` | Cu/Ni/Al element mapping |
| `nonuniform`, `empty`, `migration` | Uneven/empty partitions and observed slab/PBC crossings |
| `nvt`, `npt` | Berendsen temperature/pressure coupling |
| `multi_run`, `append`, `restart` | Output lifecycle, valid orthogonal box change, shared real restart |
| `water` | Short liquid-water trajectory |
| `energy_solid`, `energy_water` | 100-step NVE drift checks |

Ordinary cases run 10 steps; multiple-run, append and repeatability controls
use at most 20 steps per path. Energy checks run 100 steps (25 fs for solid,
10 fs for water); they do not establish long-time energy conservation.

Checks include atom identity/order, XYZ metadata, frame counts and times,
selected groups, restart contents, energy/virial sums and kinetic energy.
Positions, velocities, forces, per-atom energies, virials and thermodynamics
are compared with fixed bounds in `checks.py` (`TOLERANCES`):
per-atom potential `atol=5e-5` eV, force
`atol=5e-4` eV/angstrom with `rtol=2e-5`, for example. Failures show maximum
and RMS error, tolerances and the number of values outside the bounds.

The FCC Cu input has 9,216 atoms, except intentionally sparse cases. Water
has 6,048 atoms, replicated from the existing
[63-atom liquid-water MD snapshot](../../tests_pytest/fixtures/structures/water-nat63-from-md.xyz).
Fixed velocities, shuffled IDs and tiny unique mass offsets make these
numerical test fixtures. No melting/equilibration stage is run; the water
snapshot is not claimed to be equilibrated under NEP89.

The potential is referenced directly from
[NEP89](../../potentials/nep/nep89_20250409/nep89_20250409.txt).
No duplicate potential or water data is included.

## Performance

Only two large workloads are measured:

| System | Atoms | NVE time step |
|---|---:|---:|
| Cu solid | 1,105,920 | 0.5 fs |
| Liquid water | 1,102,248 | 0.25 fs |

Each path runs three fresh processes per workload, alternating execution order.
Each process has 10 warmup steps and **100 measured steps**. The last measured
step writes one thermo row, synchronizing GPU work before GPUMD stops its
timer; this small output cost is included in both rates. No trajectory is
written. Throughput comes from the measured run and is checked against
`atoms * 100 / elapsed`.

The screen table shows both arithmetic mean throughputs in **atom-step/s**,
the ratio **ZHEYONG / single**, and both sample CVs in percent. There is no
speedup pass/fail threshold. `Performance: COMPLETE` requires every sample
to finish and pass the timing/finite-output checks. If a later workload fails,
the table still shows any fully completed workload; partial groups have no mean.

In `all` mode, correctness must pass before performance starts. Performance-only
mode runs a small preflight pair and explicitly reports full correctness as
`NOT_RUN`. These are single-device partition tests; they do not measure physical
multi-GPU scaling, communication or efficiency. The package remains separate
from the 255-case regression suite and is not required for every PR.

## Test the Python tools

```sh
python3 -B tests_regression/multigpu/test_tools.py
```

These CPU checks cover parsing, independent numerical checks, controller
failures/cleanup and performance arithmetic. They do not validate GPU physics.
