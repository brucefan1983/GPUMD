# NEP multi-GPU release tests

This is an opt-in test package for the existing single-node NEP force
implementation. It runs the public `gpumd` executable with one or several
visible GPUs. A release maintainer runs it before a release; ordinary PRs do
not need multi-GPU hardware. It is separate from the 255-case GPUMD/NEP
regression suite and the TB and pytest suites.

Requirements: Python 3.9 or newer (standard library only), Linux, and CUDA or
HIP. Physical multi-GPU testing needs at least two GPUs. The `--zheyong` mode
checks three logical partitions on one physical GPU using the existing
`-DZHEYONG` build option. No pytest, NumPy, MPI, or downloaded dataset is needed.
The suite references the repository model
[`NEP89`](../../potentials/nep/nep89_20250409/nep89_20250409.txt)
through `--repo-root` (the repository containing this script by default).
It does not copy the potential into the test package or modify source code.
Both executables must support this model's current file format.

## Three-partition correctness on one GPU

Supply an ordinary executable with `--new` and an executable compiled with
`-DZHEYONG` with `--zheyong`. Both must come from the same source version and
use the same backend, architecture and compiler flags apart from that macro.
The runner exposes the same one physical GPU to both processes. It checks
that both logs report one visible GPU and that only the macro build enters
the three-partition NEP path.

This mode runs correctness tests only. It rejects `--old`, `--counts`,
`--mode performance`, and `--mode all`. `--devices` defaults to `0` and must
select exactly one device. Both executables receive identical inputs sized
for three partitions: 9,216 atoms for ordinary cases, with sparse cases
containing fewer atoms. Liquid preparation and the shared restart use the
ordinary executable. Results check the partitioning, middle-partition path,
atom order and numerical outputs; the report labels physical multi-GPU
validation and performance as unverified. It does not compute a simulated
multi-GPU speedup or efficiency.

### Build without changing the working source tree

These commands copy the current `src` into fresh build directories below
`reports/`. They preserve local source edits and leave `src/`, `src_old/`,
their executables, and their Makefiles untouched. Start from ordinary compiler
flags without `-DZHEYONG`. Clean the copies so old objects cannot retain
different compiler options. Build only the `gpumd` target.

From the repository root:

```sh
set -eu
mkdir -p tests_regression/multigpu/reports
build_dir=$(mktemp -d "$PWD/tests_regression/multigpu/reports/zheyong-build.XXXXXX")
cp -a src "$build_dir/normal"
cp -a src "$build_dir/zheyong"
```

Choose one backend. For CUDA, adjust the example architecture to your GPU:

```sh
backend=cuda
make -C "$build_dir/normal" -f makefile clean
make -C "$build_dir/zheyong" -f makefile clean
make -C "$build_dir/normal" -f makefile -j4 gpumd CUDA_ARCH=-arch=sm_89
make -C "$build_dir/zheyong" -f makefile -f - -j4 gpumd CUDA_ARCH=-arch=sm_89 <<'MAKE'
override CFLAGS += -DZHEYONG
MAKE
```

For HIP, adjust `HIP_ARCH` as needed:

```sh
backend=hip
make -C "$build_dir/normal" -f makefile.hip clean
make -C "$build_dir/zheyong" -f makefile.hip clean
make -C "$build_dir/normal" -f makefile.hip -j4 gpumd HIP_ARCH=gfx1100
make -C "$build_dir/zheyong" -f makefile.hip -f - -j4 gpumd HIP_ARCH=gfx1100 <<'MAKE'
override CFLAGS += -DZHEYONG
MAKE
```

The second Makefile is read from standard input after the existing Makefile;
it appends the macro while preserving the normal flags, including `-DUSE_HIP`.
Run the comparison in the same shell:

```sh
python3 tests_regression/multigpu/run_tests.py \
  --new "$build_dir/normal/gpumd" --zheyong "$build_dir/zheyong/gpumd" \
  --backend "$backend" --devices 0 \
  --output "$build_dir/results"
```

An existing ordinary executable built from the same source and configuration
can be reused as `--new`; in that case only the isolated macro build is
needed. Keep the two executables under distinct paths. The macro build always
uses three logical partitions, even when only one physical GPU is visible.
All three workspaces occupy that GPU's memory. Physical multi-GPU correctness,
peer access, device switching, synchronization and performance still require
the tests below when hardware is available.

## Correctness on two AMD GPUs

From the repository root:

```sh
python3 tests_regression/multigpu/run_tests.py \
  --new src/gpumd --backend hip --devices 0,1 \
  --output tests_regression/multigpu/reports/release-correctness
```

This checks the candidate against itself on one and two GPUs. To also compare
an accepted executable, add `--old src_old/gpumd`:

```sh
python3 tests_regression/multigpu/run_tests.py \
  --old src_old/gpumd --new src/gpumd --backend hip --devices 0,1 \
  --output tests_regression/multigpu/reports/old-new-correctness
```

Use `--backend cuda` for NVIDIA. Build old and new with the same backend,
compiler, architecture, feature and optimization flags. Record these with
`--build-note "..."`. In particular, a CUDA baseline cannot be used on an
AMD-only host. The runner performs short preflight runs before preparing the
liquid or starting the suite, and stops if a loader, device, or execution
check fails. An unavailable GPU or missing library is an error, not a skip.

`--devices` explicitly selects the device order. One-GPU runs use the first
device; N-GPU runs use the first N. CUDA and HIP visibility aliases are set
consistently for each subprocess. An inherited `ROCR_VISIBLE_DEVICES` mask
is retained: HIP indices are then relative to that restricted device set.
Run in the intended scheduler allocation, with no other conflicting device
filters. Effective masks and GPUMD's GPU names and peer-access messages are
retained in the report and process logs.

The default counts are 1 and the number of selected devices. For example,
on four GPUs use `--devices 0,1,2,3 --counts 1,2,4`. Up to 16 selected GPUs
are accepted, matching the current NEP implementation. Geometry grows with
the largest count; larger configurations also require more memory.

## Coverage and acceptance

| Cases | Purpose |
|---|---|
| `axis_x`, `axis_y`, `axis_z`, `auto` | Explicit and automatic spatial partitioning; shuffled input IDs |
| `triclinic` | A valid skewed, fully periodic cell |
| `alloy` | Cu/Ni/Al types in deliberately shuffled atom order |
| `nonuniform`, `empty` | Uneven populations and GPU domains with no owned atoms |
| `partial_pbc` | Nonperiodic y, partition along periodic x |
| `migration` | Rigid drift across slab and periodic boundaries; require observed crossings |
| `nvt`, `npt` | Short deterministic Berendsen temperature/pressure coupling |
| `multi_run` | Output reset between runs, a run with no output, and valid orthogonal `change_box` |
| `append` | Two executions in the same directory; correct appended frame/row counts |
| `restart` | Load a real preflight restart shared by every comparison |
| `liquid` | Short dynamics from a common equilibrated Cu liquid snapshot |
| `energy_solid`, `energy_liquid` | NVE total-energy drift over 500 fs |

The ordinary correctness inputs contain 6,144 atoms for two physical GPUs
and 9,216 atoms in `--zheyong` mode, except the intentionally sparse cases.
The x/y/z cases have the same size in different orientations. NEP89's 6 A
radial cutoff sets the geometry: at least ten bins of width cutoff/2 per
logical partition along the partition direction, with a margin for skew and
pressure coupling. The empty-domain case is a diagnostic stress test, not a
performance workload.

The Cu solid starts from an FCC lattice with small deterministic displacements
and velocities. Masses carry tiny unique offsets (1e-8 Da per input index) to
check atom identity independently; these are numerical fixtures, not material
property benchmarks. The multi-element test is likewise a short numerical
comparison, not an equilibrium alloy study.

Liquid preparation uses one accepted GPU executable (`--old` when supplied,
otherwise `--new`): 5 ps at 3000 K, cooling to 2000 K over 2 ps, then 3 ps at
2000 K. Every comparison uses the same resulting restart. The preparation
must show increasing center-of-mass-corrected unwrapped MSD above 2 A² over
the final 2 ps and a final mean temperature between 1800 and 2200 K. The report
retains these diagnostics. This is evidence of diffusion, not a full phase
characterization; a failed preparation gate stops the suite.

Each short run checks finite fields, property schemas, atom identities and
original order, frame counts and times, group subsets, restart contents,
potential/virial sums and kinetic energy. For Berendsen NVT/NPT, the kinetic
check accounts for thermo being sampled before the final velocity rescaling.
Short single-/multi-GPU outputs compare positions, velocities, forces,
per-atom energies and all nine virial components without sorting the output.
The first solid case repeats both GPU paths to measure reproducibility.
With `--old`, every case additionally compares new/old at matching GPU counts
and old multi-/single-GPU results. Long NVE runs use an independent energy-drift
bound and an early total-energy comparison rather than chaotic trajectory
equality. Restart checks do not require bitwise continuation of a trajectory.

For a full correctness result, run all 18 cases (the default) and require
`Correctness: PASS`, with no failed, erroneous or skipped checks. `--cases
axis_x,alloy` is a diagnostic subset and is labeled as such. `--list` lists
cases without launching GPUMD. A pass in `--zheyong` mode applies to the
single-GPU partition tests; it does not satisfy the physical multi-GPU release
validation. This package covers ordinary NEP89 only; it
does not establish multi-GPU support for qNEP, spin, temperature-dependent
NEP, replica dynamics, or every ensemble and measurement.

## Numerical bounds and rechecking

[`tolerances.json`](tolerances.json) contains explicit `[atol, rtol]` pairs.
The comparison is `abs(candidate-reference) <= atol + rtol*abs(reference)`.
Units are Å, Å/fs, eV/Å, eV and GPa as appropriate. Thermo energies are
normalized per atom. Energy-drift bounds are eV/atom measured from the first
sample (2.5 fs). Initial absolute bounds include 5e-5 eV for per-atom energy,
5e-4 eV/Å for force, and 2e-3 eV for each virial component. These are initial
engineering bounds, **not NEP89 limits calibrated on every CUDA/HIP device**.
The historical carbon-model comparison does not validate these NEP89 bounds.

On the first hardware run, inspect repeatability controls and raw maximum/RMS
errors, then decide whether the bounds are suitable. Do not automatically
relax them to turn failures into passes. Structural, identity and finite-value
checks cannot be disabled with the tolerance file. Independent aggregation
checks allow only summation and output-rounding error.

Preserve original reports and evaluate revised bounds without rerunning MD:

```sh
python3 tests_regression/multigpu/recheck_results.py \
  tests_regression/multigpu/reports/old-new-correctness \
  --tolerances tests_regression/multigpu/tolerances.json
```

Without `--tolerances`, the original report's bounds are used. A new
`recheck-<timestamp>/` is created, retaining the original report. Inputs,
outputs and logs are verified against their recorded hashes. Keep the model
reference available and unchanged for rechecking. No executable or GPU is
required to recheck existing output. Execution errors and missing comparisons
remain errors, even when numerical bounds are changed.

## Performance

Performance testing uses physical GPUs and requires ordinary builds without
`-DZHEYONG`.

After correctness passes, measure the original implementation using solid
and liquid states, three sizes and interleaved repetitions:

```sh
python3 tests_regression/multigpu/run_tests.py \
  --old src_old/gpumd --new src/gpumd --backend hip --devices 0,1 \
  --mode performance --profiles minimal,output \
  --output tests_regression/multigpu/reports/old-new-performance
```

`--old` is optional. `--mode all` performs correctness first and starts
performance only after it passes. A performance-only invocation explicitly
reports correctness as `NOT_RUN`; it does not inherit a previous pass.

The default is 100 warmup steps followed by 500 measured NVE steps at 0.5 fs,
with three fresh process repetitions. On two GPUs, small/medium/large sizes
are 24,576 / 110,592 / 393,216 atoms, generated by replicating the common
solid or equilibrated liquid input. The same size is used at every GPU count
in a scaling comparison. Each repetition shuffles build/GPU-count order with
a fixed recorded seed. Avoid concurrent jobs on the selected GPUs.

`minimal` has no measured trajectory or thermo output. `output` writes thermo
about five times and one position frame during the measured run. Its timing
therefore includes that I/O. An additional one-step run after measurement
checks finite thermodynamics. The reported rate comes from GPUMD's second
`run`, excluding model loading, input replication, warmup and the final probe.
Process wall times are retained separately. Liquid preparation is also excluded.

Performance reports use the arithmetic mean of each repetition's atom-step/s;
new/old and scaling ratios divide those means. Efficiency is scaling divided
by GPU count. CV is the sample standard deviation divided by the mean,
**expressed as a percentage**, and is undefined with only one repetition.
Incomplete or failed groups do not receive a successful aggregate rate.
There is no automatic speedup pass/fail threshold. Inspect variance and
individual rates before drawing conclusions; increase `--steps` and
`--repeats` when runs are too short or unstable.

Use `--sizes small`, `--phases solid`, `--profiles minimal`, `--steps`,
`--warmup`, and `--repeats` for focused benchmarks. The per-process timeout
is 7200 seconds by default; increase `--timeout` for slower or larger runs.
Runtime and retained disk usage depend on hardware and selected profiles.

## Reports and package self-tests

Every invocation requires a fresh output directory. Nothing removes old
results. `summary.md` gives status and comparisons; `report.json` includes
process order, effective environment, binary/model/input/output hashes,
individual metrics and timings; `performance.csv` contains aggregate rates.
The scripts used for the run are copied into `code/` for provenance. All
inputs and logs remain in their individual run directories. Generated reports
are ignored by Git. Return code is nonzero for execution, validation or
comparison failures, including timeouts. Prerequisite failures are explicit;
they cannot silently become passing comparisons.

Run the standard-library CPU checks separately:

```sh
python3 -m unittest discover -s tests_regression/multigpu -p 'test_*.py' -v
```

These validate readers, corrupted/truncated output handling, identity checks,
device selection, failure propagation, result rechecking and performance
arithmetic. Small temporary executable emulators exercise the controller;
they are not GPU or NEP physics tests. Real CUDA/HIP validation remains a
release-machine task.
