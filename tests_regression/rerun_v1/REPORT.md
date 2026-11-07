# GPUMD `rerun` V1 validation report

Date: 2026-11-07 (Asia/Shanghai)

## Current review follow-up

An independent review identified that the previous input-collision guard did
not cover dynamically generated `dump_xyz` filenames (`*` and `split N`). The
failure was reproduced on the prior executable before the fix: per-frame
output was accepted and changed the trajectory bytes. The new source-level
guard and H100 regression results are recorded in the follow-up section at the
end of this report. The historical V1 results below describe the earlier e5
acceptance run.

The tested fix rejects `dump_xyz` paths resolving to the input trajectory,
including relative, absolute, parent-component, and symlink aliases. It is in
the focused commit listed above. The clean source checkout was verified at
that commit before compiling; the pre-fix and patched builds were separate.

## Historical V1 environment and source (e5 acceptance)

All machine-specific paths below are generic placeholders. Substitute paths
from the local build and test environment when reproducing commands.

- Host: H100 compute node, NVIDIA H100 PCIe, CUDA capability 9.0 (`sm_90`).
- CUDA: 12.8.61; GNU C++: 11.4.0.
- PLUMED: official PLUMED 2.10.1 at
  `/path/to/PLUMED_PREFIX`.
- Source checkout: `/path/to/GPUMD_SOURCE`.
- Branch and source commit before final acceptance builds: `feature/rerun`,
  `e5f1b4f6dc8072a29aaa13946755d07a6fd9b383`.
- GPU tests ran from
  `/path/to/RERUN_TEST_ROOT/latest_e5f1b4f/`.
- Reproducible executable artifacts are retained in that run directory's
  `build_artifacts/` folder, outside the source checkout.
- The base under-test commit is
  `6f548c78baaad2f80b8d1bcbf8ccc88301515727`; the only production change is
  the path-alias guard in the e5 commit.

## Compilation and CPU reader

Fresh CUDA build, after checking out e5 and confirming `git rev-parse HEAD`:

```bash
cmake -S /path/to/GPUMD_SOURCE \
  -B /path/to/GPUMD_SOURCE/build-rerun-v1-e5f1b4f-sm90 \
  -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release
cmake --build /path/to/GPUMD_SOURCE/build-rerun-v1-e5f1b4f-sm90 \
  --target gpumd nep -j 12
```

Status: **PASS**. The separately generated PLUMED executable
was compiled from a clean worktree at the same e5 commit using the repository's
documented Makefile configuration:

```bash
cd /path/to/PLUMED_BUILD_SOURCE/src
make gpumd -j12 CUDA_ARCH=-arch=sm_90 \
  CFLAGS="-std=c++14 -O3 -arch=sm_90 -DUSE_PLUMED" \
  INC="-I./ -I/path/to/PLUMED_PREFIX/include" \
  LDFLAGS="-L/path/to/PLUMED_PREFIX/lib -lplumed -lplumedKernel"
```

Status: **PASS**; `ldd` resolved both PLUMED libraries.
The CPU reader regression was compiled and run from the e5 checkout:

```bash
g++ -std=c++14 -Isrc/main_gpumd \
  tests_regression/rerun_reader_test.cpp -o $TMPDIR/gpumd_rerun_reader_test
$TMPDIR/gpumd_rerun_reader_test
```

Status: **PASS**, exit 0.

## Historical e5 acceptance test matrix

| Check | Status | Result |
|---|---|---|
| Patched commit identity before build | PASS | Exact e5 SHA above; fresh sm_90 and PLUMED builds. |
| CUDA `gpumd` and `nep` build | PASS | CUDA 12.8.61, H100 `sm_90`. |
| PLUMED-enabled build | PASS | PLUMED 2.10.1; both runtime libraries resolve. |
| CPU reader regression | PASS | Command above, exit 0. |
| PLUMED structural CV, interval 1 | PASS | 100 CV rows; maximum independent distance error `4.3147198e-7 nm`; time error `3.47e-18 ps`; 100 `dump_xyz` frames and 100 thermo rows. |
| PLUMED structural CV, interval 10 | PASS | 10 CV rows at expected output cadence; same maximum CV error; 100 XYZ and thermo outputs, independently scheduled. |
| PLUMED sparse input | PASS | Input original steps 20 and 100; 2 ordered CV rows, 2 XYZ frames, 2 thermo rows; same maximum CV error. |
| `first`, `last`, `every`, `skip` individually | PASS | `first 20` selects steps 20–100; `last 50` selects 1–50; `every 30` gives 1,30,60,90; `skip 3` gives 1,4,…,100. |
| Selection combinations and empty range | PASS | `first 20 last 90 every 30 skip 2` gives 20,60; empty range gives no frames. |
| Output intervals and sparse steps | PASS | XYZ interval 20 => 20,40,60,80,100; thermo interval 25 => 4 rows. Sparse steps 20,100,200,300,400,500 schedule XYZ at 100…500 and thermo at 200,400. |
| Repeated/decreasing steps and step 0 | PASS | Existing semantics preserved; repeated/decreasing selection gives 1,2,3; step 0 is retained. |
| Malformed XYZ | PASS | Rejected, exit 1, clear missing-lattice diagnostic. |
| Invalid timestep metadata | PASS | `Time=NaN` rejected, exit 1, `invalid Time` diagnostic. |
| Atom count mismatch | PASS | Rejected, exit 1, `rerun atom count differs from model.xyz`. |
| Invalid box | PASS | Rejected, exit 1, invalid lattice entry count diagnostic. |
| Missing velocity with `dump_thermo` | PASS | Rejected, exit 1, `rerun output requires vel:R:3 in every selected XYZ frame`. |
| Consecutive reruns / cleanup | PASS | Ten sequential rerun commands in one process; all ten output files written and process exited normally. Two-command output cadence also passed in selection matrix. |
| Exact/relative/absolute path collision | PASS | Exact path, `./trajectory.xyz`, `sub/../trajectory.xyz`, absolute path, and symlink alias all rejected; input SHA-256 unchanged. |
| Normal `run` behavior | PASS | Two-step NVE output from pre-fix 6f and patched e5 executables byte-identical for both Extended XYZ and thermo output. |
| Working tree / branch | PASS | Final test/report commit leaves the patched source checkout clean on `feature/rerun`; no `master` edits and no push. |

PLUMED numeric CV checks used `DISTANCE ATOMS=1,2` and an independent
minimum-image Cartesian distance calculated from each input configuration.
The output unit conversion to nm was included. COLVAR row counts, original
frame order, original timestep-derived physical time, and the independent
`dump_xyz`/`dump_thermo` schedules were checked. This validates the tested
stateless structural action; MetaD/OPES stateful behavior is outside this
acceptance set.

## Historical e5 `tests_pytest` suite

The full suite was run from the patched source checkout with the fresh e5
`gpumd` and `nep` executables symlinked temporarily at the paths selected by
`tests_pytest/conftest.py`. A shared isolated Python environment was used on
the GPU node (Python 3.11.15, pytest 9.2.0, calorine 4.1, hypothesis 6.170.14,
ASE 3.29.0, NumPy 2.4.4). The source checkout was at `b7bc708` with the tested
binary source commit `e5f1b4f` as an ancestor; the executable SHA-256 was
`376f50241d1c4051f5f6e0355562305b5e657afe4aeaf99c5ec2561f3c7840fe`.

Exact command, run from `tests_pytest/`:

```bash
/path/to/PYTEST_ENV/bin/python -m pytest -q \
  --junitxml=/path/to/RERUN_TEST_ROOT/latest_e5f1b4f/pytest_final.junit.xml
```

The suite completed successfully in 22 minutes 28 seconds (pytest
reported 22 minutes 20.83 seconds): **748 passed, 0 failed, 482 skipped,
0 xfailed** (1,230 collected). JUnit independently reports 1,230 cases,
0 failures, 0 errors, and 482 skipped. Since no test failed, there were no
failure cases requiring parent-build attribution. The pytest log and JUnit
XML are retained at the paths above.

The preinstalled pytest found in the local `pfd` environment was Python 3.10
and lacked calorine and hypothesis; H100's existing conda environments had no
pytest. The run therefore used an isolated, shared-filesystem Python 3.11
environment with only the missing test packages installed. A first attempt
used node-local temporary storage and did not start the suite; the reported
run used the shared environment.

## Previously completed numerical and temperature checks

These e5 runtime comparisons were completed before the final acceptance pass
and are retained in the same H100 test root:

| Case | Status | Maximum differences |
|---|---|---|
| Cu EAM, 32 atoms, 100 frames; repeated rerun | PASS | Energy 0 eV; force RMSE/max 0 eV/Å; full 9-element virial max/relative 0; position max `8.88e-16 Å`; box 0 Å. Repeated outputs match. |
| Ordinary C NEP, 159 atoms, 100 frames, no ensemble | PASS | Energy 0 eV; force RMSE/max 0 eV/Å; virial max/relative 0; position max `8.88e-16 Å`; box 0 Å. |
| Temperature-dependent F NEP, 512 atoms, fixed 950 K ensemble | PASS | Energy, all forces, all virial elements agree exactly at reported precision; position max `3.55e-15 Å`; box 0 Å; 100 thermo rows. |
| Temperature-dependent NEP, no ensemble | PASS | Clear nonzero exit requiring a positive configured ensemble temperature. |
| Temperature-dependent NEP, 300-to-950 K ramp | PASS | Clear nonzero exit requiring a constant-temperature ensemble. |

Numerical comparisons use the unmodified physical force/virial output before
any PLUMED bias. Energy tolerance was `1e-5 eV`, force RMSE target
`1e-5 eV/Å`, and Virial max difference target `1e-5 eV`; all were met without
relaxing tolerances.

## Reproduction commands and artifacts

The submitted runtime scripts and logs are retained under the H100 test root.
Key commands, from that directory, were:

```bash
sbatch latest_e5_energy_smoke.slurm
sbatch latest_e5_temperature_tests.slurm
sbatch latest_e5_plumed_tests.slurm
sbatch latest_e5_selection_tests.slurm
sbatch final_acceptance.slurm
sbatch normal_run_compare.slurm
sbatch run_tests_pytest_final.slurm
```

PLUMED, selection/scheduling, negative/path/lifecycle, normal-run comparison,
energy, temperature, and full `tests_pytest` runs all completed successfully.
Outputs and raw logs are in the corresponding `energy_smoke`,
`temperature_final`, `plumed_final`, `selection`, `final_acceptance`, and
`normal_run_compare` subdirectories.

## Historical e5 acceptance commit state

- Production fix: `e5f1b4f6dc8072a29aaa13946755d07a6fd9b383` —
  `Reject rerun dump paths that alias input`.
- This report and reproducible final acceptance drivers are committed on
  `feature/rerun` in the validation commit immediately following the fix
  commit; the final validation commit hash is recorded by repository HEAD.
- Working tree is clean after the validation commit.
- At that time, nothing was pushed and `master` was not checked out or modified.

## Historical e5 recommendation

**READY TO MERGE.** This was the historical e5 recommendation only; the
current review-follow-up status is recorded below.

## P1 independent-review follow-up

### Root cause and patch

Before the patch, Dump_XYZ::rerun_overwrites_input() returned false for
per-frame (*) and split N modes. The H100 executable accepted a per-frame
output whose generated filename was the input trajectory, returned success,
and changed the trajectory bytes. Per-frame outputs use "w" and therefore
truncated the file. Split output uses numbered filenames and skips existing
paths, but lacked a final collision check immediately before opening a
generated file.

The minimal fix passes the trajectory path into the existing dump_xyz Measure
action during pre_rerun(). Normal output is checked just before its open;
per-frame and split modes check the actual generated filename before opening.
Canonical path comparison covers relative, absolute, parent-component, and
symlink aliases; filesystem identity comparison also covers hard links.
Existing filename rules are unchanged, and the guard is inactive during
normal MD run.

Production fix:  b52f9bb6c37cdc52112d2a81eca6b4108b5667b9.
The collision regression was expanded in 380f6a3f4f5fbe3005ac114b0650661942a7f2ef
and its PLUMED structural-CV mode in
58ea5c30614f257fcaee261a45caecbfaf9a0e5f.

### Current source and builds

- Tested production source: 380f6a3f4f5fbe3005ac114b0650661942a7f2ef
  (the following test-driver commit changed tests only).
- Device: NVIDIA H100 PCIe; CUDA 12.8.61; target sm_90.
- PLUMED: official 2.10.1.
- The source checkout was clean at the tested commit before both fresh builds.
  CUDA and PLUMED outputs were built in new build directories/worktrees.

Commands (environment-specific paths replaced by placeholders):

~~~~sh
cmake -S /path/to/GPUMD_SOURCE -B /path/to/TEST_ROOT/build-cmake-p1 \
  -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release
cmake --build /path/to/TEST_ROOT/build-cmake-p1 --target gpumd nep -j12

cd /path/to/TEST_ROOT/plumed-src/src
make gpumd -j12 CUDA_ARCH=-arch=sm_90 \
  CFLAGS="-std=c++14 -O3 -arch=sm_90 -DUSE_PLUMED" \
  INC="-I./ -I/path/to/PLUMED_PREFIX/include" \
  LDFLAGS="-L/path/to/PLUMED_PREFIX/lib -lplumed -lplumedKernel"

g++ -std=c++14 -Isrc/main_gpumd tests_regression/rerun_reader_test.cpp \
  -o /tmp/gpumd_rerun_reader_test_p1
/tmp/gpumd_rerun_reader_test_p1
~~~~

CUDA and PLUMED builds: **PASS**. CPU reader regression: **PASS**.
For PLUMED runtime, PLUMED_KERNEL and LD_LIBRARY_PATH were set to the
installed PLUMED 2.10.1 library directory.

### Collision regression matrix

The pre-fix reproduction used the previous e5 executable and failed as
expected: per-frame collision returned 0 and the input SHA-256 changed. The
post-fix driver was run on H100 with both fresh executables:

| Case | Status | Result |
|---|---|---|
| Normal direct output collision | PASS | Rejected; input hash unchanged. |
| Normal relative path with .. | PASS | Rejected; input hash unchanged. |
| Normal symlink and absolute aliases | PASS | Rejected; input hash unchanged. |
| Per-frame direct * collision | PASS | Rejected before opening; input hash unchanged. |
| Per-frame symlink/.., absolute symlink, hard link | PASS | All rejected; input hash unchanged. |
| Split output where _0001 is the input | PASS | Existing chunk skipped; input unchanged and _0002 written. |
| Split output where _0001 symlinks to input | PASS | Aliased chunk skipped safely; input unchanged. |
| Successful normal, per-frame, and split output | PASS | Expected output file created in every mode. |
| PLUMED structural distance with successful reruns | PASS | One COLVAR row per one-frame replay. |
| Normal GPUMD NVE run | PASS | Baseline and patched executables produced byte-identical two-step XYZ and thermo output. |

Reproduction commands, run from the H100 test environment:

~~~~sh
python3 tests_regression/rerun_v1/rerun_dump_collision_test.py \
  --gpumd /path/to/TEST_ROOT/build-cmake-p1/gpumd \
  --trajectory /path/to/cu_trajectory.xyz \
  --model tests_regression/fixtures/systems/cu_grouped.xyz \
  --potential potentials/eam/Cu_Zhou_2004.txt

python3 tests_regression/rerun_v1/rerun_dump_collision_test.py \
  --gpumd /path/to/TEST_ROOT/build-plumed-p1/gpumd \
  --trajectory /path/to/cu_trajectory.xyz \
  --model tests_regression/fixtures/systems/cu_grouped.xyz \
  --potential potentials/eam/Cu_Zhou_2004.txt --with-plumed
~~~~

### Full tests_pytest rerun

The suite ran on H100 against the fresh sm_90 executable from the tested
production source commit. It used an isolated Python 3.11 environment with
pytest 9.2.0, calorine 4.1, hypothesis 6.170.14, ASE 3.29.0, and NumPy 2.4.4.
The executable and NEP utility were temporarily linked into a separate clean
worktree at the paths used by tests_pytest/conftest.py. Command, from
tests_pytest/:

~~~~sh
python -m pytest -q --junitxml=/path/to/TEST_ROOT/pytest-p1.junit.xml
~~~~

Result: **748 passed, 0 failed, 482 skipped, 0 xfailed** (1,230 collected),
completed in 11 minutes 51.10 seconds. JUnit reports 1,230 tests, 0 failures,
0 errors, and 482 skipped.

### Current disposition

The vulnerability is fixed. Fresh CUDA and PLUMED builds, CPU reader,
H100 collision tests, PLUMED structural-CV runtime checks, normal-MD byte
comparison, and full pytest all pass. No changes were made to master. The
follow-up commits are intended for the existing feature/rerun PR.

**Current recommendation: READY TO MERGE.**
