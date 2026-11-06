# GPUMD `rerun` V1 final acceptance report

Date: 2026-11-06 (Asia/Shanghai)

## Verdict

**READY TO MERGE**, subject to normal code review. All requested final acceptance
checks passed on the patched `feature/rerun` implementation
`e5f1b4f6dc8072a29aaa13946755d07a6fd9b383`. No changes were made to `master`
and nothing was pushed.

The tested fix rejects `dump_xyz` paths resolving to the input trajectory,
including relative, absolute, parent-component, and symlink aliases. It is in
the focused commit listed above. The clean source checkout was verified at
that commit before compiling; the pre-fix and patched builds were separate.

## Environment and source

- Host: H100 node `gpu-10`, NVIDIA H100 PCIe, CUDA capability 9.0 (`sm_90`).
- CUDA: 12.8.61; GNU C++: 11.4.0.
- PLUMED: official PLUMED 2.10.1 at
  `/data/home/user/soft/Lammps+deepmd+plumed_newversion/installs/plumed-2.10.1-official`.
- Source checkout: `/data/home/user/soft/GPUMD-rerun-latest`.
- Branch and source commit before final acceptance builds: `feature/rerun`,
  `e5f1b4f6dc8072a29aaa13946755d07a6fd9b383`.
- GPU tests ran from
  `/data/home/user/PG/MD/GPUMD/950K/solid_left/Rerun_test/latest_e5f1b4f/`.
- Reproducible executable artifacts are retained in that run directory's
  `build_artifacts/` folder, outside the source checkout.
- The base under-test commit is
  `6f548c78baaad2f80b8d1bcbf8ccc88301515727`; the only production change is
  the path-alias guard in the e5 commit.

## Compilation and CPU reader

Fresh CUDA build, after checking out e5 and confirming `git rev-parse HEAD`:

```bash
cmake -S /data/home/user/soft/GPUMD-rerun-latest \
  -B /data/home/user/soft/GPUMD-rerun-latest/build-rerun-v1-e5f1b4f-sm90 \
  -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release
cmake --build /data/home/user/soft/GPUMD-rerun-latest/build-rerun-v1-e5f1b4f-sm90 \
  --target gpumd nep -j 12
```

Status: **PASS**, Slurm job 128196. The separately generated PLUMED executable
was compiled from a clean worktree at the same e5 commit using the repository's
documented Makefile configuration:

```bash
cd /data/home/user/soft/GPUMD-rerun-latest-plumed-e5f1b4f-src/src
make gpumd -j12 CUDA_ARCH=-arch=sm_90 \
  CFLAGS="-std=c++14 -O3 -arch=sm_90 -DUSE_PLUMED" \
  INC="-I./ -I/data/home/user/soft/Lammps+deepmd+plumed_newversion/installs/plumed-2.10.1-official/include" \
  LDFLAGS="-L/data/home/user/soft/Lammps+deepmd+plumed_newversion/installs/plumed-2.10.1-official/lib -lplumed -lplumedKernel"
```

Status: **PASS**, Slurm job 128197; `ldd` resolved both PLUMED libraries.
The CPU reader regression was compiled and run from the e5 checkout:

```bash
g++ -std=c++14 -Isrc/main_gpumd \
  tests_regression/rerun_reader_test.cpp -o /tmp/gpumd_rerun_reader_test
/tmp/gpumd_rerun_reader_test
```

Status: **PASS**, exit 0.

## Final acceptance test matrix

| Check | Status | Result |
|---|---|---|
| Patched commit identity before build | PASS | Exact e5 SHA above; fresh sm_90 and PLUMED builds. |
| CUDA `gpumd` and `nep` build | PASS | CUDA 12.8.61, H100 `sm_90`, job 128196. |
| PLUMED-enabled build | PASS | PLUMED 2.10.1, job 128197; both runtime libraries resolve. |
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

## Existing `tests_pytest` suite

The full suite was run from the patched source checkout with the fresh e5
`gpumd` and `nep` executables symlinked temporarily at the paths selected by
`tests_pytest/conftest.py`. A shared isolated Python environment was used on
the GPU node (Python 3.11.15, pytest 9.2.0, calorine 4.1, hypothesis 6.170.14,
ASE 3.29.0, NumPy 2.4.4). The source checkout was at `b7bc708` with the tested
binary source commit `e5f1b4f` as an ancestor; the executable SHA-256 was
`376f50241d1c4051f5f6e0355562305b5e657afe4aeaf99c5ec2561f3c7840fe`.

Exact command, run from `tests_pytest/`:

```bash
/data/home/user/soft/gpumd-pytest-e5/bin/python -m pytest -q \
  --junitxml=/data/home/user/PG/MD/GPUMD/950K/solid_left/Rerun_test/latest_e5f1b4f/pytest_final.junit.xml
```

Slurm job 128229 completed successfully in 22 minutes 28 seconds (pytest
reported 22 minutes 20.83 seconds): **748 passed, 0 failed, 482 skipped,
0 xfailed** (1,230 collected). JUnit independently reports 1,230 cases,
0 failures, 0 errors, and 482 skipped. Since no test failed, there were no
failure cases requiring parent-build attribution. The pytest log and JUnit
XML are retained at the paths above.

The preinstalled pytest found in the local `pfd` environment was Python 3.10
and lacked calorine and hypothesis; H100's existing conda environments had no
pytest. The run therefore used an isolated, shared-filesystem Python 3.11
environment with only the missing test packages installed. A first attempt
used node-local `/tmp` and did not start the suite; job 128229 used the shared
environment and is the reported run.

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

Final acceptance job IDs: PLUMED 128210, selection/scheduling 128211,
negative/path/lifecycle 128213, normal-run comparison 128216. Earlier e5
energy and temperature jobs 128205 and 128206 also completed successfully.
The complete `tests_pytest` suite passed in job 128229.
Outputs and raw logs are in the corresponding `energy_smoke`,
`temperature_final`, `plumed_final`, `selection`, `final_acceptance`, and
`normal_run_compare` subdirectories.

## Fix and commit state

- Production fix: `e5f1b4f6dc8072a29aaa13946755d07a6fd9b383` —
  `Reject rerun dump paths that alias input`.
- This report and reproducible final acceptance drivers are committed on
  `feature/rerun` in the validation commit immediately following the fix
  commit; the final validation commit hash is recorded by repository HEAD.
- Working tree is clean after the validation commit.
- Nothing was pushed; `master` was not checked out or modified.

## Recommendation

**READY TO MERGE.** The patched commit passes the requested build, regression,
numerical, PLUMED, trajectory selection/scheduling, negative-input,
resource-lifecycle, path collision, normal-MD, and existing `tests_pytest`
checks. Review the focused
commits on `feature/rerun`; the branch has not been merged or pushed.
