# GPUMD `rerun` V1 test log

Date: 2026-11-06 (Asia/Shanghai)

## Test target

- Branch: `feature/rerun`.
- Base implementation tested: `6f548c78baaad2f80b8d1bcbf8ccc88301515727`.
- Production fix: `e5f1b4f6dc8072a29aaa13946755d07a6fd9b383`
  (`Reject rerun dump paths that alias input`).
- Current validation commit: `6de063f` (sanitized test paths; no code changes
  after the tested fix).
- GPU: NVIDIA H100 PCIe, compute capability 9.0; CUDA 12.8.61; GNU 11.4.0.
- PLUMED: official 2.10.1.
- Test artifacts: the dedicated `latest_e5f1b4f` run directory under the
  configured `$RERUN_TEST_ROOT`; binaries were archived outside the source
  checkout. The patched `gpumd` SHA-256 was
  `376f50241d1c4051f5f6e0355562305b5e657afe4aeaf99c5ec2561f3c7840fe`.

## Build and reader tests

| Test | Status | Result |
|---|---|---|
| CUDA build (`gpumd`, `nep`), architecture `sm_90` | PASS | Slurm job 128196; fresh build at the e5 source commit. |
| PLUMED-enabled CUDA build, architecture `sm_90` | PASS | Slurm job 128197; PLUMED 2.10.1 libraries resolved with `ldd`. |
| CPU Extended XYZ reader regression | PASS | `g++ -std=c++14 -Isrc/main_gpumd tests_regression/rerun_reader_test.cpp -o /tmp/gpumd_rerun_reader_test && /tmp/gpumd_rerun_reader_test`; exit 0. |

## Numerical and model behavior

| Test | Status | Result |
|---|---|---|
| Cu EAM, 32 atoms, 100 frames; repeated execution | PASS | Energy max Δ 0 eV; force RMSE/max 0 eV/Å; full 9-element Virial max/relative Δ 0; position max Δ `8.88e-16 Å`; box max Δ 0. Repeated outputs matched. |
| Ordinary C NEP, 159 atoms, no ensemble | PASS | Energy, force RMSE/max, and Virial max/relative differences 0; position max Δ `8.88e-16 Å`; box max Δ 0. |
| Temperature-dependent F NEP, 512 atoms, constant 950 K ensemble | PASS | Energy, forces, and Virial matched at reported precision; position max Δ `3.55e-15 Å`; box max Δ 0; 100 thermo rows. |
| Temperature-dependent NEP without ensemble | PASS | Rejected with a clear requirement for positive configured simulation temperature. |
| Temperature ramp for temperature-dependent NEP | PASS | Rejected with a clear constant-temperature requirement. |
| Normal two-step NVE run, pre-fix vs patched executable | PASS | Extended XYZ and thermo output files were byte-identical. |

Numerical thresholds were energy max Δ `1e-5 eV`, force RMSE `<1e-5 eV/Å`,
and Virial max Δ `1e-5 eV`; observed differences met these thresholds without
relaxation. Force and Virial comparisons used physical output before PLUMED
bias.

## PLUMED, selection, and output scheduling

| Test | Status | Result |
|---|---|---|
| Structural `DISTANCE ATOMS=1,2`, interval 1 | PASS | 100 CV rows; max difference from independent minimum-image distance `4.3147198e-7 nm`; time max Δ `3.47e-18 ps`; 100 XYZ frames and 100 thermo rows. |
| Same structural CV, interval 10 | PASS | 10 CV rows; same max CV difference; XYZ and thermo each independently emitted 100 rows. |
| Sparse trajectory (original Steps 20, 100) | PASS | 2 ordered CV rows, 2 XYZ frames, 2 thermo rows; same max CV difference. |
| `first 20` | PASS | Selected Steps 20–100. |
| `last 50` | PASS | Selected Steps 1–50. |
| `every 30` | PASS | Selected Steps 1, 30, 60, 90. |
| `skip 3` | PASS | Selected Steps 1, 4, …, 100. |
| Combined filters | PASS | `first 20 last 90 every 30 skip 2` selected Steps 20, 60. |
| Empty selected range | PASS | No frames emitted. |
| Output intervals | PASS | XYZ interval 20 selected 20,40,60,80,100; thermo interval 25 emitted 4 rows. Sparse Steps 20,100,200,300,400,500 emitted XYZ at 100…500 and thermo at 200,400. |
| Repeated/decreasing Step and Step 0 | PASS | Repeated/decreasing Steps selected 1,2,3; Step 0 was retained. |
| Consecutive rerun lifecycle | PASS | Ten rerun commands in one process produced ten outputs and exited normally; two sequential reruns also passed. |

## Negative and input/output safety tests

| Input / condition | Status | Observed result |
|---|---|---|
| Malformed XYZ | PASS | Rejected, exit 1, missing-lattice diagnostic. |
| Invalid timestep metadata (`Time=NaN`) | PASS | Rejected, exit 1, `invalid Time`. |
| Atom-count mismatch | PASS | Rejected, exit 1, atom count differs from `model.xyz`. |
| Invalid box | PASS | Rejected, exit 1, lattice entry-count diagnostic. |
| Missing velocity with `dump_thermo` | PASS | Rejected, exit 1, requires `vel:R:3` in selected XYZ frames. |
| Exact input/output path collision | PASS | Rejected before writing; input SHA-256 unchanged. |
| Relative aliases `./trajectory.xyz` and `sub/../trajectory.xyz` | PASS | Both rejected; input SHA-256 unchanged. |
| Absolute path alias | PASS | Rejected; input SHA-256 unchanged. |
| Symlink alias | PASS | Rejected; input SHA-256 unchanged. |

## Existing `tests_pytest` suite

The full suite was run against the archived executable built from the e5 source
commit. The source checkout also contained the validation-only follow-up; its
code tree did not change after the executable build. Python environment:
Python 3.11.15, pytest 9.2.0, calorine 4.1, hypothesis 6.170.14, ASE 3.29.0,
NumPy 2.4.4.

Command run from `tests_pytest/`:

```bash
python -m pytest -q --junitxml="$RERUN_TEST_ROOT/pytest_final.junit.xml"
```

The command used an isolated shared-filesystem environment on the H100 compute
node, with `src/gpumd` and `src/nep` temporarily resolving to the e5 binaries.
Slurm job 128229 completed in 22m28s. **748 passed, 0 failed, 482 skipped,
0 xfailed** (1,230 collected). JUnit reported 0 failures and 0 errors. No test
failed, so no parent-build failure attribution was needed.

## Overall result

**PASS — READY TO MERGE.** CUDA and PLUMED builds, reader regression,
numerical/model tests, PLUMED CV tests, trajectory selection and output
scheduling, negative cases, input collision protection, lifecycle checks,
normal MD comparison, and the full existing pytest suite passed. The tested
stateful PLUMED actions MetaD/OPES were outside this test set. No changes were
made to `master`; nothing was pushed or merged.
