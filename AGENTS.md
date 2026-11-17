# GPUMD development instructions

These instructions apply to contributors and coding agents throughout this
repository. Contributors are responsible for understanding their changes and
checking the numerical results, including code produced with AI assistance.

## Working on a change

- Read the relevant implementation, nearby code, tests, and user documentation
  before editing. Find an existing implementation with similar responsibilities.
- Reuse existing interfaces, parsers, lifecycle hooks, GPU utilities, and neighbor
  lists when their contracts fit the new functionality. Check cutoff, periodicity,
  atom ordering, and list lifetime before reusing a neighbor list.
- Keep each change focused. Avoid unrelated refactoring, renaming, formatting,
  new dependencies, or speculative abstraction.
- Preserve existing behavior outside the requested change. Extend an existing
  method through its established interface when possible, keeping shared logic
  in one place.
- Preserve local edits, reference executables, and datasets. Put generated
  inputs, logs, benchmarks, and temporary files in separate working directories.

## Build

Run from the repository root. For CUDA, select an architecture supported by
the target GPU and installed toolkit; `sm_89` is an example:

```sh
make -C src -j4 CUDA_ARCH=-arch=sm_89
```

For HIP:

```sh
make -C src -f makefile.hip -j4
```

The default Make targets produce `gpumd`, `nep`, and `tb` in `src/`.
Read [src/makefile](src/makefile) and [src/makefile.hip](src/makefile.hip) for
source discovery, compiler flags, optional targets, and linked libraries.
Check both build paths when adding source files or changing shared code.
Clean the affected build when compiler options, toolkits, or architectures
change. Keep new code compatible with the existing compiler settings; do not
raise the required language standard as an incidental change.

## Architecture and extension points

| Location | Responsibility |
| --- | --- |
| `src/main_gpumd/` | Entry point, `Run`, input dispatch, and top-level simulation flow |
| `src/model/` | Atoms, simulation box, groups, and related data |
| `src/force/` | `Force`, MD `Potential` implementations, and neighbor management |
| `src/integrate/` | `Integrate`, `Ensemble` implementations, constraints, and motion control |
| `src/measure/` | `Measure`, run-time `Action` implementations, measurements, and dumps |
| `src/minimize/`, `src/phonon/`, `src/mc/` | Minimization, phonons, and Monte Carlo |
| `src/utilities/` | Shared GPU wrappers, containers, parsing, and error helpers |
| `src/main_nep/` | NEP training and prediction |
| `src/main_tb/` | Tight-binding training |
| `doc/` | User documentation |

For MD, `RunInput` tokenizes commands and `Run` dispatches them. Implement
parsing and behavior in the module that owns the functionality. Put new
potential, ensemble, or action state in its corresponding class. Reserve
direct additions to `Run` for operations that control overall execution.

The main MD lifecycle is:

```text
Integrate::initialize -> Measure::pre_run
Force::compute -> Measure::setup_force
for each step:
    Integrate::compute1 -> Measure::post_integrate1
    Measure::pre_force -> Force::compute -> Measure::post_force
    Integrate::compute2 -> Measure::end_of_step
Measure::post_run -> Integrate::finalize
```

Use the matching lifecycle hook. Check initialization, repeated `run` commands,
cleanup, and restart when adding stateful behavior.

The training `Potential` in `src/main_nep/` is separate from the MD `Potential`.
`ParameterInput` identifies model selectors and `create_potential()` constructs
the model. The model owns its parameters, labels, workspaces, physical losses,
model/restart I/O, reporting, and prediction. `Fitness` owns batch scheduling
and common regularization; `SNES` owns population generation and optimization.
Keep model-specific behavior behind the training interface instead of adding
concrete-model branches to `Fitness` or `SNES`.

Training data are prepared through `prepare_data()`, descriptor scaling through
`prepare_training()`, and candidate evaluation through `evaluate()` and
`accumulate_fitness()`. Prediction uses `read_model()` and `predict()`.
Reuse suitable helpers in `src/main_nep/common/`; keep model-specific data in
the corresponding model directory. Add selectors in `parameter_input` and
register construction in `potential.cu`; parse ordinary model keywords within
the selected model and reject unsupported keywords.

## Code style and GPU work

- Follow neighboring code and [.clang-format](.clang-format). Format changed
  code without reformatting unrelated files. Preserve existing copyright and
  license notices and use the project header for new source files.
- Use existing input and error utilities, including
  `src/utilities/error.cuh`. Validate argument counts, types, ranges, and
  incompatible options before use; avoid introducing exception-based handling.
- Keep mutable state in the owning object. Keep file-local kernels and helpers
  `static` or in an anonymous namespace; export only required interfaces.
- Follow existing ownership conventions, including `std::unique_ptr` for
  owned polymorphic objects and `GPU_Vector` for persistent device arrays.
- Reuse allocated device workspaces. Avoid unnecessary allocation, host-device
  transfers, and synchronization in MD and training loops.
- Use `src/utilities/gpu_macro.cuh` for shared CUDA/HIP code and localize
  unavoidable backend-specific implementations.
- Preserve the established precision, units, virial conventions, normalization,
  and accumulation semantics unless changing them is part of the task.
  Basic units include eV, angstrom, Dalton, K, and elementary charge; magnetic
  moments and magnetic forces use muB and eV/muB.

## Validation

Choose focused checks for the changed behavior, then complete the applicable
regression suite. Documentation-only edits need path, link, and patch checks;
they do not require rebuilding executables.

| Area | Instructions |
| --- | --- |
| GPUMD/NEP differential regression | [tests_regression/README.md](tests_regression/README.md) |
| TB differential regression | [tests_regression/tb/README.md](tests_regression/tb/README.md) |
| Pytest integration and numerical checks | [tests_pytest/README.md](tests_pytest/README.md) |
| Spin model, training, runtime, and integration checks | [tests_pytest/SPIN_VALIDATION.md](tests_pytest/SPIN_VALIDATION.md) |

For source changes covered by GPUMD/NEP regression, follow its `full` acceptance
contract; a focused subset does not replace the full run. The TB suite runs
separately. Add or update a focused test for a new feature or bug fix, using
physical expectations, an independent formula, or an accepted baseline.

Build baseline and candidate executables with matching toolchains, backends,
architectures, and flags. Follow each suite's reproducibility requirements,
including `-DDEBUG` for NEP differential training tests. Preserve the baseline
and its matching source tree. Do not regenerate reference outputs from the
candidate implementation or relax tolerances just to make a failure pass.

When changing NEP evaluation code, compare native and runtime-specialized
paths using identical parameters and descriptor scalers. Confirm that requested
specialization actually compiled and loaded; a generic fallback is not a
successful specialization check. Keep the matching source tree available
(`GPUMD_SRC` can select `src/`, and `CUDACXX` can select the compiler).
Use fixed seeds for training comparisons and inspect physical predictions as
well as losses: small floating-point changes can alter population ranking.

Record the commands, build configuration, and results. Distinguish passing,
failing, skipped, and unrun checks. If suitable hardware or dependencies are
unavailable, report that limitation explicitly. Support performance claims with
comparable workloads and numerical checks.

## Documentation and pull requests

Update `doc/` when changing keywords, defaults, units, physical definitions,
file formats, output, or restart behavior. Reuse existing formats and manual
sections where appropriate; check readers, writers, and examples together.

Keep general development rules here and detailed test procedures with their
tests. Keep user documentation focused on how to use the software.

Before submitting, review the complete diff and run `git diff --check`.
Follow [.github/pull_request_template.md](.github/pull_request_template.md):
explain the problem, resulting behavior, relevant validation, and remaining
limitations. Keep development logs and intermediate proposals out of the PR
description and permanent documentation.
