# GPUMD development instructions

These instructions apply to contributors and coding agents throughout the
repository. Contributors are responsible for understanding every change they
submit, including AI-assisted work, and validating the results it affects.

## Working on a change

Propose changes you need in your own work with GPUMD. Before starting work,
agree on the change with the maintainers as described in the
[contribution policy](README.md#support-and-contributions). If another
contributor leads the affected module, raise the issue first so that developer
can take the lead on a fix or performance improvement.

- Keep the PR branch up to date with `master` throughout development and review.
- Read the relevant implementation, nearby code, tests, and user documentation
  before editing. Find an existing implementation with similar responsibilities.
- Reuse existing interfaces, parsers, lifecycle hooks, GPU utilities, and neighbor
  lists when their contracts fit the new functionality. Check cutoff, periodicity,
  atom ordering, and list lifetime before reusing a neighbor list.
- Keep each change focused. Avoid unrelated refactoring, renaming, formatting,
  new dependencies, or speculative abstraction.
- Implement independent features in their own modules. Limit changes to shared
  code to what registration and integration require. Explain why each shared
  change is needed, which files and behaviors it affects, and how it was tested.
  Coordinate broader refactoring with the maintainers in a separate prerequisite
  PR. Once it merges, update the feature branch to use the shared implementation.
- Extend an existing method in its owning module and keep its name and command
  interface. Add an option only for a choice that existing keywords cannot
  express. Reuse shared logic and preserve behavior outside the agreed scope.
- Preserve local edits, reference executables, and datasets. Put generated
  inputs, logs, benchmarks, and temporary files in separate working directories.

## Build

Run from the repository root. Set `CUDA_ARCH` for the target GPU and installed
CUDA toolkit. This example builds for compute capability 8.9.

```sh
make -C src -j4 CUDA_ARCH=-arch=sm_89
```

For HIP, set `HIP_ARCH` to match the AMD GPU. This example targets `gfx1100`.

```sh
make -C src -f makefile.hip -j4 HIP_ARCH=gfx1100
```

The default Make targets produce `gpumd`, `nep`, and `tb` in `src/`.
The two Makefiles share object files. Clean the build before switching backends
in the same tree or changing compiler flags, toolkits, or architectures.
When adding source files or changing shared code, check the applicable targets
in [src/makefile](src/makefile), [src/makefile.hip](src/makefile.hip), and
[CMakeLists.txt](CMakeLists.txt). Reconfigure an existing CMake build after
adding sources so it picks up the new files. Keep new code compatible with the
existing compiler settings.

## Architecture and extension points

| Location | Responsibility |
| --- | --- |
| `src/main_gpumd/` | Entry point, `Run`, input dispatch, and top-level simulation flow |
| `src/model/` | Atoms, simulation box, groups, and related data |
| `src/force/` | `Force`, MD `Potential` implementations, and neighbor management |
| `src/integrate/` | `Integrate`, `Ensemble` implementations, constraints, and motion control |
| `src/measure/` | `Measure`, run-time `Action` implementations, measurements, and dumps |
| `src/minimize/` | Energy minimization and structural relaxation |
| `src/phonon/` | Hessian and phonon calculations |
| `src/mc/` | Monte Carlo moves for hybrid Monte Carlo and molecular dynamics |
| `src/utilities/` | Shared GPU wrappers, containers, parsing, and error helpers |
| `src/main_nep/` | NEP training and prediction |
| `src/main_tb/` | Tight-binding model training and prediction |
| `doc/` | User documentation |

For a standard MD run, `RunInput` tokenizes commands and `Run` dispatches them.
Implement parsing and behavior in the module that owns the functionality.
Put new potential, ensemble, or action state in its corresponding class.
Reserve direct additions to `Run` for operations that control overall execution.

The main integration and measurement hooks run in the order shown below.
See [`Run::perform_a_run()`](src/main_gpumd/run.cu) for the full sequence.

```text
Integrate::initialize -> Measure::pre_run
Run::compute_force -> Measure::setup_force
for each step:
    Integrate::compute1 -> Measure::post_integrate1
    Measure::pre_force -> Run::compute_force -> Measure::post_force
    Integrate::compute2 -> Measure::end_of_step
Measure::post_run -> Integrate::finalize
```

Use the matching lifecycle hook. Check initialization, repeated `run` commands,
cleanup, and restart when adding stateful behavior. Extend existing output and
restart readers and writers where applicable, rather than maintaining parallel
implementations. Save and restore all state needed for continuation, including
algorithm counters and random-number state when required by the method.

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
register construction in `potential.cu`. Reuse common parsers for shared
keywords. Parse model-specific keywords in the owning model and reject
unsupported keywords.

## Code style and GPU work

- Follow neighboring code. Use clang-format 19 with the repository's
  [.clang-format](.clang-format) configuration. For every PR, format all added
  or modified C++/CUDA source and header files in full, and add contributor
  information as appropriate. Do not reformat unrelated files. Preserve existing
  copyright and license notices and use the project header for new source files.
- Name the original implementers in the header of each new source file.
  When a change adds major functionality, fixes a significant correctness issue,
  substantially improves performance, or substantially refactors an existing
  file, update its header in the same change to credit the contributors
  responsible. Describe each contribution briefly and specifically.
- Use the existing input utilities to check argument counts, types, ranges, and
  incompatible options. Do not introduce C++ exceptions for error handling.
  Report fatal errors and exit through the appropriate project helper in
  [src/utilities/error.cuh](src/utilities/error.cuh), such as `PRINT_INPUT_ERROR`
  for invalid input or `PRINT_SCANF_ERROR` for a failed formatted read.
- When reading structured numerical files, check that each required value was
  read successfully. For malformed or truncated input, identify the input file
  and the affected field or record in the diagnostic.
- Keep mutable state in the owning object and make members private by default.
  Keep helper types and functions used only by one class private to that class
  where practical. Keep file-local kernels and helpers `static` or in an
  anonymous namespace. Do not wrap internal implementation in a named namespace.
  Export only interfaces with actual cross-file users.
- Keep implementation-only includes in the `.cu` file. Use a forward
  declaration in the header when the declaration needs only the type name.
- Follow existing ownership conventions, including `std::unique_ptr` for
  owned polymorphic objects and `GPU_Vector` for persistent device arrays.
- Let the calculation or model own its GPU workspace. Reuse allocated storage
  and avoid unnecessary host-device transfers and synchronization in MD and
  training loops.
- Use `src/utilities/gpu_macro.cuh` for shared CUDA/HIP code and localize
  unavoidable backend-specific implementations.
- Remove temporary debugging output, obsolete transition code, and hidden
  switches that disable, zero, or change the sign of physical contributions.
  Put reproducible diagnostic checks in tests.
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

Test behavior primarily through the executables and their public inputs and
outputs. Test public model-file interfaces directly where appropriate.
Do not expose internal state or add `friend` declarations or accessors solely
for tests.

Build baseline and candidate executables with matching toolchains, backends,
architectures, and flags. Follow each suite's reproducibility requirements,
including `-DDEBUG` for NEP differential training tests. Preserve the baseline
and its matching source tree. Do not regenerate reference outputs from the
candidate implementation or relax tolerances just to make a failure pass.

When changing `nep` evaluation code that supports runtime specialization,
compare native and specialized kernels on Linux with CUDA using identical
parameters and descriptor scalers. Confirm that the specialized kernels
compiled and loaded. Keep the matching source tree available. If needed,
use `GPUMD_SRC` to select its `src/` directory and `CUDACXX` to select the
CUDA compiler.
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
Document extensions to an existing method in its existing manual page. Give
independent functionality a dedicated page, with only necessary additions to
shared pages. Keep model-specific restrictions local to that model's
documentation and avoid duplicating common explanations.

Keep general development rules here and detailed test procedures with their
tests. Keep user documentation focused on how to use the software.

Before submitting, review the full PR diff and check it for whitespace errors.
For example, with `origin/master` up to date as the target branch, run
`git diff --check origin/master...HEAD`.
Follow [.github/pull_request_template.md](.github/pull_request_template.md):
explain the problem, resulting behavior, relevant validation, and remaining
limitations. Keep development logs and intermediate proposals out of the PR
description and permanent documentation.
