# GPUMD Developer Guide

This document gives a short overview of the current GPUMD code structure and the main extension points. It focuses on the `gpumd` and `nep` executables.

## 1. Building

GPUMD uses GNU Make.

For NVIDIA GPUs:

```bash
cd src
make
```

The default build produces:

```text
gpumd
nep
```

The CUDA architecture can be changed when needed, for example:

```bash
make CUDA_ARCH="-arch=sm_89"
```

For AMD GPUs:

```bash
cd src
make -f makefile.hip
```

The makefiles are the reference for compiler flags, GPU architecture settings, and linked libraries.

## 2. Source structure

The main source directories are:

| Directory | Purpose |
| --- | --- |
| `src/main_gpumd/` | `gpumd` entry point, input dispatch, and top-level simulation flow |
| `src/model/` | Atoms, simulation box, groups, and related data structures |
| `src/force/` | Potential models and force evaluation |
| `src/integrate/` | Ensembles, integration, constraints, and motion control |
| `src/measure/` | Run-time actions, measurements, dumps, and related operations |
| `src/minimize/` | Energy minimization |
| `src/phonon/` | Phonon calculations |
| `src/mc/` | Monte Carlo functionality |
| `src/utilities/` | Shared utilities, GPU wrappers, input helpers, and containers |
| `src/main_nep/` | `nep` executable |

Files added under an existing source directory are normally collected automatically by the makefiles.

## 3. Main `gpumd` architecture

The top-level execution is controlled by `Run` in `src/main_gpumd/`.

The main objects are:

```text
Run
├── Atom
├── Velocity
├── Box
├── Group[]
├── Force
├── Integrate
└── Measure
```

Their roles are deliberately separated:

- `Run` controls the overall simulation flow and dispatches top-level input commands.
- `Force` owns the active `Potential` objects and evaluates forces, energies, and virials.
- `Integrate` owns the active `Ensemble` and controls integration-related operations.
- `Measure` owns run-time `Action` objects.

`run.in` is first parsed by `RunInput`. Each keyword is then sent to the module that owns the corresponding functionality.

In simplified form:

```text
run.in
  |
  v
Run
  +--> Force
  +--> Integrate
  +--> Measure
  +--> top-level operations handled by Run
```

A new keyword should normally be implemented in the module responsible for its behavior rather than directly in `Run`.

## 4. MD run lifecycle

The main MD loop follows this structure:

```text
Integrate::initialize()
Measure::pre_run()
Force::compute()
Measure::setup_force()

for each step:
    Integrate::compute1()
    Measure::post_integrate1()

    Measure::pre_force()
    Force::compute()
    Measure::post_force()

    Integrate::compute2()
    Measure::end_of_step()

Measure::post_run()
Integrate::finalize()
```

This lifecycle determines where new run-time functionality should be placed.

## 5. Main extension points

### Potential

Potential models derive from `Potential` and are managed by `Force`.

A new potential should keep model-specific data and reusable GPU workspaces inside the potential object and implement the required `compute()` interface. Expensive setup, allocation, or data conversion should not be repeated inside every force evaluation unless necessary.

### Ensemble or integration method

Ensembles derive from `Ensemble` and are managed by `Integrate`.

The main integration interfaces are `compute1()` and `compute2()`. Run-level initialization and cleanup can be implemented through the corresponding ensemble lifecycle functions.

Algorithm-specific state should remain inside the derived ensemble class.

### Run-time action

Many commands used during MD are implemented as classes derived from `Action` and managed by `Measure`.

Available hooks include:

```text
pre_run
setup_force
post_integrate1
pre_force
post_force
end_of_step
post_run
```

Use the lifecycle hook that matches the intended operation instead of adding special cases to the main MD loop.

### Top-level operation

Functionality belongs directly in `Run` only when it changes the overall execution flow or performs a system-level operation, such as starting a run, minimization, or major reconfiguration of the simulated system.

## 6. GPU code

`GPU_Vector` is the standard container for many persistent device arrays.

For performance-sensitive code:

- reuse persistent GPU memory when possible;
- avoid unnecessary host-device transfers and synchronization;
- avoid repeated allocation inside MD steps;
- use the GPU abstraction layer in `src/utilities/gpu_macro.cuh` for code shared by CUDA and HIP;
- keep backend-specific code localized when a common implementation is not practical.

The main ownership pattern for polymorphic objects uses `std::unique_ptr`, for example for potentials, ensembles, and actions.

## 7. Input and errors

Input lines are tokenized by `RunInput` before being passed to individual modules.

When adding a keyword:

- validate the number and type of arguments before use;
- use the existing parsing and error utilities;
- keep parsing in the module that owns the keyword;
- keep the implementation synchronized with the user manual.

Common error utilities are defined in `src/utilities/error.cuh`.

## 8. The `nep` executable

The `nep` executable is implemented mainly under `src/main_nep/` and shares utilities with GPUMD.

Its top-level structure is compact:

```text
Parameters
    |
    v
Fitness
    |
    v
SNES
```

`Parameters` handles the input configuration, `Fitness` contains the model/data evaluation logic, and `SNES` performs optimization.

Changes to `nep.in`, model formats, or training data formats should remain synchronized with the NEP manual.

## 9. Units and documentation

The basic internal units are:

- energy: eV
- length: Å
- mass: Dalton
- temperature: K
- charge: elementary charge

User-facing changes to keywords, file formats, units, or physical definitions should be reflected in the documentation under `doc/`.
