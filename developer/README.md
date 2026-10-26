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

Files added under an existing source directory are normally collected automatically by the makefiles. The CUDA and HIP makefiles also collect `.cu` files one directory below `src/force/` for `gpumd`, and `src/main_nep/` for `nep`.

Spin inference, multi-GPU evaluation, external magnetic fields, and Spin-specific
small-box kernels live in `src/force/nep_spin/`. Training in
`src/main_nep/nep_spin/` includes the same magnetic descriptor and force headers;
ordinary NEP keeps its own small-box kernels in `src/force/nep_small_box.cuh`.

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
- keep backend-specific code localized when a common implementation is not practical;
- keep kernels and file-local helpers `static` or in an anonymous namespace; expose only the wrappers needed by other translation units.

The main ownership pattern for polymorphic objects uses `std::unique_ptr`, for example for potentials, ensembles, and actions.

## 7. Input and errors

For `gpumd`, input lines are tokenized by `RunInput` before being passed to individual modules. The `nep` input path is described below.

When adding a keyword:

- validate the number and type of arguments before use;
- use the existing parsing and error utilities;
- keep parsing in the module that owns the keyword;
- keep the implementation synchronized with the user manual.

Common error utilities are defined in `src/utilities/error.cuh`.

## 8. The `nep` executable

The `nep` executable is implemented under `src/main_nep/` and shares utilities with GPUMD. Its training `Potential` interface is separate from the MD `Potential` interface in `src/force/`.

### Source layout

| Directory under `src/main_nep/` | Purpose |
| --- | --- |
| Top level | Entry point, model selection, optimizer, scheduling, and shared data structures |
| `common/` | Reusable NEP parameter, data, loss, I/O, and runtime-compilation helpers |
| `nep/` | Ordinary and temperature-dependent NEP |
| `nep_spin/` | Spin3 parameters, magnetic labels/neighbors, response supervision, and model I/O |
| `nep_charge/` | NEP charge modes 1 and 2 |
| `nep_vdw/` | NEP with long-range dispersion |
| `nep_charge_vdw/` | NEP with charge and dispersion |
| `tnep/` | Dipole and polarizability models |

Helpers in `common/` may serve a subset of models, such as the two charge models. Model-independent data and logic can remain shared.

### Construction and responsibilities

`ParameterInput` reads `nep.in` and identifies the model selectors. `create_potential()` in `potential.cu` constructs the selected model, whose parameter class parses and validates the remaining settings. The entry point then uses:

```cpp
std::unique_ptr<Potential> potential = create_potential();
Parameters& para = potential->get_parameters();
Fitness fitness(*potential);
```

| Component | Responsibility |
| --- | --- |
| `Parameters` | Common optimizer and scheduling settings; model-specific settings belong to the model's parameter class |
| `Potential` implementation | Parameter layout and initialization, model and restart I/O, data preparation, evaluation, physical losses, reporting, and prediction |
| `Fitness` | Batch scheduling, common L1/L2 regularization, and delegation through `Potential` |
| `SNES` | Population generation and optimization; obtains parameter groups, initialization, and restart output through `Potential` |

The model owns its parameter object. `Fitness` and `SNES` use references to the model and should not branch on its concrete type.

Construction configures the model. The `Fitness` constructor calls `prepare_data()` to build datasets and initialize model workspaces. At zero-based generation 0, training calls `prepare_training()`, which prepares descriptor scaling for the existing NEP models; subsequent generations call `evaluate()`. Prediction reads the model through `read_model()` and calls `predict()`, without initializing an SNES population.

### Data and losses

Shared structure and dataset code handles common data. Model-specific labels, GPU arrays, validation, and losses belong to the corresponding model or a helper shared by the models that need them. Existing models use `NEP_Data_Reader` hooks and `NEP_Batch_Data<T>` for model-owned batch/device data.

`evaluate()` receives contiguous candidate parameter vectors on the host. The model owns intermediate losses; `accumulate_fitness()` adds their contributions after common regularization. Keep model-specific reduction and transfer decisions behind these interfaces, so later GPU reduction or population parallelism does not require model branches in `Fitness` or `SNES`.

Spin3 keeps `Structure` and `Dataset` unchanged and stores magnetic labels and device
buffers in `Spin_Data`, indexed by `NEP_Batch_Data`. Labels are matched through
`Structure::index_in_file` after batch sorting or energy-difference grouping.
Its reader supplies the largest structural/magnetic cutoff to the shared geometry
reader. `response.xyz` has separate device data and contributes only response loss;
it does not fit the energy reference or scalers. Spin3 uses the shared
`energy_reference` and block model/restart formats while preserving its
fixed-reference scaler and the common generation-zero lifecycle. Legacy Spin3
model and two-column restart files remain readable.

### Adding a training model

1. Implement the `Potential` interface in the model's subdirectory, with its parameter parsing, labels, workspaces, losses, and output handling.
2. Add any new model selector to `parameter_input.cu/.cuh` and register construction in `potential.cu`. Ordinary model-specific keywords should be parsed by the selected model; unsupported keywords must be rejected. For example, `lambda_q` is accepted only by charge models.
3. Reuse suitable common helpers. New physical quantities should normally be implemented in the model's own reader, data, loss, and output code. Extend shared structures only for common data or behavior.
4. Cover initialization, restart, training, prediction, and supported device configurations with regression cases. Keep input and output documentation synchronized with the implementation.

The existing internal modes remain within their classes: ordinary/temperature NEP, dipole/polarizability TNEP, and charge modes 1/2 do not need separate model classes.

### Runtime specialization

Each model owns its specialized entry points and launcher bindings. Shared compilation and reusable kernels live in `common/`. Model-specific `*_specialized.cu` files use `NEP_SPECIALIZED_RUNTIME_BUILD`; device kernels remain local, while the required launcher wrappers are exported.

On Linux/CUDA builds, `nep_compile on` requests runtime specialization for supported training configurations. Standalone prediction does not use this training JIT path. Keep the matching source tree available; `GPUMD_SRC` can select its `src` directory and `CUDACXX` can select the compiler. Compilation or loading failures report a warning and fall back to the generic path.

When changing evaluation code, compare `nep_compile on` and `off` with identical parameters and descriptor scalers, and confirm that the on run actually used the specialized path after successful compilation and loading. Use fixed seeds for training regressions; small floating-point differences can change population ranking and subsequent optimization trajectories.

## 9. Units and documentation

The basic internal units are:

- energy: eV
- length: Å
- mass: Dalton
- temperature: K
- charge: elementary charge

User-facing changes to keywords, file formats, units, or physical definitions should be reflected in the documentation under `doc/`.

Spin moments are in muB and magnetic forces in eV/muB. See the
[model input units](../doc/gpumd/input_files/model_xyz.rst) and
[external magnetic field](../doc/gpumd/input_parameters/add_mfield.rst)
documentation for the input and Zeeman conventions.

See [Spin regression checks](spin_validation.md) for portable format, training,
and GPU validation commands.

Spin lattice coupling and restart handling live in `src/integrate/spin/`.
`Spin_Lattice_NHC` and `Spin_Lattice_MTTK` reuse the upstream mechanical
integrators and add endpoint persistence and resumed temperature/pressure
protocols only for Spin ensembles. Ordinary NHC/MTTK classes remain unchanged.
The XYZ reader and restart writer delegate Spin metadata to this directory.
