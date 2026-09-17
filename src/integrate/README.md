# Adding an integrator

The integration code follows one `ensemble`--`run` lifetime. Commands before a
`run` configure that run; exactly one `ensemble` must be present. `Integrate`
owns the selected `Ensemble` with a `std::unique_ptr`, initializes it when the
run starts, and destroys it when the run finishes. An ensemble must not carry
state into a later `run`.

## Class responsibilities

Put a new integrator in its own `ensemble_<name>.cuh` and
`ensemble_<name>.cu` files and derive it from `Ensemble`.

- Keep keyword-specific parsing and validation in the concrete class, normally
  in its constructor or a private parsing helper. Preserve precise diagnostics
  for invalid input.
- Keep `Integrate::parse_ensemble` as the single keyword-selection and object-
  construction point. It should only select the class and copy metadata needed
  by the run driver, such as the ensemble type or temperature endpoints.
- Store only configuration and algorithm-owned state in the object. Pass
  ordinary run objects such as `Atom`, `Box`, `Group`, and `thermo` explicitly
  by reference; do not cache pointers or references to them as members.
- Use RAII containers for dynamically allocated C++ storage; do not introduce
  owning `new`/`delete` raw pointers. If a C handle such as `FILE*` cannot use
  RAII, close it in normal finalization and keep destruction as a fallback.

Add an `EnsembleType` value only when the run driver or another shared
component must distinguish the new method. Do not use the enum as a substitute
for a virtual operation that belongs to the concrete class.

## Run lifecycle

The hooks have distinct timing:

1. `initialize_run(...)` runs once before the initial force calculation. Use it
   for run-owned allocation and state that does not require the initial force.
2. `initialize_before_first_step(...)` runs once from the first `compute1`,
   after the initial force and first-step velocity/time-step adjustments. Use it
   only when initialization needs that state.
3. For every step, the order is `compute1(...)`, force calculation, then
   `compute2(...)`. `compute1` therefore sees the force from the preceding
   configuration; `compute2` sees the newly calculated force.
4. `finalize_run(...)` runs after end-of-run measurements and before the
   ensemble is destroyed. Use it for required output or other normal-run
   finalization; use destructors and RAII for resource cleanup.

The `time_step` argument of `compute1` and `compute2` is the current step size
and can change during a run. If an algorithm intentionally needs the original
run step size, save that value in `initialize_run` and name the member to make
the distinction explicit.

A concrete integrator must provide both `compute1` and `compute2`, either by
overriding them or by inheriting an implementation from an intermediate base
class. An unused argument may remain unnamed in the definition. Use the
`Force&` supplied to `compute2` only when the algorithm needs force-model
services or state. Pass it to the helper that needs it instead of storing it in
the ensemble.

## Registration checklist

1. Add the class files and include its header from `integrate.cu`.
2. Add one branch to `Integrate::parse_ensemble`; construct the class there and
   copy only metadata required outside it.
3. Check parameter counts, numeric ranges, group indices, and incompatible
   options inside the concrete parser.
4. Verify the first-step initialization point and whether each time-step use is
   the current or run-initial value.
5. Add a valid regression case and invalid-input cases for important parser
   diagnostics. Add an adaptive-time-step case if either step-size meaning is
   relevant, and a multiple-`run` case for lifecycle-sensitive behavior.
6. Compare the candidate executable with the saved baseline, including the
   integrator's meaningful output (`compute.out` for heat transport), then run
   the full regression suite.

Refactoring, algorithm changes, and independent bug fixes should be separate
commits so that a numerical change is never hidden inside an interface change.
