# Spin integration validation

Build on a CUDA GPU with the repository toolchain:

```sh
make -C src -j8 gpumd nep
export GPUMD_COMMAND="$PWD/src/gpumd"
export NEP_COMMAND="$PWD/src/nep"
python -m pytest --noconftest -q tests_pytest/test_spin_tspin.py tests_pytest/test_spin_restart.py tests_pytest/test_spin_sib.py tests_pytest/test_spin_fire.py tests_pytest/test_spin_force_modifiers.py
```

These tests run the public `gpumd` executable. TSPIN is compared with an
independent one-step formula. Restart checks cover physical-state round trips,
new NVT/NPT runs, and output-independent run boundaries. Nonmagnetic elements remain masked.
Spin FIRE checks cover field-free spin, atomic and cell relaxation. Fields declared
before minimization affect only the following MD run; the tests check this
separation and the field cleanup after that run.
The force-modifier checks cover SIB/GLSD correctors in small and large boxes,
including additive magnetic fields, retained mechanical noise, and clearing
Actions before a subsequent ordinary run.
SIB is compared with an independent Cayley and thermal-increment reference, including external magnetic fields.

On an allocation with four visible GPUs:

```sh
python tests_pytest/validate_spin_glsd_runtime.py --binary "$GPUMD_COMMAND" --output "$RESULTS/glsd"
```

The GLSD command checks analytic zero/constant-field trajectories, deterministic
radial relaxation, physical-state restart, fresh per-run noise seeding, and 1/2/4 GPU parity.
The bundled-model comparisons use a 0.01 fs timestep and require convergence
within 100 midpoint corrections; default-cap behavior is checked separately.
It does not establish equilibrium sampling of an arbitrary trained potential.

For deterministic spin/lattice timestep convergence on one GPU:

```sh
python tests_pytest/validate_spin_glsd_coupled.py --binary "$GPUMD_COMMAND" --output "$RESULTS/glsd-coupled"
python -m pytest --noconftest -q tests_pytest/test_mttk_chain.py
python -m pytest --noconftest -q tests_pytest/test_nhc_stability.py
```

The coupled check keeps the model, initial state, elapsed time and physical
thermostat/barostat periods fixed while halving the timestep. It checks position,
velocity and spin errors for weak and finite NHC coupling, plus the cell for
triclinic MTTK, and verifies continuity across a periodic crossing. The MTTK
temperature-chain check compares production stepping with independent host RK4.
An optional `--reference /path/to/previous/gpumd` supplies a negative control;
both executables must support the same model format.

The common NHC integrator retains four RESPA substeps and checks for nonfinite
thermostat states. Its host RK4 check uses a 0.5 fs timestep and coupling times
of 100 steps for four atoms and 500 steps for 5,575,680 atoms. A shorter coupling
time can still destabilize a large reservoir with a large temperature mismatch;
the fixed-four integrator does not increase substeps automatically.

For accumulated periodic images on one GPU:

```sh
python tests_pytest/validate_spin_glsd_images.py --binary "$GPUMD_COMMAND" --output "$RESULTS/glsd-images"
python -m pytest --noconftest -q tests_pytest/test_spin_midpoint_position.py
```

Five ordinary NVE steps accumulate independent positive and negative images.
The following GLSD steps must match a control restarted from the same wrapped
positions, velocities and spins, for both orthogonal and triclinic cells.
An optional `--reference /path/to/previous/gpumd` verifies the old orthogonal
image-history failure. The host test also covers deforming cells and partial PBC.

Use [runtime validation](spin_runtime_validation.md) for the fixed-spin and
ordinary full regression commands, and [training validation](spin_validation.md)
for native/JIT model and training checks. Run on the same source snapshot; record
the source and binary hashes, GPU allocation, toolchain, and actual results.

TSPIN/SIB trajectory and magnetic endpoint comparison on four allocated GPUs:

```sh
python tests_pytest/validate_spin_integrators_multigpu.py --binary "$GPUMD_COMMAND" --output "$RESULTS/integrators"
```
