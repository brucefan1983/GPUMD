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
It does not establish equilibrium sampling of an arbitrary trained potential.

Use [runtime validation](spin_runtime_validation.md) for the fixed-spin and
ordinary full regression commands, and [training validation](spin_validation.md)
for native/JIT model and training checks. Run on the same source snapshot; record
the source and binary hashes, GPU allocation, toolchain, and actual results.

TSPIN/SIB trajectory and magnetic endpoint comparison on four allocated GPUs:

```sh
python tests_pytest/validate_spin_integrators_multigpu.py --binary "$GPUMD_COMMAND" --output "$RESULTS/integrators"
```
