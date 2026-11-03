# Spin integration validation

Build on a CUDA GPU with the repository toolchain:

```sh
make -C src -j8 gpumd nep
export GPUMD_COMMAND="$PWD/src/gpumd"
export NEP_COMMAND="$PWD/src/nep"
python -m pytest --noconftest -q tests_pytest/test_spin_tspin.py tests_pytest/test_spin_lattice_restart.py tests_pytest/test_spin_sib.py tests_pytest/test_spin_fire.py
```

These tests run the public `gpumd` executable. TSPIN is compared with an
independent one-step formula, and lattice checkpoints exercise NHC/MTTK
restoration and run-block ownership. Nonmagnetic elements remain masked.
SIB is compared with an independent Cayley and thermal-increment reference, including external magnetic fields.

On an allocation with four visible GPUs:

```sh
python tests_pytest/validate_spin_glsd_runtime.py --binary "$GPUMD_COMMAND" --output "$RESULTS/glsd"
```

The GLSD command checks analytic zero/constant-field trajectories, deterministic
radial relaxation, checkpoint continuity, noise seeding, and 1/2/4 GPU parity.
It does not establish equilibrium sampling of an arbitrary trained potential.

Use [runtime validation](spin_runtime_validation.md) for the fixed-spin and
ordinary full regression commands, and [training validation](spin_validation.md)
for native/JIT model and training checks. Run on the same source snapshot; record
the source and binary hashes, GPU allocation, toolchain, and actual results.

TSPIN/SIB trajectory and magnetic endpoint comparison on four allocated GPUs:

```sh
python tests_pytest/validate_spin_integrators_multigpu.py --binary "$GPUMD_COMMAND" --output "$RESULTS/integrators"
```
