# Spin validation

Build `src/nep` and `src/gpumd` on the target GPU host. Set `GPUMD_SRC` to
the absolute path of the `src/` directory, and set `NEP_COMMAND` and
`GPUMD_COMMAND` to the matching binaries.
Keep results outside the repository.

```sh
python3 tests_pytest/validate_spin_model.py --require-jit --workdir "$RESULTS/soc"
python3 tests_pytest/validate_spin_nosoc.py --generate-fixtures "$RESULTS/oracle"
python3 tests_pytest/validate_spin_nosoc.py --fixtures "$RESULTS/oracle" --workdir "$RESULTS/nosoc"
python3 tests_pytest/validate_spin_compiled_shapes.py --workdir "$RESULTS/shapes"
python3 tests_pytest/validate_spin_training.py --check-jit
python3 tests_pytest/validate_spin_training_lifecycle.py
python3 -m pytest tests_pytest/test_spin_model_format.py tests_pytest/test_spin_validation.py
```

SOC uses frozen independent FP64/autograd fixtures. Non-SOC uses an independent
FP64 Legendre oracle and finite differences; fixture generation needs NumPy.
Checks cover derivatives, symmetries, zero spins, inactive species, periodic
cells, trained predictions and restarts. See [fixtures](../tests_pytest/fixtures/nep_spin/README.md).

For runtime drivers, run the `test_spin_*` pytest modules and the standalone
GLSD/FIRE validators. Four-GPU checks use `validate_spin_multigpu.py`,
`validate_spin_integrators_multigpu.py` and `validate_spin_training_multigpu.py`.
The runtime model for Fe/Ge tests is `tests_pytest/fixtures/nep_spin/c3l2s2_fege.nep`.
`test_spin_force_modifiers.py` compares random-force samples across processes and
requires a `DEBUG` build with a fixed PRNG seed. Other runtime tests can use the
normal build. Restart checks preserve spins and spin velocities at double
precision; a new SIB/GLSD ensemble initializes its noise counter anew.
Numerical checks do not establish accuracy on an application dataset.

## MD kernel specialization

MD contracts the fixed angular coefficients with the spin-channel projection once
when loading a model. C3/L2/S2 SOC and non-SOC layouts use compiled constant tables
and four neighboring lanes per atom/channel. With angular basis size 8, one edge
pass caches the radial value and derivative for the descriptor and force passes.
The cache is refreshed on every force evaluation, including spin correctors;
it does not assume positions stay unchanged between calls. Other shapes use the
precontracted generic kernels. No runtime compiler or new input keyword is required.

Regenerate the constant layouts from the shared layout builder, without changing
any independent oracle, using:

```sh
c++ -std=c++14 -Isrc tests_pytest/generate_spin_md_layouts.cpp -o "$RESULTS/generate-spin-md"
"$RESULTS/generate-spin-md" > src/force/nep_spin/spin_md_layouts.cuh
```

Performance acceptance should compare identical models and initial velocities in
interleaved GPUMD runs. Report NVE force evaluation separately from spin integrators
such as SIB, which evaluate the force again for the corrector.
