# Spin validation

Build `src/nep` and `src/gpumd` on the target GPU host. Set `GPUMD_SRC`,
`NEP_COMMAND` and `GPUMD_COMMAND` to the source tree and matching binaries.
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
Numerical checks do not establish accuracy on an application dataset.
