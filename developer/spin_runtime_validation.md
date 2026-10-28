# Fixed-spin runtime validation

Build `gpumd` and `nep` on a CUDA node, choosing the device architecture:

```sh
make -C src -j8 CUDA_ARCH=-arch=sm_70 gpumd nep
export GPUMD_COMMAND="$PWD/src/gpumd"
export NEP_COMMAND="$PWD/src/nep"
```

Use Python with NumPy and pytest. Set `RESULTS` to a new output directory outside
this checkout. The first two GPU scripts need one GPU; run them with one
allocated device visible. The third requires four allocated GPUs in one process.

```sh
python -m pytest --noconftest -q tests_pytest/test_spin_model_format.py tests_pytest/test_spin_validation.py
python tests_pytest/validate_nep_spin3_runtime.py
python tests_pytest/validate_mfield_runtime.py
python tests_pytest/validate_spin_multigpu.py --binary "$GPUMD_COMMAND" --model tests_pytest/fixtures/nep_spin3/nep_spin_o3c2_uniform.nep --output "$RESULTS/runtime"
```

The runtime oracle checks cover energy, atomic force, magnetic force and all
nine virial components, plus energy-reference invariance, excluded-spin
masking, replication and fixed-spin restart. The frozen reference provenance
is documented in [fixtures](../tests_pytest/fixtures/nep_spin3/README.md).
The external-field script checks analytic Zeeman energy and magnetic force,
group and region selection, boundary crossing, excluded spins and invalid inputs.
Multi-GPU comparisons use componentwise tolerances and reject nonfinite values.
They cover small cells, nonuniform and triclinic systems, cutoff boundaries,
partition directions, motion across partition boundaries and energy references.

Run the [training validation](spin_validation.md) against the same build to
check the shared descriptor implementation and native/JIT agreement. Keep its
model, seed, budget and GPU settings fixed when comparing revisions.

For ordinary GPUMD and NEP behavior, build an unmodified upstream checkout and
run the full regression suite. `BASELINE_GPUMD` and `BASELINE_NEP` must identify
those independently built upstream executables:

```sh
python tests_regression/run_regression.py --repo-root "$PWD" \
  --baseline "$BASELINE_GPUMD" --candidate "$GPUMD_COMMAND" \
  --baseline-nep "$BASELINE_NEP" --candidate-nep "$NEP_COMMAND" --suite full
```

These checks validate implementation contracts, not fit quality or performance.
Spin integrators and their checkpoint state are outside this suite.
