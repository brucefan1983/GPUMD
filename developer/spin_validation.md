# Spin regression checks

Build on the target GPU with its CUDA architecture, for example:

```sh
make -C src -j8 CUDA_ARCH=-arch=sm_70 gpumd nep
```

Use Python with NumPy and pytest. Set RESULTS to a new directory outside the
checkout. Multi-GPU checks require four allocated GPUs visible through
CUDA_VISIBLE_DEVICES. GPUMD_COMMAND selects the runtime binary; otherwise tests
use src/gpumd. Keep the frozen model/oracle values independent of the candidate.

```sh
python -m pytest --noconftest -q tests_pytest/test_spin_model_format.py
python tests_pytest/validate_main_nep_spin3_training.py --check-jit
python tests_pytest/validate_spin_training_lifecycle.py
python tests_pytest/validate_spin_training_multigpu.py --output "$RESULTS/training"
```

Training checks cover five representative layouts, 24 frozen-model oracle cases,
actual energy errors, magnetic loss components, model IO and restart.
`--check-jit` requires successful structural and Spin layout specialization,
then compares native and compiled checkpoints. Run it on a supported Linux/CUDA
build with nvcc available; omit the flag for native-only checks. Flexible ZBL is rejected;
universal ZBL remains supported. The CPU model codec is checked with strict
C++14 compilation. Build the focused probes with `-std=c++14` as well.
Initialization checks share `restart_parity.cu`; torque and response edge
cases share `spin_mforce_split.cu` and `response_fitness.cu`.

Models use only `model nep_spin` and `spin_version 1`, with named parameter
and scaler blocks. Restart uses the current shared `nep_restart` format.
