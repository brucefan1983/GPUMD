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
python tests_pytest/validate_main_nep_spin3_training.py
python tests_pytest/validate_spin_training_lifecycle.py
python tests_pytest/validate_spin_training_multigpu.py --output "$RESULTS/training"
python tests_pytest/validate_nep_spin3_runtime.py
python tests_pytest/validate_nep_spin3_runtime.py --block-format
python tests_pytest/validate_spin_multigpu.py --binary src/gpumd --model tests_pytest/fixtures/nep_spin3/nep4_spin3_o3c2_uniform.nep --block-format --output "$RESULTS/runtime"
python -m pytest --noconftest -q tests_pytest/test_spin_tspin.py tests_pytest/test_spin_lattice_restart.py
python tests_pytest/validate_spin_tspin_runtime.py
```
