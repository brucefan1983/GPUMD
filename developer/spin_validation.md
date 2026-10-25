# Spin regression checks

Run these commands from the repository root. Use Python with NumPy and pytest,
a C++ compiler for the format tests, and CUDA with a supported NVIDIA GPU for
the runtime tests. Load your site's CUDA environment before building; the tests
inherit the environment and do not configure library paths.

## Model format and runtime

The format tests compile the production codec on the CPU:

```sh
python -m pytest --noconftest -q tests_pytest/test_spin_model_format.py
```

Build both executables on the target machine. Set `CUDA_ARCH` for its GPU; this
example targets V100:

```sh
make -C src -j8 CUDA_ARCH=-arch=sm_70 gpumd nep
python tests_pytest/validate_nep_spin3_runtime.py
python tests_pytest/validate_nep_spin3_runtime.py --block-format
python -m pytest --noconftest -q tests_pytest/test_spin_fire.py \
  tests_pytest/test_spin_sib.py tests_pytest/test_spin_tspin.py
python tests_pytest/validate_spin_tspin_runtime.py
```

These tests default to `src/gpumd`; `GPUMD_COMMAND` selects another executable.
`--noconftest` runs the self-contained Spin tests without importing the ordinary
NEP suite's ASE/Calorine fixtures. Runtime tests require a working GPU executable.

`tests_pytest/fixtures/nep_spin3` contains two legacy models (C=2 and C=3), one
structure, and three frozen oracle files. The C=2 typewise case reuses the C=2
uniform parameters and changes only `spin_cutoff 6.0` to `spin_cutoff 4.0 6.0`.
Each oracle records the SHA256 of its complete model input; the loader verifies
that hash, including the generated typewise header. The versioned `model nep_spin`
format and legacy files are checked against the same frozen energy, force,
magnetic force, and virial values. Lifecycle checks cover prediction, MD,
restart, and compatibility with existing files and inputs. Unsupported versions
and conflicting selectors must fail. Training examples use `spin_version 1`.
Do not regenerate expected values with the candidate implementation when fixing
a test.

## Training and multiple GPUs

The following self-contained training checks generate their small datasets in
temporary directories:

```sh
python tests_pytest/validate_main_nep_spin3_training.py
python tests_pytest/validate_spin_training_lifecycle.py
```

For GPU scaling checks, expose four allocated GPUs through `CUDA_VISIBLE_DEVICES`.
Choose an output directory outside the checkout:

```sh
python tests_pytest/validate_spin_training_multigpu.py --output "$RESULTS/training"
python tests_pytest/validate_spin_multigpu.py --binary src/gpumd \
  --model tests_pytest/fixtures/nep_spin3/nep4_spin3_o3c2_uniform.nep \
  --block-format --output "$RESULTS/runtime"
```

The training comparison uses one, two, and four GPUs, multiple structures per
batch, and independently recomputed energy errors. The runtime comparison covers
domain partitioning, boundary conditions, energy, forces, and magnetic forces.

Additional probes must be linked against the current NEP objects, excluding its
`main` object. On Linux/CUDA, after building `nep`, for example:

```sh
for probe in spin_mforce_split restart_parity response_fitness; do
  objects=$(find src/main_nep src/utilities -name '*.cu' ! -path 'src/main_nep/main.cu' \
    | sed 's/\.cu$/.o/')
  nvcc -std=c++17 -O2 -arch=sm_70 -Isrc "tests_pytest/$probe.cu" $objects \
    -lcublas -lcusolver -lcufft -ldl -lpthread -o "$RESULTS/$probe"
done
python tests_pytest/validate_spin_mforce_split.py \
  --oracle "$RESULTS/spin_mforce_split" --output "$RESULTS/split"
RESTART_PROBE="$RESULTS/restart_parity" python tests_pytest/validate_restart_scaler.py
RESPONSE_FITNESS_PROBE="$RESULTS/response_fitness" \
  python tests_pytest/validate_response_file_runtime.py
```

Set `RESULTS` to an existing absolute directory before using these commands.
The restart check covers both ordinary NEP and Spin, preserves the SNES search
mean separately from the exported elite, and checks scaler import. The split
probe checks parallel/perpendicular magnetic losses against an independent
projection calculation, including zero spins and parser rejection cases.

## GLSD and lattice-driver restart

The standalone GLSD probe checks deterministic convergence and additive-noise
statistics against independent analytic expectations. It needs no trained model:

```sh
nvcc -std=c++14 -O3 -arch=sm_70 -Isrc tests_pytest/spin_glsd_analytic.cu \
  src/integrate/spin_glsd.cu -o "$RESULTS/spin_glsd_analytic"
python tests_pytest/validate_spin_glsd_analytic.py "$RESULTS/spin_glsd_analytic"
GPUMD_COMMAND=src/gpumd python tests_pytest/validate_spin_sib_runtime.py
```

Full runtime and restart checks require four allocated GPUs, a compatible Fe
Spin model, and independently preserved reference executables. Use absolute
paths and fresh output directories outside the checkout:

```sh
python tests_pytest/validate_spin_glsd_runtime.py \
  --binary /path/to/current/gpumd --reference /path/to/pre-glsd/gpumd \
  --mobility-reference /path/to/pre-damping/gpumd \
  --model /path/to/Fe/nep.txt --output "$RESULTS/glsd"
python tests_pytest/validate_mttk_restart.py \
  --binary /path/to/current/gpumd --reference /path/to/pre-mttk-restart/gpumd \
  --model /path/to/Fe/nep.txt --output "$RESULTS/mttk"
python tests_pytest/validate_nhc_restart.py \
  --binary /path/to/current/gpumd --reference /path/to/pre-nhc-restart/gpumd \
  --model /path/to/Fe/nep.txt --output "$RESULTS/nhc"
```

These reference executables and the Fe model are external prerequisites, not
bundled fixtures. Record their source revisions and hashes; never substitute
the candidate for an independent reference. The GLSD mobility reference must
support the former dimensional input. Restart checks compare continuous and
resumed trajectories step by step, exercise invalid inputs, and use missing-state
controls to detect incomplete restoration. See the
[restart documentation](../doc/gpumd/input_parameters/dump_restart.rst) for the
supported continuation contracts and the
[GLSD documentation](../doc/gpumd/input_parameters/ensemble_glsd.rst) for the
physical equation and timestep limits.

Keep run logs, machine details and numerical reports outside the source tree.
Passing these regressions establishes the covered software contracts; it does
not establish magnetic phase behavior or extrapolative model accuracy.
