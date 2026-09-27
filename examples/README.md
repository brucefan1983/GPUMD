# GPUMD examples

This directory contains small examples for the `nep` and `gpumd` executables. They are intended to demonstrate basic input and output workflows rather than provide complete application tutorials.

Compile GPUMD first in `src/`, then run the appropriate executable from an example directory:

```bash
/path/to/nep
# or
/path/to/gpumd
```

The Python analysis scripts require NumPy and Matplotlib:

```bash
python -m pip install numpy matplotlib
```

## Examples

| Folder | Purpose |
| --- | --- |
| `nep_train` | Train a standard NEP potential for PbTe. |
| `qnep_train` | Train a charge-aware qNEP potential for PbTe. |
| `nep_prediction` | Use a trained NEP model to predict a dataset. |
| `gpumd_static` | Run a one-step static force calculation with NEP. |
| `gpumd_static_qnep` | Run a one-step static force calculation with qNEP. |
| `gpumd_dynamic` | Run a short NVT molecular-dynamics simulation with NEP. |
| `gpumd_dp_pytorch` | Run GPUMD with a DeePMD-kit PyTorch model. |
| `gpumd_nnap` | Run GPUMD with an NNAP potential. |

Each example has its own `README.md` with the required files and commands.

For more realistic workflows, see the GPUMD Tutorials repository: https://github.com/brucefan1983/GPUMD-Tutorials
