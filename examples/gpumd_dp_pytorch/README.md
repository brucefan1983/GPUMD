# DeePMD-kit PyTorch potential

This example runs a short NVE simulation of Cu using a DeePMD-kit PyTorch model.

This example requires GPUMD to be compiled with Deep Potential support
(`USE_DEEPMD`) and the required DeePMD-kit/PyTorch libraries. See the
[installation guide](../../doc/installation.rst) for the build instructions.

The model file is not included. The supplied `run.in` expects a DPA4
AOTInductor model named `frozen_model.pt2`. For DPA2/DPA3, use the corresponding
`.pth` model and update the model filename in `run.in`.

Run:

```bash
/path/to/gpumd
```

The main outputs are `thermo.out`, `dump.xyz`, and `restart.xyz`.
