# DeePMD-kit PyTorch potential

This example runs a short NVE simulation of Cu using a DeePMD-kit PyTorch model.

The model file is not included. Place a frozen model named `frozen_model.pt2` in this directory. A compatible model can be trained or obtained with DeePMD-kit and frozen with its PyTorch backend.

Run:

```bash
/path/to/gpumd
```

The main outputs are `thermo.out`, `dump.xyz`, and `restart.xyz`.
