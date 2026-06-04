# GPUMD with DeePMD-kit PyTorch Backend

This example demonstrates running GPUMD with a DeePMD-kit PyTorch model (DPA2/DPA3/DPA4).

## Files

- `run.in` — GPUMD input script (NVE, 100 steps)
- `model.xyz` — Cu FCC 2×2×2 supercell (32 atoms)
- `dp_settings.txt` — Type map for the DP model

## Model file (not included due to size)

You can use pre-trained universal models from AIS Square or other sources.

For example, to use **DPA-2.3.1** from AIS Square (DPA4 architecture):

1. Visit https://aissquare.com and search for "DPA-2.3.1"
2. Download the model checkpoint (e.g., `dpa-2.3.1-v3.0.0b4-medium`)
3. Freeze it to the PyTorch format:

```bash
unzip dpa-2.3.1-v3.0.0b4-medium.zip
cd dpa-2.3.1-v3.0.0b4-medium
dp --pt freeze -o frozen_model.pth  # DPA4 produces .pt2 automatically
```

The frozen model will be `frozen_model.pt2` for DPA4.

Alternatively, train your own model with DeePMD-kit and freeze it using `dp --pt freeze`.

## Running

Place the frozen model in this directory, then:

```bash
gpumd
```

## Expected output

- `thermo.out` — Thermodynamic quantities per step
- `dump.xyz` — Trajectory in extended XYZ format
- `restart.xyz` — Final configuration with velocities
