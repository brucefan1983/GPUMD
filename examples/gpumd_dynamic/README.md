# Molecular dynamics with NEP

This example runs 10,000 steps of NVT Langevin molecular dynamics at 300 K using the NEP model from `../nep_train/nep.txt`.

Run:

```bash
/path/to/gpumd
python3 plot_thermo.py
```

`thermo.out` is written every 10 steps. `plot_thermo.py` reads it and writes `thermo.png`, showing the temperature as a function of simulation time.
