# Static qNEP calculation

This example performs a one-step, zero-time-step GPUMD calculation with the qNEP model from `../qnep_train/nep.txt` and writes forces to `dump.xyz`.

Run:

```bash
/path/to/gpumd
python check_force.py
```

`check_force.py` compares the GPUMD forces with the corresponding forces produced by the `nep` executable. The differences should be very small.
