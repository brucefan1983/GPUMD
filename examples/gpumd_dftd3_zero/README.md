# NEP with the DFT-D3(0) dispersion correction

This example runs 500 steps of NVT Langevin molecular dynamics at 300 K using the NEP model from `../nep_train/nep.txt`, with the DFT-D3 correction with zero damping, D3(0), added by the line `dftd3 pbe 12 6 zero` in `run.in`. The functional (`pbe`), the potential cutoff (12 Angstrom), and the coordination number cutoff (6 Angstrom) are those of the `dftd3` keyword; omit the last token (or use `bj`) for the Becke-Johnson damping.

Run:

```bash
/path/to/gpumd
```

The header printed at the start of the run lists the functional, the damping, and the D3 parameters. `thermo.out` is written every 10 steps.

This example only demonstrates the input. Whether a D3 correction is appropriate depends on how the NEP model was trained; see the notes on the `dftd3` page of the manual about double counting.
