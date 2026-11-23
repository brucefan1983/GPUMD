"""Write zero_nep_H_C_O_Cu.txt (neighbour capacity 20): a tiny NEP model (types H C O Cu) whose parameters are all zero.

The model gives exactly zero energy, force and virial, so that the output of a run with `dftd3` is
the D3 contribution alone and is bit-reproducible (NEP89 forces/energy are not, run to run, on this
GPU). Used by the BJ bit-identity tests (test_dftd3_zero.py). Run with any python:

    python make_zero_nep.py
"""
from pathlib import Path

T, NEURONS = 4, 2
DIM = (1 + 1) + (1 + 1) * 2                       # n_max 1 1, l_max 2 0 0 (L = 1, 2)
N_ANN = T * (NEURONS * DIM + NEURONS + NEURONS) + 1
N_DESC = (1 + 1) * (1 + 1) * T * T * 2            # radial + angular, basis_size 1 1
N = N_ANN + N_DESC

def model(mn):
    return f"""nep {T} H C O Cu
format_version 1
model nep
cutoff 3 3 {mn} {mn}
n_max 1 1
basis_size 1 1
l_max 2 0 0
ann {NEURONS} 0
energy_reference {' '.join(['0'] * T)}
@ parameters {N}
{chr(10).join(['0'] * N)}
@ q_scaler {DIM}
{chr(10).join(['1'] * DIM)}
"""
Path(__file__).with_name('zero_nep_H_C_O_Cu.txt').write_text(model(20))
