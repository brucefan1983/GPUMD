from calorine.calculators import GPUNEP
from pathlib import Path
from ase.io import read
import os
import numpy as np

steps = [(2000, 0.5), (10000, 0.1), (20000, 0.05), (100000, 0.01)]

frequency_au = 0.002 # au 
coupling_ratio_au = 1.132 # au

# Convert coupling ratio and frequency from atomic units to eV & fs
HARTREE_TO_EV = 27.2114079527
BOHR_TO_ANGSTROM = 0.529177249
AU_TIME_TO_FS = 0.02418884326586

frequency = frequency_au / AU_TIME_TO_FS
coupling_strength = coupling_ratio_au * np.sqrt(2 * frequency_au * HARTREE_TO_EV) / BOHR_TO_ANGSTROM

print(f'Frequency: {frequency:.3f} fs^-1, coupling ratio: {coupling_strength:.3f}')

for (step, dt) in steps:
    print(f"Running {step} with a timestep of {dt}")
    atoms = read('150mol.xyz')
    folder = f'dt{dt}'
    prefix = '/home/elindgren/repos/GPUMD/tests/gpumd/cavity/energy-conservation'

    calc = GPUNEP('nep-pes.txt', atoms=atoms, command="/home/elindgren/repos/GPUMD/src/gpumd 2>&1")
    calc.set_directory(folder)
    parameters = [('potential', f'{prefix}/nep-pes.txt'),
                  ('potential', f'{prefix}/nep-dipole.txt'),
                  ('time_step', dt),
                  ('velocity', 300),
                  ('ensemble', 'nve'),
                  ('dump_thermo', 1),
                  ('run', step)]
    calc.run_custom_md(parameters)


