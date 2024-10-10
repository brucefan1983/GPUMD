from calorine.calculators import GPUNEP
from pathlib import Path
from ase.io import read
import os
import numpy as np

steps = [(10000, 0.1), (20000, 0.05), (100000, 0.01)]

#frequency_au = 0.002 # au 
frequency_fs = 0.099137602 # fs^-1 
coupling_ratio_au = 0.1    # au

# Convert coupling ratio and frequency from atomic units to eV & fs
HARTREE_TO_EV = 27.2114079527
BOHR_TO_ANGSTROM = 0.529177249
AU_TIME_TO_FS = 0.02418884326586 # fs, so number of fs per au
AU_FREQUENCY_TO_FS = 1 / AU_TIME_TO_FS # > 1, time unit is smaller so frequency unit should be larger

frequency_au = frequency_fs * AU_TIME_TO_FS
coupling_strength = coupling_ratio_au * np.sqrt(2 * frequency_au * HARTREE_TO_EV) / BOHR_TO_ANGSTROM

print(f'Frequency: {frequency:.3f} fs^-1, coupling ratio: {coupling_strength:.3f}')
exit()
for (step, dt) in steps:
    print(f"Running {step} with a timestep of {dt}")
    atoms = read('150mol.xyz')
    folder = f'dt{dt}'
    prefix = '/home/elindgren/repos/GPUMD/tests/gpumd/cavity/energy-conservation'
    #dump = int(0.1 / dt)
    dump = 1
    calc = GPUNEP('nep-pes.txt', atoms=atoms, command="/home/elindgren/repos/GPUMD/src/gpumd 2>&1")
    calc.set_directory(folder)
    parameters = [('potential', f'{prefix}/nep-pes.txt'),
                  ('time_step', dt),
                  ('velocity', 300),
                  ('ensemble', 'nve'),
                  ('cavity', (f'{prefix}/nep-dipole.txt', 1.0, 0.09914, dump)),
                  ('dump_thermo', dump),
                  ('run', step)]
    calc.run_custom_md(parameters)


