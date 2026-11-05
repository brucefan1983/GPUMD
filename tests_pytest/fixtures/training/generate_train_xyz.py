"""Generate the small labelled training set used by test_nep_model_consistency.py.

The structures are rattled copies of the BaTiO3 fixture cell, labelled with energies and forces
from the committed nep_BaTiO3.txt model, so the labels are consistent with a real NEP model rather
than synthetic.
The set exists only to let the `nep` executable start and finish a very short training run, not to
produce a meaningful fit.

Run from this directory with a built gpumd executable available:

    python3 generate_train_xyz.py
"""
from pathlib import Path

from ase.io import read, write
from calorine.calculators import GPUNEP

FIXTURES_DIR = Path(__file__).resolve().parents[1]
GPUMD_EXECUTABLE = FIXTURES_DIR.parents[1] / 'src' / 'gpumd'
NUMBER_OF_STRUCTURES = 4
RATTLE_STANDARD_DEVIATION = 0.05  # Angstrom

reference = read(FIXTURES_DIR / 'structures' / 'BaTiO3-nat40-rattled.xyz')
model = FIXTURES_DIR / 'models' / 'nep_BaTiO3.txt'

structures = []
for seed in range(NUMBER_OF_STRUCTURES):
    structure = reference.copy()
    structure.rattle(stdev=RATTLE_STANDARD_DEVIATION, seed=seed)
    structure.calc = GPUNEP(str(model), command=str(GPUMD_EXECUTABLE))
    structure.info['energy'] = structure.get_potential_energy()
    structure.arrays['force'] = structure.get_forces()
    structure.calc = None
    structures.append(structure)

write(
    FIXTURES_DIR / 'training' / 'train.xyz',
    structures,
    format='extxyz',
    columns=['symbols', 'positions', 'force'],
)
