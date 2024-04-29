from calorine.calculators import CPUNEP
from ase.io import read, write

structures = read('dump.xyz', ':')

predicted = []
for structure in structures:
    calc = CPUNEP('../C_2022_NEP3_MODIFIED.txt')
    atoms = structure.copy()
    info = atoms.info
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    info = {}
    info['Time'] = atoms.info['Time']
    info['energy'] = energy
    atoms.info = info
    
    arrays = {}
    arrays['numbers'] = atoms.arrays['numbers']
    arrays['positions'] = atoms.arrays['positions']
    arrays['forces'] = forces
    atoms.arrays = arrays
    predicted.append(atoms)
write('predicted.xyz', predicted)
