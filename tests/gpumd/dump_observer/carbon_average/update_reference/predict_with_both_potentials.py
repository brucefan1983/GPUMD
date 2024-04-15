from calorine.calculators import CPUNEP
from ase.io import read, write

files = [
'../C_2022_NEP3.txt',
'../C_2022_NEP3_MODIFIED.txt'
]
structures = read('observer.xyz', ':')

predicted = {
        'potential0': [],
        'potential1': [],
}
for structure in structures:

    for i, file in enumerate(files):
        calc = CPUNEP(file)
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
        predicted[f'potential{i}'].append(atoms)
write('predicted0.xyz', predicted['potential0'])
write('predicted1.xyz', predicted['potential1'])
