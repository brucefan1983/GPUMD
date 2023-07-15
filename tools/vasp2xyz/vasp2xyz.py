'''
Extract the training set from the VASP output file "vasprun.xml".
Notice:
    Here the 'stress' of VASP (unit: eV/Å^3) is converted to 'virial' (unit: GPa)
    The 'energy' used is 'free_energy', in order to correspond to the shell file.
Run example:
    $ python vasp2xyz.py ${indoc}
Contributors:
    Zezhu Zeng
    Yuwen Zhang
    Ke Xu
'''

import os, sys
import numpy as np
from ase.io import read, write
from ase import Atoms, Atom
from tqdm import tqdm


def Convert_atoms(atom):
    # 1 eV/Å^3 = 160.21766 GPa
    xx,yy,zz,yz,xz,xy = -atom.calc.results['stress']*atom.get_volume() # *160.21766 
    atom.info['virial'] = np.array([(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])
    atom.calc.results['energy'] = atom.calc.results['free_energy']
    del atom.calc.results['stress']
    del atom.calc.results['free_energy']


def find_vasprun(start_path='.'):
    result = []
    for root, dirs, files in os.walk(start_path):
        if 'vasprun.xml' in files:
            result.append(os.path.join(root, 'vasprun.xml'))
    return result


file_list = find_vasprun(start_path=sys.argv[1])

cnum = 0     # total number of configuration
atoms_list, err_list = [], []
for dir_name in tqdm(file_list):
    try:
        atoms = read(dir_name.strip('\n'), index=":")
    except:
        err_list.append(dir_name)
        continue
    for ai in range(len(atoms)):
        Convert_atoms(atoms[ai])
        atoms_list.append(atoms[ai])
    cnum += len(atoms)

write('train.xyz', atoms_list, format='extxyz')
print('The total number of configurations is: {} \n'.format(cnum))

if err_list:
    print("The list of failed calculation files is as follows.")
    for err_dirname in err_list:
        print(err_dirname)
else:
    print("All calculations are successful!")
