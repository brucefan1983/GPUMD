import numpy as np
from ase.io import read, write
from ase import Atoms, Atom
from tqdm import tqdm
import glob 

i = 0    # total number of configuration
file_list = glob.glob('vasprun*.xml') # please put you all vasprun.xml files to one document
atoms_list = []  
for dir_name in tqdm(file_list):
    i+= 1
    atoms = read(dir_name.strip('\n'))
    atoms_list.append(atoms) 

write('train.xyz', atoms_list, format='extxyz')
print('The total number of configurations is: 'i)
