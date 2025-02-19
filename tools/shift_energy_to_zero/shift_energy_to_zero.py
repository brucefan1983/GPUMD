#!/usr/bin/env python
# coding: utf-8
'''
Purpose:
    Shift total energy to ~0 by substracting atomic energies using the least square method.
Run example:
    $ python shift_energy_to_zero.py your.xyz
    A new file shifted.xyz will be generated which contains the shifted potential energies.
Author:
    Nan Xu (tamas@zju.edu.cn)
'''
import numpy as np
from ase.io import read,write
import sys
import os

def SVD_A(A,b):
    #solve A*x=b
    U,S,V=np.linalg.svd(A)
    B=np.matmul(U.T,b) 
    X=B[:len(S),:]/S.reshape(len(S),-1)
    x=np.matmul(V.T,X)#attention
    return x

if len(sys.argv)==1:
    raise SystemError("Wrong usage. Example: python shift_energy_to_zero.py your.xyz")
if not os.path.exists(sys.argv[1]):
    raise SystemError("Your xyz file does not exist.")
if sys.argv[1].split('.')[-1].upper() != "XYZ":
    raise SystemError("You should provide file with XYZ format.")
all_frames=read(sys.argv[1],index=":")

flatten_comprehension = lambda matrix: [item for row in matrix for item in row]
all_elements = sorted(set(flatten_comprehension([i.get_chemical_symbols() for i in all_frames])))
coeff_matrix = np.zeros((len(all_frames),len(all_elements)))
energy_matrix = np.zeros((len(all_frames),1))

#test whether get_potential_energy works
default_energy_store = True
try:
    all_frames[0].get_potential_energy()
except RuntimeError:
    tmp = all_frames[0].info
    tmp =list(tmp.keys())
    tmp1 = [i.upper() for i in tmp]
    if "ENERGY" not in tmp1:
        raise SystemError("Ensure your xyz file has energy.")
    keyword = tmp[tmp1.index("ENERGY")]
    default_energy_store= False

if default_energy_store:
    for i in range(len(all_frames)):
        for j in range(len(all_elements)):
            coeff_matrix[i][j]= all_frames[i].get_chemical_symbols().count(all_elements[j])
        energy_matrix[i][0] = all_frames[i].get_potential_energy()
else:
    for i in range(len(all_frames)):
        for j in range(len(all_elements)):
            coeff_matrix[i][j]= all_frames[i].get_chemical_symbols().count(all_elements[j])
        energy_matrix[i][0] = all_frames[i].info[keyword]    
print('Normalizing energy....')
#check if matrix is underdetermined

if np.linalg.matrix_rank(coeff_matrix) < len(all_elements):
    print("Warning! The coeff_matrix is underdetermined, adding constrains....")
    number_of_constrains = len(all_elements) - np.linalg.matrix_rank(coeff_matrix)
    import itertools
    to_add_constrain_pairs = []
    for i in itertools.combinations(range(len(all_elements)), 2):
        additional_matrix = np.zeros(len(all_elements))
        additional_matrix[i[0]] = 1
        additional_matrix[i[1]] = -1
        additional_energy = np.zeros(1)
        coeff_matrix = np.r_[coeff_matrix,[additional_matrix]]
        energy_matrix = np.r_[energy_matrix,[additional_energy]]
    
print('Atomic energy---->')
atomic_shifted_energy = SVD_A(coeff_matrix,energy_matrix)
for i in range(len(all_elements)):
    print("%s:%f" %(all_elements[i],atomic_shifted_energy[i][0]))
    
shifted_energy = (energy_matrix-np.matmul(coeff_matrix,atomic_shifted_energy)).flatten()

print("Averaged energies now: %f eV. " %shifted_energy[:len(all_frames)].mean())
print("Absolute maximum energy now: %f eV." %max(abs(shifted_energy[:len(all_frames)])))

for i in range(len(all_frames)):
    try:
        forces = all_frames[i].get_forces()
        all_frames[i].new_array('forces',forces)
    except:
        pass
    all_frames[i].calc = None
    all_frames[i].info['energy'] = shifted_energy[i]
write("shifted.xyz",all_frames)
print("Done! New shifted.xyz is generated.")
