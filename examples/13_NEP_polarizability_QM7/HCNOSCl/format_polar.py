import os
import numpy as np
from ase.io import read as ase_read

result = np.loadtxt("polarizability_train.out")[:,0]

frames = ase_read("train.xyz",index=":")

atomic_numbers = [len(frame.get_atomic_numbers()) for frame in frames ]
tmp = result.reshape((-1,len(frames))) *np.array(atomic_numbers)

output = np.array([tmp[0],tmp[3],tmp[5],tmp[3],tmp[1],tmp[4],tmp[5],tmp[4],tmp[2]])
np.savetxt("NEP-polar.txt",output.T) 

result = np.loadtxt("polarizability_train.out")[:,1]

frames = ase_read("train.xyz",index=":")

atomic_numbers = [len(frame.get_atomic_numbers()) for frame in frames ]
tmp = result.reshape((-1,len(frames))) *np.array(atomic_numbers)

output = np.array([tmp[0],tmp[3],tmp[5],tmp[3],tmp[1],tmp[4],tmp[5],tmp[4],tmp[2]])
np.savetxt("NEP-DFT.txt",output.T) 
