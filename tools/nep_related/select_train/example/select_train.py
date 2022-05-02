"""
Function:
Select strutures from the existing-train.in file and merge them to a new output-train.in file

Usage: 
1. Select i-th to j-th strutures of the existing train.in file:
python select_train.py existing-train.in i:j output-train.in

2. Select i-th to j-th, m-th to n-th strutures (can select multiple groups) of the existing train.in file:
python select_train.py existing-train.in i:j m:n output-train.in

3. Randomly select k structures from the existing train.in file. The ran_sample.txt after running saves 
the random sampled index of the existing train.in file:
python select_train.py existing-train.in random k output-train.in
"""
import sys
from turtle import st
import numpy as np
import random

def loadtrain(file):

    tr = open(file,'r').readlines()
    num = int(tr[0])
    atom_num = []
    if_virial = []
    for i in tr[1:num + 1]:
        atom_num.append(int(i.split()[0]))
        if_virial.append(int(i.split()[1]))
    atom_num = np.array(atom_num)
    cum = (atom_num + 2).cumsum()
    cum = np.insert(cum,0,0)

    structures = []
    for i,t in enumerate(cum[:-1]):
        ss = {}
        head = tr[num+t+1].rstrip('\n').split()
        ss.update({'e':float(head[0])})
        if if_virial[i] == 1:
            vv = np.array([float(head[1]),float(head[2]),float(head[3]),float(head[4]),float(head[5]),float(head[6])])
            ss.update({'virials':vv})
        if if_virial[i] == 0:
            ss.update({'virials':None})

        c = tr[num+t+2].rstrip('\n').split()
        cc = np.array([[float(c[0]),float(c[1]),float(c[2])],[float(c[3]),float(c[4]),float(c[5])],[float(c[6]),float(c[7]),float(c[8])]])
        ss.update({'cell':cc})
        
        

        p = []
        f = []
        s = []
        for j,k in enumerate(tr[num+cum[i]+3:num+1+cum[i+1]]):
            pf = k.rstrip('\n').split()
            s.append(pf[0])
            pp = np.array([float(pf[1]),float(pf[2]),float(pf[3])])
            ff = np.array([float(pf[4]),float(pf[5]),float(pf[6])])
            p.append(pp)
            f.append(ff)
        
        ss.update({'symbols':s})
        ss.update({'positions':p})
        ss.update({'forces':f})
        ss.update({'atom_num':atom_num[i]})
        ss.update({'if_virial':if_virial[i]})
        
        structures.append(ss)
    
    return structures 

def tonep(structures,output):
    n = len(structures)
    with open(output,'w') as f:
        f.write('{}\n'.format(n))
        for i in range(n):
            f.write('{} {}\n'.format(structures[i]['atom_num'],structures[i]['if_virial']))
        for i in range(n):
            if (i+1) % 100 == 0 :
                print("Merging {} stutures".format(i+1))
            if i == n-1:
                print("Merged total {} stutures".format(n))
            if structures[i]['if_virial'] == 0: 
                f.write('{:.12f}\n'.format(structures[i]['e']))
            if structures[i]['if_virial'] == 1:
                f.write('{:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f}\n'.format(structures[i]['e'],\
                    *structures[i]['virials'].flatten()))
            f.write('{:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f}\n'.format(*structures[i]['cell'].flatten()))
            for k,a in enumerate(structures[i]['symbols']):
                f.write('{} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f}\n'.format(\
                    a, *structures[i]['positions'][k].flatten(), *structures[i]['forces'][k].flatten()))


if __name__ == "__main__":

    structures = loadtrain(sys.argv[1])
    print("There are {} inital strutures in the {}".format(len(structures), sys.argv[1]))
    if sys.argv[2] == 'random':
        sp = random.sample(range(len(structures)),int(sys.argv[3]))
        np.savetxt('ran_sample.txt', sp ,fmt = "%d")
        new_structures = [structures[i] for i in sp]
        tonep(new_structures, output = sys.argv[-1])
    
    else:
        new_structures = []
        for v in sys.argv[2:-1]:
            b = int(v.split(':')[0]) - 1; e = int(v.split(':')[1]) - 1
            new_structures = new_structures + structures[b:e]
        tonep(new_structures, output = sys.argv[-1])
        


