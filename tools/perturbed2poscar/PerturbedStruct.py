#!/usr/bin/env python

import os
import sys
import numpy as np
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.lattice import Lattice


def GRP_Structures(filename, tolerance=None, n_struct=None):
    """
    This script is used to generate many strucures with random perturbation in lattice.
    :param filename: original structure path
    :param tolerance: tolerance for changing lengths and angles, which is a list.
    :param n_struct: the number of structures gernerated by this script.

    """
    if tolerance is None:
        tolerance = [5, 5]

    if n_struct is None:
        n_struct = 10

    origin_struct = Poscar.from_file(filename, check_for_POTCAR=False).structure
    old_latt = origin_struct.lattice.as_dict(verbosity=1)
    old_latt = list(old_latt.values())[3:9]
    deform_struct = origin_struct
    delta, sigma = map(int, tolerance)
    Dlength = range(-delta, delta+1)
    Dangle = range(-sigma, sigma+1)
    lengths = old_latt[:3]
    angles = old_latt[3:]

    Pfile = 'Perturbe'
    if not os.path.exists(Pfile):
        os.makedirs(Pfile)

    for rnd in range(n_struct):
        lengths_loss = np.random.choice(Dlength, 3)/100
        lengths_loss = np.array(lengths)*lengths_loss
        angle_loss = np.random.choice(Dangle, 3)
        loss = np.append(lengths_loss, angle_loss)
        Perturbed_lattice = np.array(old_latt) - loss
        a, b, c, alpha, beta, gamma = Perturbed_lattice
        new_latt = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        deform_struct.lattice = new_latt
        out_poscar = Poscar(deform_struct, comment=str(rnd)+'_Perturbed structure')
        out_poscar.write_file(Pfile+'/perturbed_'+str(rnd)+'.vasp')

    return print("{} structures with random perturbation in lattice were generated! Bye!". format(size))


if __name__ == '__main__':
    #file = 'POSCAR'
    #tolerance = [5, 5]
    #size = 20
     if len(sys.argv) < 5:
         raise SystemError('Sytax Error! Run as python PerturbedStruct.py CONTCAR 5 5 3')

     file = sys.argv[1]
     tolerance = list(sys.argv[2:4])
     size = int(sys.argv[4])

     GRP_Structures(file, tolerance, size)
