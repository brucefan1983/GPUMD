"""
    Purpose:
        Convert mtp input file format to xyz.
    Ref:
        dpdata: https://github.com/deepmodeling/dpdata
    Run:
        python mtp2xyz.py train.cfg Symbol1 Symbol2 Symbol3 ...
"""

import os
import sys
import numpy as np
from ase.atoms import Atoms
from collections import defaultdict


def load_cfg(filename, type_to_symbol):
    frames = []
    with open(filename) as f:
        line = 'chongchongchong!'
        while line:
            line = f.readline()
            if 'BEGIN_CFG' in line:
                cell = np.zeros((3, 3))
            if 'Size' in line:
                line = f.readline()
                natoms = int(line.split()[0])
                positions = np.zeros((natoms, 3))
                forces = np.zeros((natoms, 3))
                energies = np.zeros(natoms)
                symbols = ['X'] * natoms
            if 'Supercell' in line: 
                for i in range(3):
                    line = f.readline()
                    for j in range(3):
                        cell[i, j] = float(line.split()[j])
            if 'AtomData' in line:
                d = defaultdict(int)
                for (i, x) in enumerate(line.split()[1:]):
                    d[x] = i

                for _ in range(natoms):
                    line = f.readline()
                    fields = line.split()
                    i = int(fields[d['id']]) - 1
                    symbols[i] = type_to_symbol[int(fields[d['type']])]
                    positions[i] = [float(fields[d[attr]]) for attr in ['cartes_x', 'cartes_y' ,'cartes_z']]
                    forces[i] = [float(fields[d[attr]]) for attr in ['fx', 'fy' ,'fz']]
                    energies[i] = float(fields[d['site_en']])

                atoms = Atoms(symbols=symbols, cell=cell, positions=positions, pbc=True)
                if d['fx'] != 0:
                    atoms.info['forces'] = forces
                if d['site_en'] != 0:
                    atoms.info['energies'] = energies

            if 'Energy' in line and 'Weight' not in line:
                line = f.readline()
                atoms.info['energy'] = float(line.split()[0])
            if 'PlusStress' in line:
                line = f.readline()
                plus_stress = np.array(list(map(float, line.split())))
                atoms.info['virial'] = plus_stress
            if 'END_CFG' in line:
                frames.append(atoms)
            if 'EnergyWeight' in line:
                line = f.readline()
                atoms.info['energy_weight'] = float(line.split()[0])
            if 'identification' in line:
                atoms.info['identification'] = int(line.split()[2])

    return frames


def dump_nep(frames):
    with open('train.in', 'w') as f:
        n_frames = len(frames)
        f.write(str(n_frames) + '\n')
        for atoms in frames:
            has_virial = int('virial' in atoms.info)
            f.write('{} {} \n'.format(len(atoms), has_virial))
        for atoms in frames:
            ret = str(atoms.info['energy'])
            if 'virial' in atoms.info:
                for v in atoms.info['virial'][[0, 1, 2, 5, 3, 4]]:
                    ret += ' ' + str(v)
            ret += '\n{:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e}\n'.format(*atoms.get_cell().reshape(-1))
            s = atoms.get_chemical_symbols()
            p = atoms.get_positions()
            forces = atoms.info['forces']
            for i in range(len(atoms)):
                ret += '{:2} {:>15.8e} {:>15.8e} {:>15.8e} {:>15.8e} {:>15.8e} {:>15.8e}\n'.format(s[i], *p[i], *forces[i])
            f.write(ret)


def dump_xyz(frames):

    Out_string = ""
    n_frames = len(frames)
    for atoms in frames:
        Out_string += str(len(atoms)) + '\n'
        Out_string += "energy=" + str(atoms.info['energy']) + " "
        Out_string += "config_type=mtp2nep "
        Out_string += "pbc=\"T T T\" "
        Out_string += "virial=\"" + " ".join(list(map(str, atoms.info['virial']))) + "\" "
        Out_string += "Lattice=\"" + " ".join(list(map(str, atoms.get_cell().reshape(-1)))) + "\" "
        Out_string += "Properties=species:S:1:pos:R:3:force:R:3\n"

        s = atoms.get_chemical_symbols()
        p = atoms.get_positions()
        forces = atoms.info['forces']
        for i in range(len(atoms)):
            Out_string += '{:2} {:>15.8e} {:>15.8e} {:>15.8e} {:>15.8e} {:>15.8e} {:>15.8e}\n'.format(s[i], *p[i], *forces[i])

    os.makedirs("XYZ", exist_ok=True)
    fo = open(os.path.join("XYZ", 'mtp2xyz.xyz'), 'w')
    fo.write(Out_string)
    fo.close()


if __name__ == "__main__":
    type_to_symbol = {i: s for i, s in enumerate(sys.argv[2:])}
    frames = load_cfg(sys.argv[1], type_to_symbol)
    #dump_nep(frames)
    dump_xyz(frames)
