"""
    Purpose:
        Convert the nep training file (train.in) to xyz format (*.xyz).
        And be further converted into other format through the "dpdata" tool.
    Ref:
        dpdata: https://github.com/deepmodeling/dpdata
    Run:
        python nep2xyz.py indoc
    Author:
        Ke Xu <twtdq(at)qq.com>
"""

import os
import sys
import numpy as np


ELEMENTS=['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', \
         'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',\
         'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',\
         'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', \
         'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']


# The virial force in NEP is a unit of "eV",
#     which is converted to "bar" by dividing by the volume.
def eV_volume2bar(nep_virial, volume):
    eV_A2N = 1.6021773e-9
    A2m = 1e-10
    eV_A32bar = eV_A2N/A2m/A2m * 1e-5
    return nep_virial/volume*eV_A32bar

# get the volume from cells.
def vec2volume(cells):
    va = cells[:3]
    vb = cells[3:6]
    vc = cells[6:]
    return np.dot(va, np.cross(vb,vc))


# Load NEP dataset file (train.in)
# input: the path of train.in file.
def load_type(folder):

    data = {}
    data['infile'] = os.path.join(folder, 'train.in')

    with open(data['infile'], 'r') as fraw:

        flines = fraw.readlines()

        # first: we should get the number of frames.
        data['nframe'] = int(flines[0])
        data['atom_numbs'] = np.zeros((data['nframe']))
        data['has_virial'] = np.zeros((data['nframe']))
        data['energies'] = np.zeros((data['nframe']))
        data['virials'] = np.zeros((data['nframe'], 6))
        data['cells'] = np.zeros((data['nframe'], 9))
        data['volume'] = np.zeros((data['nframe']))
        data['atom_names'] = {}
        data['coords'] = {}
        data['forces'] = {}

        # second, get the number of atoms in each frame.
        for i in range(data['nframe']):
            data['atom_numbs'][i] = int(flines[i+1].split()[0])
            data['has_virial'][i] = int(flines[i+1].split()[1])

        # # check the file lines.
        data['nlines'] = np.sum(data['atom_numbs'])+data['nframe']*3+1
        if data['nlines'] != len(flines):
            print('Need lines:', data['nlines'], 'len of file:', len(flines))
            raise "Please check your train.in file."

        nstart = data['nframe']+1
        for i in range(data['nframe']):

            # third, get the cells and the volume of each frame.
            # No volume, no unit conversion of the virials.
            cnline = int(2*i + sum(data['atom_numbs'][:i]) + nstart + 1) # enline+1
            data['cells'][i] = np.array(list(map(float, flines[cnline].split())))
            volume = vec2volume(data['cells'][i])
            data['volume'][i] = volume

            # fourth, get the energy and virials(if this frame has virial).
            enline = int(2*i + sum(data['atom_numbs'][:i]) + nstart)
            data['energies'][i] = float(flines[enline].split()[0])
            if data['has_virial'][i] == 1:
                nep_virial = np.array(list(map(float, flines[enline].split()[1:])))
                data['virials'][i] = nep_virial  # the units of virial in xyz is eV.
                #data['virials'][i] = eV_volume2bar(nep_virial, volume)  # the units of virial in xyz is bar.
            elif data['has_virial'][i] == 0:
                data['virials'][i] = np.zeros((6))

            atom_type = []
            atom_coor = []
            atom_forc = []
            anline = int(2*i + sum(data['atom_numbs'][:i]) + nstart + 2) # enline+2
            # fifth, get the coords and forces of each atom in the frame.
            for j in range(int(data['atom_numbs'][i])):
                tnline = anline + j
                atom_type.append(str(flines[tnline].split()[0]))
                atom_coor.append(np.array(list(map(float, flines[tnline].split()[1:4]))))
                atom_forc.append(np.array(list(map(float, flines[tnline].split()[4:]))))
            data['atom_names'][i] = atom_type
            data['coords'][i] = atom_coor
            data['forces'][i] = atom_forc

    return data


# There are only six values in NEP, so it needs to be expanded
def convervirial(invirial):

    vxx = invirial[0]
    vyy = invirial[1]
    vzz = invirial[2]
    vxy = invirial[3]
    vyz = invirial[4]
    vzx = invirial[5]

    return [vxx, vxy, vzx, vxy, vyy, vyz, vzx, vyz, vzz]


# output
def print_xyz(data, folder):

    Out_string = ""
    for i in range(data['nframe']):

        if data['has_virial'][i] == 1: continue
        Out_string += str(int(data['atom_numbs'][i])) + "\n"
        Out_string += "energy=" + str(data['energies'][i]) + " "
        Out_string += "config_type=nep2xyz "
        Out_string += "pbc=\"T T T\" "
        Out_string += "Lattice=\"" + " ".join(list(map(str, data['cells'][i]))) + "\" "
        Out_string += "Properties=species:S:1:pos:R:3:force:R:3\n"

        for j in range(int(data['atom_numbs'][i])):

            Out_string += data['atom_names'][i][j] + " "
            Out_string += " ".join(list(map(str, data['coords'][i][j]))) + " "
            Out_string += " ".join(list(map(str, data['forces'][i][j]))) + "\n"

    fo = open(os.path.join(folder, 'novirial.xyz'), 'w')
    fo.write(Out_string)
    fo.close()

    if sum(data['has_virial']) == 0: return 1

    Out_string = ""
    for i in range(data['nframe']):

        if data['has_virial'][i] == 0: continue
        Out_string += str(int(data['atom_numbs'][i])) + "\n"
        myvirial = convervirial(data['virials'][i])   
        Out_string += "energy=" + str(data['energies'][i]) + " "
        Out_string += "config_type=nep2xyz "
        Out_string += "pbc=\"T T T\" "
        Out_string += "virial=\"" + " ".join(list(map(str, myvirial))) + "\" "
        Out_string += "Lattice=\"" + " ".join(list(map(str, data['cells'][i]))) + "\" "
        Out_string += "Properties=species:S:1:pos:R:3:force:R:3\n"

        for j in range(int(data['atom_numbs'][i])):

            Out_string += data['atom_names'][i][j] + " "
            Out_string += " ".join(list(map(str, data['coords'][i][j]))) + " "
            Out_string += " ".join(list(map(str, data['forces'][i][j]))) + "\n"

    fo = open(os.path.join(folder, 'hasvirial.xyz'), 'w')
    fo.write(Out_string)
    fo.close()


# If you have installed dpdata,
# you can convert xyz to training set in other formats,
# such as "deepmd"
def conver2deepmd(instr):
    import dpdata
    xyz_multi_systems = dpdata.MultiSystems.from_file(file_name='./'+instr+'/novirial.xyz',fmt='quip/gap/xyz')
    xyz_multi_systems.to_deepmd_raw('./deepmd_data'+instr+'/novirial')
    xyz_multi_systems = dpdata.MultiSystems.from_file(file_name='./'+instr+'/hasvirial.xyz',fmt='quip/gap/xyz')
    xyz_multi_systems.to_deepmd_raw('./deepmd_data'+instr+'/hasvirial')


def main():

    instr = sys.argv[1]

    data = load_type('./'+instr)
    print_xyz(data, './'+instr)

    conver2deepmd(instr)


if __name__ == "__main__":
    main()
