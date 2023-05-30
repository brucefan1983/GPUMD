"""
    Purpose:
        Convert the runner training file (input.data) to xyz format (*.xyz).
        And be further converted into other format through the "dpdata" tool.
    Ref:
        dpdata: https://github.com/deepmodeling/dpdata
    Example:
        python runner2xyz.py input.data test.xyz 0
"""

import os
import sys
import numpy as np


chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']


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

# get the volume from cells.
def type2weight(types):
    mass_sum = 0
    for i in types:
        atom_name = i
        atom_index = chemical_symbols.index(i)
        atom_weight = atomic_masses[atom_index]
        #print(atom_name,atom_index,atom_weight)
        mass_sum += atom_weight
    return mass_sum

atomic_masses = np.array([1.0,1.008,4.002602,6.94,9.0121831,10.81,12.011,14.007,15.999,
18.998403163,20.1797,22.98976928,24.305,26.9815385,28.085,
30.973761998,32.06,35.45,39.948,39.0983,40.078,44.955908,47.867,
50.9415,51.9961,54.938044,55.845,58.933194,58.6934,63.546,65.38,
69.723,72.630,74.921595,78.971,79.904,83.798,85.4678,87.62,88.90584,
91.224,92.90637,95.95,97.90721,101.07,102.90550,106.42,107.8682,
112.414,114.818,118.710,121.760,127.60,126.90447,131.293,132.90545196,
137.327,138.90547,140.116,140.90766,144.242,144.91276,150.36,151.964,
157.25,158.92535,162.500,164.93033,167.259,168.93422,173.054,174.9668,
178.49,180.94788,183.84,186.207,190.23,192.217,195.084,196.966569,
200.592,204.38,207.2,208.98040,208.98243,209.98715,222.01758,223.01974,
226.02541,227.02775,232.0377,231.03588,238.02891,237.04817,244.06421,
243.06138,247.07035,247.07031,251.07959,252.0830,257.09511,258.09843,
259.1010,262.110,267.122,268.126,271.134,270.133,269.1338,278.156,281.165,
281.166,285.177,286.182,289.190,289.194,293.204,293.208,294.214])

# Load NEP dataset file (train.in)
# input: the path of train.in file.
def load_type(infile, map_name=None):

    data = {}
    data['infile'] = infile

    with open(data['infile'], 'r') as fraw:

        flines = fraw.readlines()

        # first: we should get the number of frames.
        data['nframe'] = 0

        lbegin = []
        lend = []
        for i, fli in enumerate(flines):
            if 'begin' in fli:
                data['nframe'] += 1
                lbegin.append(i)
            if 'end' in fli:
                lend.append(i)

        data['atom_numbs'] = np.zeros((data['nframe']))
        data['has_energy'] = np.zeros((data['nframe']))
        data['energies'] = np.zeros((data['nframe']))
        data['has_virial'] = np.zeros((data['nframe']))
        data['virials'] = np.zeros((data['nframe'], 6))
        data['has_cell'] = np.zeros((data['nframe']))
        data['cells'] = np.zeros((data['nframe'], 9))
        data['volume'] = np.zeros((data['nframe']))
        data['atom_names'] = {}
        data['atom_types'] = {}
        data['coords'] = {}
        data['forces'] = {}
        data['gross_mass'] = {}
        data['density'] = {}
        #print(lbegin, lend, data['nframe'])

        # second, get the number of atoms in each frame.
        for nfi in range(data['nframe']):
            box = []
            coord = []
            force = []
            atom_names = []
            atom_types = []
            volume = []
            energies = []
            atomnum = 0
            for li in range(lbegin[nfi]+1, lend[nfi]):
                if 'lattice' in flines[li]:
                    data['has_cell'][nfi] = 1
                    line = flines[li].split()[1:4]
                    box.append(list(map(float,line)))
                if flines[li].startswith('atom'):
                    xyz = flines[li].split()[1:4]
                    fxyz = flines[li].split()[7:10]
                    ele = flines[li].split()[4]
                    coord.append(list(map(float,xyz)))
                    force.append(list(map(float,fxyz)))
                    atom_names.append(ele)
                    if map_name != None:
                        atom_types.append(map_name.index(ele))
                    atomnum += 1
                if 'energy' in flines[li]:
                    weight = type2weight(atom_names)
                    data['gross_mass'][nfi] = weight
                    if data['has_virial'][nfi] == 1:
                        data['virials'][nfi] = np.zeros((6))
                    if data['has_cell'][nfi] == 1:
                        box = np.array(box).flatten()*Bohr2a
                        volume = vec2volume(box)
                        data['cells'][nfi] = box
                        data['volume'][nfi] = volume
                        data['density'][nfi] = weight/volume * 1.6605390666
                    data['has_energy'][nfi] = 1
                    energies = float(flines[li].split()[1])*Hartree2eV
                    data['energies'][nfi] = energies
                    data['atom_numbs'][nfi] = atomnum
                    # https://jerkwin.github.io/gmxtool/calc/calc.html
                    data['coords'][nfi] = np.array(coord)*Bohr2a
                    data['forces'][nfi] = np.array(force)*conforce
                    data['atom_names'][nfi] = atom_names
                    if map_name != None:
                        data['atom_types'][nfi] = atom_types
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
def print_xyz(data, folder, outfile='runner.xyz', shift_energy_peratom=0):

    fdataout = open('out.dat', 'w')
    print("#nfr ene gross_mass volume density", file=fdataout)
    print("# eV amu A3 g/cm3", file=fdataout)

    mark_energy = 0
    mark_natoms = 0
    fo = open(os.path.join(folder, outfile), 'w')
    for i in range(data['nframe']):

        xyz_max_min = [[10000000, -10000000],[10000000, -10000000],[10000000, -10000000]]
        Out_xyz = ""
        mark_energy += data['energies'][i]
        mark_natoms += data['atom_numbs'][i]
        shift_energy = shift_energy_peratom * data['atom_numbs'][i]
        for j in range(int(data['atom_numbs'][i])):
            for mi in range(3):
                if xyz_max_min[mi][0] > data['coords'][i][j][mi]:
                    xyz_max_min[mi][0] = data['coords'][i][j][mi]
                if xyz_max_min[mi][1] < data['coords'][i][j][mi]:
                    xyz_max_min[mi][1] = data['coords'][i][j][mi]
            Out_xyz += data['atom_names'][i][j] + " "
            Out_xyz += " ".join(list(map(str, data['coords'][i][j]))) + " "
            Out_xyz += " ".join(list(map(str, data['forces'][i][j]))) + "\n"

        Out_string = ""
        if data['has_virial'][i] == 1:
            Out_string += "virial=\"" + " ".join(list(map(str, myvirial))) + "\" "
        Out_string += str(int(data['atom_numbs'][i])) + "\n"
        if data['has_energy'][i] == 1:
            Out_string += "energy=" + str(data['energies'][i]-shift_energy) + " " # eV
        Out_string += "gross_mass=" + str(data['gross_mass'][i]) + " " # (amu)
        if data['has_cell'][i] == 1:
            Out_string += "Lattice=\"" + " ".join(list(map(str, data['cells'][i]))) + "\" " # A
            Out_string += "volume=" + str(data['volume'][i]) + " " # (A3)
            Out_string += "density=" + str(data['density'][i]) + " " # (g/cm3)
            print(i, data['energies'][i]-shift_energy, data['gross_mass'][i], data['volume'][i], data['density'][i], file=fdataout)
        else:
            lx = xyz_max_min[0][1] - xyz_max_min[0][0] + 20
            ly = xyz_max_min[1][1] - xyz_max_min[1][0] + 20
            lz = xyz_max_min[2][1] - xyz_max_min[2][0] + 20
            Out_string += f"Lattice=\"{lx} 0 0 0 {ly} 0 0 0 {lz}\" " # A
            print(i, data['energies'][i]-shift_energy, data['gross_mass'][i], file=fdataout)
        Out_string += "config_type=runner2xyz "
        Out_string += "pbc=\"T T T\" "
        Out_string += f"shift_energy=\"{shift_energy}\" "
        Out_string += "Properties=species:S:1:pos:R:3:force:R:3\n"
        Out_string += Out_xyz

        fo.write(Out_string)
    fo.close()
    fdataout.close()
    print(f"Each atom energy:{mark_energy/mark_natoms}")


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
    # Units:
    # https://compphysvienna.github.io/n2p2/interfaces/pair_nnp.html#pair-nnp
    # File format:
    # https://compphysvienna.github.io/n2p2/topics/cfg_file.html#cfg-file
    global Bohr2a, Hartree2eV, conforce
    Bohr2a = 1/1.8897261328
    Hartree2eV = 1/0.0367493254
    conforce = Hartree2eV/Bohr2a

    instr = sys.argv[1]
    outstr = sys.argv[2]
    energy_peratom = float(sys.argv[3])

    data = load_type('./'+instr)  # n2p2 file name
    print_xyz(data, './xyz', outfile=outstr, shift_energy_peratom=energy_peratom)
    #conver2deepmd(instr)


if __name__ == "__main__":
    main()
