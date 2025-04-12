"""
    Purpose:
        Convert deepmd input file format to xyz.
    Ref:
        dpdata: https://github.com/deepmodeling/dpdata
    Run:
        python deep2xyz.py deepmd-npy nepxyz-from-npy
        python deep2xyz.py deepmd-mixed nepxyz-from-mixed
"""

import os
import sys
import glob
import numpy as np

ELEMENTS=['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', \
         'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',\
         'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',\
         'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', \
         'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']

def vec2volume(cells):
    va = cells[:3]
    vb = cells[3:6]
    vc = cells[6:]
    return np.dot(va, np.cross(vb,vc))

def cond_load_data(fname) :
    tmp = None
    if os.path.isfile(fname) :
        tmp = np.load(fname)
    return tmp

def load_type(folder, type_map=None) :
    data = {}
    data['my_type_map'] = None
    data['atom_types'] \
        = np.loadtxt(os.path.join(folder, 'type.raw'), ndmin=1).astype(int)
    ntypes = np.max(data['atom_types']) + 1
    data['atom_numbs'] = []
    for ii in range (ntypes) :
        data['atom_numbs'].append(np.count_nonzero(data['atom_types'] == ii))
    data['atom_names'] = []
    # if find type_map.raw, use it
    if os.path.isfile(os.path.join(folder, 'type_map.raw')) :
        with open(os.path.join(folder, 'type_map.raw')) as fp:
            my_type_map = fp.read().split()
        data['my_type_map'] = my_type_map
    # else try to use arg type_map
    elif type_map is not None:
        my_type_map = type_map
    # in the last case, make artificial atom names
    else:
        my_type_map = []
        for ii in range(ntypes) :
            my_type_map.append('Type_%d' % ii)
    assert(len(my_type_map) >= len(data['atom_numbs']))
    for ii in range(len(data['atom_numbs'])) :
        data['atom_names'].append(my_type_map[ii])

    return data

def load_set(folder):
    cells  = np.load(os.path.join(folder, 'box.npy'))
    coords = np.load(os.path.join(folder, 'coord.npy'))
    eners  = cond_load_data(os.path.join(folder, 'energy.npy'))
    forces = cond_load_data(os.path.join(folder, 'force.npy'))
    virs   = cond_load_data(os.path.join(folder, 'virial.npy'))
    set_types = cond_load_data(os.path.join(folder, 'real_atom_types.npy'))
    return cells, coords, eners, forces, virs, set_types


def to_system_data(folder):
    # data is empty
    data = load_type(folder) # define type_map (ex. ['O', 'H']) if there isnot type_map.raw file.
    data['orig'] = np.zeros([3])
    data['docname'] = folder
    sets = sorted(glob.glob(os.path.join(folder, 'set.*')))
    all_cells = []
    all_coords = []
    all_eners = []
    all_forces = []
    all_virs = []
    real_set_types = []
    all_nframes = 0
    for ii in sets:
        cells, coords, eners, forces, virs, set_types = load_set(ii)
        # (2000, 9) (2000, 30) (2000,) (2000, 30) (2000, 9) (2000, 10)
        nframes = np.reshape(cells, [-1,3,3]).shape[0]
        all_nframes += nframes
        all_cells.append(np.reshape(cells, [nframes,3,3]))
        all_coords.append(np.reshape(coords, [nframes,-1,3]))
        if eners is not None:
            eners = np.reshape(eners, [nframes])
        if eners is not None and eners.size > 0:
            all_eners.append(np.reshape(eners, [nframes]))
        if forces is not None and forces.size > 0:
            all_forces.append(np.reshape(forces, [nframes,-1,3]))
        if virs is not None and virs.size > 0:
            all_virs.append(np.reshape(virs, [nframes,9]))
        if set_types is not None and set_types.size > 0:
            real_set_types.append(np.reshape(set_types, [nframes, -1]))
    data['frames'] = all_nframes
    data['cells'] = np.concatenate(all_cells, axis = 0)
    data['coords'] = np.concatenate(all_coords, axis = 0)
    if len(all_eners) > 0 :
        data['energies'] = np.concatenate(all_eners, axis = 0)
    if len(all_forces) > 0 :
        data['forces'] = np.concatenate(all_forces, axis = 0)
    if len(all_virs) > 0:
        data['virials'] = np.concatenate(all_virs, axis = 0)
    if len(real_set_types) > 0:
        data['set_types'] = np.concatenate(real_set_types, axis = 0)
    if os.path.isfile(os.path.join(folder, "nopbc")):
        data['nopbc'] = True
    return data

def read_multi_deepmd(folder):

    data_multi = {}

    list_dir = []
    for dirpath, filedir, filename in os.walk(folder):
        if 'type.raw' in filename:
            list_dir.append(dirpath)
    #print(list_dir)

    for i, fi in enumerate(list_dir):
        ifold = fi
        idata = to_system_data(ifold)
        if 'virials' in idata and len(idata['virials']) == idata['frames']:
            idata['has_virial'] = np.ones((idata['frames']), dtype=bool)
        else:
            idata['has_virial'] = np.zeros((idata['frames']), dtype=bool)
        if 'set_types' in idata and len(idata['set_types']) == idata['frames']:
            idata['has_set_types'] = np.ones((idata['frames']), dtype=bool)
        else:
            idata['has_set_types'] = np.zeros((idata['frames']), dtype=bool)
        data_multi[i] = idata

    nframes = np.sum([data_multi[i]['frames'] for i in data_multi])

    data = {}
    data['nframe'] = nframes
    data['docname'] = ['' for i in range(nframes)]
    data['atom_numbs'] = np.zeros((data['nframe']))
    data['has_virial'] = np.zeros((data['nframe']))
    data['virials'] = np.zeros((data['nframe'], 9))
    data['has_set_types'] = np.zeros((data['nframe']))
    data['energies'] = np.zeros((data['nframe']))
    data['cells'] = np.zeros((data['nframe'], 9))
    data['volume'] = np.zeros((data['nframe']))
    data['type_maps'] = {}
    data['atom_names'] = {}
    data['atom_types'] = {}
    data['coords'] = {}
    data['forces'] = {}
    data['set_types'] = {}

    ifr = -1
    for i in data_multi:

        atom_names = [data_multi[i]['atom_names'][j] for j in data_multi[i]['atom_types']]

        for j in range(data_multi[i]['frames']):

            ifr += 1
            data['atom_numbs'][ifr] = len(data_multi[i]['atom_types'])
            data['has_virial'][ifr] = data_multi[i]['has_virial'][j]
            data['has_set_types'][ifr] = data_multi[i]['has_set_types'][j]
            data['energies'][ifr] = data_multi[i]['energies'][j]
            if data['has_virial'][ifr]:
                data['virials'][ifr] = data_multi[i]['virials'][j]
            if data['has_set_types'][ifr]:
                data['set_types'][ifr] = data_multi[i]['set_types'][j]
            data['cells'][ifr] = np.reshape(data_multi[i]['cells'][j],9)
            data['volume'][ifr] = vec2volume(data['cells'][ifr])
            data['type_maps'][ifr] = data_multi[i]['my_type_map']
            data['atom_names'][ifr] = atom_names
            data['atom_types'][ifr] = data_multi[i]['atom_types']
            data['coords'][ifr] = data_multi[i]['coords'][j]
            data['forces'][ifr] = data_multi[i]['forces'][j]
            data['docname'][ifr] = data_multi[i]['docname']

    return data


def check_data(data):

    print('Nframes:', data['nframe'])
    for i in range(data['nframe']):
        print(i, data['docname'][i])

        print('    atom_numbs', int(data['atom_numbs'][i]), end=' ')
        #print('has_virial', int(data['has_virial'][i]))
        #print('energies', data['energies'][i])
        #print('virials', len(data['virials'][i]))
        #print('volume', data['volume'][i])
        #print('cells', len(data['cells'][i]))
        print('atom_types', len(data['atom_types'][i]))
        print('    coords', len(data['coords'][i]), end=' ')
        print('forces', len(data['forces'][i]))


def check_type(data, fram_id):

    print('Nframes:', data['nframe'], fram_id)
    for i in [fram_id]:
        print(i, data['docname'][i])
        for j in range(int(data['atom_numbs'][i])):
            print(data['type_maps'][i][data['set_types'][i][j]], end=" ")
        print()


def dump_xyz(folder, data, outxyz="DEEP2XYZ.xyz", dvi=1):

    os.makedirs(folder, exist_ok = True)
    Out_string = ""
    for i in range(data['nframe']):

        if i%dvi != 0: continue

        Out_string += str(int(data['atom_numbs'][i])) + "\n"
        myvirial = data['virials'][i]
        Out_string += "energy=" + str(data['energies'][i]) + " "
        Out_string += "config_type=nep2xyz "
        Out_string += "pbc=\"T T T\" "

        if data['has_virial'][i]:
            Out_string += "virial=\"" + " ".join(list(map(str, myvirial))) + "\" "
        Out_string += "Lattice=\"" + " ".join(list(map(str, data['cells'][i]))) + "\" "
        Out_string += f"volume={data['volume'][i]} "
        Out_string += "Properties=species:S:1:pos:R:3:force:R:3\n"

        if data['has_set_types'][i]:
            for j in range(int(data['atom_numbs'][i])):
                Out_string += data['type_maps'][i][data['set_types'][i][j]] + " "
                Out_string += " ".join(list(map(str, data['coords'][i][j]))) + " "
                Out_string += " ".join(list(map(str, data['forces'][i][j]))) + "\n"
        else:
            for j in range(int(data['atom_numbs'][i])):
                Out_string += data['atom_names'][i][j] + " "
                Out_string += " ".join(list(map(str, data['coords'][i][j]))) + " "
                Out_string += " ".join(list(map(str, data['forces'][i][j]))) + "\n"

    fo = open(os.path.join(folder, outxyz), 'w')
    fo.write(Out_string)
    fo.close()


def main():

    instr = sys.argv[1]
    outstr = sys.argv[2]
    dvi_frame = 1
    data = read_multi_deepmd(instr)
    # fram_id = int(sys.argv[3])
    # check_type(data, fram_id)
    dump_xyz(outstr, data, outxyz="train.xyz", dvi=dvi_frame)


if __name__ == "__main__":
    main()
