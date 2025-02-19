#!/usr/bin/env python
# coding: utf-8
'''
Run example:
    $ add_groups.py block INF INF INF INF INF 5 4
    Group atoms with Z coordinates between INF and 5, assigning them to group 4.
    $ add_groups.py union 0-3 4 5-6 7
    Change groups 0, 1, 2, 3, 4, 5, and 6 to group 7.
    $ add_groups.py elements
    Automatically group atoms by element type, starting group numbers from 0.
    $ add_groups.py direction x 8
    Divide atoms into 8 groups along the x direction.
    $ add_groups.py all 0
    Set the group of all atoms to 0.  
    $ add_groups.py cylinder z 0 0 5 INF INF 1
    Group atoms within a cylinder along the z-axis with radius 5, centered at (0,0), and height from INF to INF, assigning them to group 1.    
    $ add_groups.py id
    Assign a unique group ID to each atom, starting from 0.
Contributors:
    Yuwen Zhang, E-mail:984307703@qq.com
'''

import sys
from ase.io import read, write
import numpy as np
from sys import argv

def group_atoms_by_region(atoms, x_min, x_max, y_min, y_max, z_min, z_max, group_id):    
    for i,atom in enumerate(atoms):
        x, y, z = atom.position
        if (x_min == 'INF' or x >= float(x_min)) and (x_max == 'INF' or x <= float(x_max)) and \
           (y_min == 'INF' or y >= float(y_min)) and (y_max == 'INF' or y <= float(y_max)) and \
           (z_min == 'INF' or z >= float(z_min)) and (z_max == 'INF' or z <= float(z_max)):
            group_ids[i] = group_id
    atoms.set_array('group', group_ids)
    return atoms

def parse_group_ids(group_ids_str):
    group_ids = []
    for part in group_ids_str:
        if '-' in part:
            start, end = map(int, part.split('-'))
            group_ids.extend(range(start, end + 1))
        else:
            group_ids.append(int(part))
    return group_ids
    

def union_groups(atoms, group_ids_str, new_group_id):
    group_ids = parse_group_ids(group_ids_str)
    current_groups = atoms.get_array('group')
    unique_groups = set(current_groups)
    # 检查是否有不存在的组编号
    for group_id in group_ids:
        if group_id not in unique_groups:
            raise ValueError(f"Group ID {group_id} does not exist in the current groups.")
    
    for i, group_id in enumerate(current_groups):
        if group_id in group_ids:
            current_groups[i] = new_group_id
    atoms.set_array('group', current_groups)
    return atoms
    
def group_atoms_by_elements(atoms):
    unique_elements = list(set(atoms.get_chemical_symbols()))
    element_to_group = {element: i for i, element in enumerate(unique_elements)}
    group_ids = np.zeros(len(atoms), dtype=int)
    for i, atom in enumerate(atoms):
        group_ids[i] = element_to_group[atom.symbol]
    atoms.set_array('group', group_ids)

    # 打印元素的分组信息
    for element, group_id in element_to_group.items():
        print(f"Element {element} is assigned to group {group_id}")

    return atoms
def modify_properties_line(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    with open(file_path, 'w') as file:
        for line in lines:
            line = line.replace('properties', 'Properties')
            file.write(line)
def group_atoms_by_direction(atoms, direction, num_groups):
    positions = atoms.get_positions()
    min_coord = np.min(positions[:, direction])
    max_coord = np.max(positions[:, direction])
    group_ids = np.zeros(len(atoms), dtype=int)
    interval = (max_coord - min_coord) / num_groups    
    for i, atom in enumerate(atoms):
        coord = atom.position[direction]
        group_ids[i] = min(int((coord - min_coord) // interval), num_groups - 1)
    
    atoms.set_array('group', group_ids)
    return atoms
def group_atoms_by_cylinder(atoms, dim, c1, c2, radius, lo, hi, group_id):
    dim_map = {'x': 0, 'y': 1, 'z': 2}
    axis = dim_map[dim]    
    for i, atom in enumerate(atoms):
        pos = atom.position
        
        if (lo == 'INF' or pos[axis] >= float(lo)) and (hi == 'INF' or pos[axis] <= float(hi)):
            print (lo,hi)
            if axis == 0:
                dist = np.sqrt((pos[1] - c1)**2 + (pos[2] - c2)**2)
            elif axis == 1:
                dist = np.sqrt((pos[0] - c1)**2 + (pos[2] - c2)**2)
            else:
                dist = np.sqrt((pos[0] - c1)**2 + (pos[1] - c2)**2)
            if dist <= radius:
                group_ids[i] = group_id
    
    atoms.set_array('group', group_ids)
    return atoms
    
    
def set_all_atoms_group(atoms, group_id):
    group_ids = np.full(len(atoms), group_id, dtype=int)
    atoms.set_array('group', group_ids)
    return atoms
def assign_unique_group_ids(atoms):
    group_ids = np.arange(len(atoms), dtype=int)
    atoms.set_array('group', group_ids)
    return atoms

def print_help():
    print('''Run example:
    add_groups.py block INF INF INF INF INF 5 4
    Group atoms with Z coordinates between INF and 5, assigning them to group 4.
    add_groups.py union 0-3 4 5-6 7
    Change groups 0, 1, 2, 3, 4, 5, and 6 to group 7.
    add_groups.py elements
    Automatically group atoms by element type, starting group numbers from 0.
    add_groups.py direction x 8
    Divide atoms into 8 groups along the x direction.
    add_groups.py all 0
    Set the group of all atoms to 0.
    add_groups.py cylinder z 0 0 5 INF INF 1
    Group atoms within a cylinder along the z-axis with radius 5, centered at (0,0), and height from INF to INF, assigning them to group 1.
    add_groups.py id
    Assign a unique group ID to each atom, starting from 0.''')    
    
modify_properties_line('./model.xyz')            
atoms = read('model.xyz')

# 检查是否已有分组信息
if 'group' in atoms.arrays:
    group_ids = atoms.get_array('group')
else:
    group_ids = np.zeros(len(atoms), dtype=int)
try:
    method = argv[1]
    if method == 'block':   
        x_min = argv[2];x_max = argv[3]
        y_min = argv[4];y_max = argv[5]
        z_min = argv[6];z_max = argv[7]
        group_id = int(argv[8])
        atoms = group_atoms_by_region(atoms, x_min, x_max, y_min, y_max, z_min, z_max, group_id)    
    elif method == 'union':
        group_ids_to_union = argv[2:-1]
        new_group_id = int(argv[-1])
        atoms = union_groups(atoms, group_ids_to_union, new_group_id)
    elif method == 'elements':
        atoms = group_atoms_by_elements(atoms)
    elif method == 'direction':
        direction = argv[2]
        num_groups = int(argv[3])
        direction_map = {'x': 0, 'y': 1, 'z': 2}
        if direction in direction_map:
            atoms = group_atoms_by_direction(atoms, direction_map[direction], num_groups)
        else:
            raise ValueError(f"Invalid direction: {direction}")
    elif method == 'cylinder':
        dim = argv[2]
        c1 = float(argv[3])
        c2 = float(argv[4])
        radius = float(argv[5])
        lo = argv[6]
        hi = argv[7]
        group_id = int(argv[8])
        atoms = group_atoms_by_cylinder(atoms, dim, c1, c2, radius, lo, hi, group_id)

    elif method == 'all':
        group_id = int(argv[2])
        atoms = set_all_atoms_group(atoms, group_id)
    elif method == 'id':
        atoms = assign_unique_group_ids(atoms)
    else:
        raise ValueError(f"Invalid method: {method}")
    write("model.xyz", atoms)
except ValueError as e:
    print(e)
    print_help()
except Exception as e:
    print(f"An error occurred: {e}")
    print_help()