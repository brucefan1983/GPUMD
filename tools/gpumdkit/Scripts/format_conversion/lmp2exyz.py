import sys
from ase.io import read, write

def lmp2exyz(dump_file, elements):
    # Read LAMMPS dump file
    frames = read(dump_file, format='lammps-dump-text', index=':')
    
    # Ensure all atomic types are within the provided elements range
    for frame in frames:
        unique_types = set(frame.get_atomic_numbers())
        if max(unique_types) > len(elements):
            raise ValueError(f" Found atomic type {max(unique_types)} which is out of the provided elements range.")
    
    # Map atomic types to specified elements
    type_to_element = {i+1: elements[i] for i in range(len(elements))}
    
    # Convert atomic types to specified elements
    for frame in frames:
        new_symbols = [type_to_element[number] for number in frame.get_atomic_numbers()]
        frame.set_chemical_symbols(new_symbols)
    
    # Write to extxyz file
    extxyz_file = 'dump.xyz'
    write(extxyz_file, frames, format='extxyz')
#   print(f" Converted {dump_file} to {extxyz_file}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(" Usage: python lmpdump2exyz.py <dump_file> <element1> <element2> ...")
        sys.exit(1)
    
    dump_file = sys.argv[1]
    elements = sys.argv[2:]
    
    lmp2exyz(dump_file, elements)
