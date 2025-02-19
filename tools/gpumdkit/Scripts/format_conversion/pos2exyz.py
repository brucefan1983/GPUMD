import sys
from ase.io import read, write

def convert_poscar_to_extxyz(poscar_filename, extxyz_filename):
    """
    Convert POSCAR file(s) to extxyz format.

    Parameters:
    - poscar_filename (str): Name of the POSCAR file to read.
    - extxyz_filename (str): Name of the output extxyz file.
    """

    # Read the structure from POSCAR file
    frames = read(poscar_filename, format='vasp')

    # Write frames to extxyz file
    write(extxyz_filename, frames)

if __name__ == '__main__':
    # Check if the number of arguments is correct
    if len(sys.argv) < 3:
        print(" Usage: python script_name.py <POSCAR_filename> <extxyz_filename>")
        sys.exit(1)

    poscar_filename = sys.argv[1]
    extxyz_filename = sys.argv[2]

    # Call function to convert POSCAR to extxyz
    convert_poscar_to_extxyz(poscar_filename, extxyz_filename)
