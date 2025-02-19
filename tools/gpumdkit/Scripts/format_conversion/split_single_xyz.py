import sys
from ase.io import read, write

def split_xyz_to_individual_models(input_xyz_filename):
    """
    Split an XYZ file into individual model files.

    Parameters:
    - input_xyz_filename (str): Name of the input XYZ file to read.

    Reads the input XYZ file, splits each frame into individual files named model_${i}.xyz.
    """

    # Read all frames from input XYZ file
    frames = read(input_xyz_filename, format='extxyz', index=':')

    # Iterate over each frame and write to separate model files
    for i, frame in enumerate(frames):
        model_filename = f'model_{i + 1}.xyz'  
        write(model_filename, frame, format='extxyz')

    print(f' All frames from "{input_xyz_filename}" have been split into individual model files.')

if __name__ == '__main__':
    # Check if the number of arguments is correct
    if len(sys.argv) < 2:
        print(" Usage: python script_name.py <input_xyz_filename>")
        sys.exit(1)

    # Get input XYZ filename from command line argument
    input_xyz_filename = sys.argv[1]

    # Call function to split XYZ file into individual models
    split_xyz_to_individual_models(input_xyz_filename)
