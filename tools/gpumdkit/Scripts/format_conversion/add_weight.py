"""
This script is used to add a weight value to the 'Weight' attribute of each structure in an input file and save the modified structures to an output file.

Usage:
    python add_weight.py <input_file> <output_file> <new_weight>

Example:
    python add_weight.py input.xyz output.xyz 5
"""

import os
import sys
from ase.io import read, write

def main():
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_file> <output_file> <new_weight>")
        sys.exit(1)

    # Parse command line arguments
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    try:
        new_weight = float(sys.argv[3])
    except ValueError:
        print("Error: The third argument (new weight) must be a number.")
        sys.exit(1)

    # Read all structures from the input file
    try:
        structures = read(input_file, index=':')
    except Exception as e:
        print(f"Error: Could not read the file {input_file}. Reason: {e}")
        sys.exit(1)

    # Modify the 'Weight' in the info dictionary for each structure
    for structure in structures:
        structure.info['Weight'] = new_weight

    # Write the modified structures to the output file
    try:
        write(output_file, structures)
    except Exception as e:
        print(f"Error: Could not write to the file {output_file}. Reason: {e}")
        sys.exit(1)

    # Print success message
    print(f"Weights in {input_file} have been updated and saved to {output_file}")

if __name__ == "__main__":
    main()

