import sys
from ase.io import read, write
import numpy as np

# Read the file name from command line arguments
file_name = sys.argv[1]

# Read the elements order from command line arguments
elements = sys.argv[2:]

# Read the atomic data from the file
atoms = read(file_name)

# Create groups corresponding to the elements
groups = []
for element in atoms.get_chemical_symbols():
    if element in elements:
        group = elements.index(element)
    else:
        raise ValueError(f"Element {element} not found in the provided elements list")
    groups.append(group)

# Add the group information to the atomic data
groups_array = np.array(groups)
atoms.new_array("group", groups_array)

# Write the output to a file
write("model.xyz", atoms)
