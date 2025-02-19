import sys
import numpy as np
from ase.io import read
from scipy.spatial.distance import pdist

# Read the file name from command line arguments
file_name = sys.argv[1]

# Read all frames from the extxyz file
frames = read(file_name, index=':')

min_distance = float('inf')

# Iterate over each frame
for frame in frames:
    # Get atomic positions
    positions = frame.get_positions()
    # Compute distances between all pairs of atoms
    distances = pdist(positions)
    # Update the minimum distance
    min_distance = min(min_distance, np.min(distances))

print(f' Minimum interatomic distance: {min_distance} Å')
