import sys
import numpy as np
from ase.io import read, write
from scipy.spatial.distance import pdist

def print_progress_bar(iteration, total, length=50):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r Progress: |{bar}| {percent}% Complete', end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()

# Read the file name from command line arguments
file_name = sys.argv[1]

# Check if the distance threshold is provided
if len(sys.argv) > 2:
    distance_threshold = float(sys.argv[2])
else:
    distance_threshold = None

# Read all frames from the extxyz file
frames = read(file_name, index=':')

total_frames = len(frames)
filtered_frames = []

# Iterate over each frame
for i, frame in enumerate(frames):
    # Get atomic positions
    positions = frame.get_positions()
    # Compute distances between all pairs of atoms
    distances = pdist(positions)
    # Find the minimum distance in the current frame
    min_distance = np.min(distances)
    
    # Check if the minimum distance meets the threshold
    if distance_threshold is None or min_distance >= distance_threshold:
        filtered_frames.append(frame)
    
    # Print progress bar
    print_progress_bar(i + 1, total_frames)

filtered_count = len(filtered_frames)
filtered_out_count = total_frames - filtered_count

# Output the filtered frames to a new XYZ file
output_file_name = 'filtered_' + file_name
write(output_file_name, filtered_frames)

# Print summary of filtering results
print(f' Total structures processed: {total_frames}')
print(f' Structures filtered out: {filtered_out_count}')
print(f' Structures retained: {filtered_count}')
print(f' Filtered structures saved to {output_file_name}')
