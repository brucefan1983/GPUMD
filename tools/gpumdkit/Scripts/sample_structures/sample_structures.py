import sys
import numpy as np
from ase.io import read, write

"""
    Purpose:
        This script samples structures from an extxyz file using either 'uniform' or 'random' sampling methods. 
        The sampled structures are then written to the 'sampled_structures.xyz' file.

    Author:
        Zihan YAN <yanzihan@westlake.edu.cn>

    Run:
    python sample_structures.py <extxyz_file> <sampling_method> <num_samples> [skip_initial]
"""

def main(sampling_method, num_samples, skip_initial):
    # Read all frames
    frames = read(f'./{sys.argv[1]}', index=':')

    # Total number of frames
    num_frames = len(frames)

    # Adjust the frames list if skip_initial is specified
    if skip_initial:
        frames = frames[skip_initial:]
        num_frames = len(frames)  # Update the number of frames after skipping

    if sampling_method == 'uniform':
        # Generate evenly spaced indices
        sampled_indices = np.linspace(0, num_frames-1, num_samples, dtype=int)
    elif sampling_method == 'random':
        # Generate random indices
        sampled_indices = np.random.choice(num_frames, num_samples, replace=False)
    else:
        raise ValueError("Invalid sampling method. Use 'uniform' or 'random'.")

    # Initialize an empty list to store the sampled frames
    sampled_frames = []

    # Collect the sampled frames
    for i, idx in enumerate(sampled_indices):

        # Collect the frame for the sample.xyz file
        sampled_frames.append(frames[idx])

    # Write the sampled frames to sample.xyz
    write('sampled_structures.xyz', sampled_frames)

    print('All sampled frames written to sampled_structures.xyz')

if __name__ == "__main__":
    # Ensure the correct number of arguments are provided
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python sample_structures.py <extxyz_file> <sampling_method> <num_samples> [skip_initial]")
        print("sampling_method: 'uniform' or 'random'")
        print("num_samples: number of frames to sample")
        print("[skip_initial]: optional, number of initial frames to skip")
        sys.exit(1)

    # Parse the command line arguments
    sampling_method = sys.argv[2]
    num_samples = int(sys.argv[3])

    # Parse the skip_initial argument if provided, otherwise default to 0
    skip_initial = int(sys.argv[4]) if len(sys.argv) == 5 else 0

    # Run the main function
    main(sampling_method, num_samples, skip_initial)
