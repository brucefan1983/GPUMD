import sys
from ase.io import read, write
import numpy as np

def filter_frames(input_file, output_file, edge_limit):
    # Read all frames from the input file
    frames = read(input_file, index=":")

    filtered_frames = []
    discarded_count = 0

    for frame in frames:
        box = frame.get_cell()  # Get the box matrix (3x3)
        edges = np.linalg.norm(box, axis=1)  # Compute lengths of box edges
        
        if np.all(edges <= edge_limit):
            filtered_frames.append(frame)
        else:
            discarded_count += 1

    # Write the filtered frames to the output file
    if filtered_frames:
        write(output_file, filtered_frames)
    
    # Print the number of filtered-out structures
    print(f"Number of structures discarded: {discarded_count}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <edge_limit>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = "filtered_by_box.xyz"
    edge_limit = float(sys.argv[2])

    filter_frames(input_file, output_file, edge_limit)

