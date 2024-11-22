import sys
from ase.io import read, write

def print_progress_bar(iteration, total, length=50):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r Progress: |{bar}| {percent}% Complete', end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()

input_file = sys.argv[1] if len(sys.argv) > 1 else 'train.xyz'

# Read all frames
frames = read(input_file, index=':')
total_frames= len(frames)

# Save to POSCAR
for i, frame in enumerate(frames):
    poscar_filename = f'POSCAR_{i + 1}.vasp'
    write(poscar_filename, frame)
    print_progress_bar(i + 1, total_frames)

print(f' All frames have been converted.')