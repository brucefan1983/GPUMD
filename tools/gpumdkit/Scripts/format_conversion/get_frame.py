import sys
from ase.io import read, write

def get_frame():
    input_file = sys.argv[1]
    frame_number = int(sys.argv[2])
    frames = read(input_file, index=':')
    
    # Check if the frame number is valid
    if frame_number < 1 or frame_number > len(frames):
        print(f"Error: Frame {frame_number} is out of range. The file contains {len(frames)} frames.")
        sys.exit(1)
    selected_frame = frames[frame_number - 1]
    output_file = f"frame_{frame_number}.xyz"
    write(output_file, selected_frame)
    
    print(f"Frame {frame_number} has been successfully written to {output_file}.")

if __name__ == "__main__":
    get_frame()
