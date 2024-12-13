#Select frames from the exyz trajectory file.
#When an atom in a frame experiences a force greater than the force_threshold, the frame is removed.   eV/A
#When the total energy difference between adjacent frames is less than energy_threshold, only one frame is retained.    eV
#When the RMSD difference between adjacent frames is less than rmsd_threshold, only one frame is retained.              A^2
#If set to 'not', do not select based on this condition.

import numpy as np

def parse_xyz_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    frames = []
    i = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())
        frame_info = lines[i + 1].strip()


        energy_str = frame_info.split('energy=')[1].split()[0]
        energy = float(energy_str)


        lattice_str = frame_info.split('Lattice="')[1].split('"')[0]
        lattice = np.array(list(map(float, lattice_str.split()))).reshape(3, 3)

        atoms = lines[i + 2:i + 2 + num_atoms]
        frames.append((num_atoms, frame_info, energy, lattice, atoms))
        i += 2 + num_atoms
    return frames

def force_exceeds_threshold(atom_line, threshold):
    if threshold == "not":
        return False
    forces = np.array(list(map(float, atom_line.split()[4:7])))  
    total_force = np.linalg.norm(forces)  # Calculate the resultant force
    return total_force > threshold

def calculate_rmsd(frame1_atoms, frame2_atoms, lattice):
    num_atoms = len(frame1_atoms)
    rmsd_sum = 0.0

    for atom1, atom2 in zip(frame1_atoms, frame2_atoms):
        pos1 = np.array(list(map(float, atom1.split()[1:4])))  
        pos2 = np.array(list(map(float, atom2.split()[1:4])))

        # Calculate the minimum image distance
        diff = pos2 - pos1
        diff -= np.round(diff @ np.linalg.inv(lattice)) @ lattice

        rmsd_sum += np.dot(diff, diff)

    return np.sqrt(rmsd_sum / num_atoms)

def filter_frames(frames, force_threshold, energy_threshold, rmsd_threshold):
    filtered_frames = []
    removed_frames_indices = []

    for j in range(len(frames)):
        current_frame = frames[j]
        num_atoms, frame_info, energy, lattice, atoms = current_frame

        # Check the resultant force
        if force_threshold != "not" and any(force_exceeds_threshold(atom, force_threshold) for atom in atoms):
            removed_frames_indices.append(j + 1)  #Record frame number, starting from 1
            continue

        # Check the energy difference
        if j > 0 and energy_threshold != "not":
            prev_energy = frames[j-1][2]
            if abs(energy - prev_energy) < energy_threshold:
                removed_frames_indices.append(j + 1)  #Record frame number, starting from 1
                continue

        # Check RMSD
        if j > 0 and rmsd_threshold != "not":
            prev_atoms = frames[j-1][4]
            rmsd = calculate_rmsd(prev_atoms, atoms, lattice)
            if rmsd < rmsd_threshold:
                removed_frames_indices.append(j + 1)  #Record frame number, starting from 1
                continue

        filtered_frames.append(current_frame)

    return filtered_frames, removed_frames_indices

def write_xyz_file(frames, output_filename):
    with open(output_filename, 'w') as file:
        for num_atoms, frame_info, energy, lattice, atoms in frames:
            file.write(f"{num_atoms}\n")
            file.write(f"{frame_info}\n")
            for atom in atoms:
                file.write(f"{atom.strip()}\n")

force_threshold = 30.0  # Force resultant threshold, if you do not want to apply this restriction, set it to "not"
energy_threshold = "not"  # Energy difference threshold
rmsd_threshold = "not"  # RMSD threshold
input_filename = 'merged_xyz.xyz'  
output_filename = 'output.xyz'  

frames = parse_xyz_file(input_filename)
filtered_frames, removed_frames_indices = filter_frames(frames, force_threshold, energy_threshold, rmsd_threshold)
write_xyz_file(filtered_frames, output_filename)

# Output the frame number that has been removed, starting from 1
print(f"Removed frame indices: {removed_frames_indices}")
