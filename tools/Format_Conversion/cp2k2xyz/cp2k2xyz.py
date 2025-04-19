#python cp2k2xyz.py [pos.xyz] [frc.xyz] [cell.cell] [-shifted yes/no]
#-shifted yes, energy shift; -shifted no, not energy shift; The default is "-shifted no".
#If only entering "python cp2k2xyz.py", read according to the default file name; Otherwise, read according to the input file name.
#Merge the box information, atomic coordinates, and atomic forces outputted by CP2K AIMD into the xyz file
#Shift total energy to ~0 by substracting atomic energies using the least square method.

import numpy as np
import sys
import os
import glob

def SVD_A(A, b):
    """Solve A*x=b using Singular Value Decomposition."""
    U, S, V = np.linalg.svd(A)
    B = np.matmul(U.T, b)
    X = B[:len(S), :] / S.reshape(len(S), -1)
    x = np.matmul(V.T, X)
    return x

def extract_xyz_data(filename):
    """Extract data from XYZ file, skipping comment lines."""
    with open(filename, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break
            line = line.strip()
            # Skip comment lines
            if line.startswith('#') or not line:
                continue
            try:
                num_atoms = int(line.split()[0])
            except ValueError:
                continue  # Skip non-integer lines
            file.readline()  # Skip the comment line
            atoms_info = []
            for _ in range(num_atoms):
                atoms_info.append(file.readline().strip().split())
            yield num_atoms, atoms_info

def extract_forces_and_energy(frc_file, num_atoms_list):
    """Extract forces and energy from a force file."""
    energies = []
    forces_list = []
    with open(frc_file, 'r') as ff:
        while True:
            line = ff.readline()
            if not line:
                break
            line = line.strip()
            if "E =" in line:
                try:
                    energy = float(line.split("E =")[-1]) * 27.211386245988  # Convert to eV
                    energies.append(energy)
                except ValueError:
                    continue
                forces = []
                for _ in range(num_atoms_list[len(energies) - 1]):
                    frc_line = ff.readline().strip().split()
                    if len(frc_line) < 4 or not frc_line[1].replace(".", "", 1).lstrip('-').isdigit():
                        continue
                    try:
                        force_x = float(frc_line[1]) * 51.42206747632590000
                        force_y = float(frc_line[2]) * 51.42206747632590000
                        force_z = float(frc_line[3]) * 51.42206747632590000
                        forces.append((force_x, force_y, force_z))
                    except ValueError:
                        continue
                forces_list.append(forces)
    return energies, forces_list

def extract_cell_data(cell_file):
    """Extract lattice information from cell file."""
    lattices = []
    with open(cell_file, 'r') as cf:
        cf.readline()  # Skip the header line in the cell file
        while True:
            line = cf.readline().strip()
            if not line:
                break
            cell_line = line.split()
            lattice = " ".join(cell_line[2:11])  # Only read the Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz columns
            lattices.append(lattice)
    return lattices

def write_xyz(output_file, pos_data, forces_list, energies, lattices):
    with open(output_file, 'w') as of:
        for idx, (num_atoms, atoms_info) in enumerate(pos_data):
            energy = energies[idx]
            forces = forces_list[idx]
            lattice = lattices[idx]
            of.write(f"{num_atoms}\n")
            of.write(f"energy={energy:.10f} config_type=cp2k2xyz pbc=\"T T T\" Lattice=\"{lattice}\" Properties=species:S:1:pos:R:3:force:R:3\n")
            for i, atom_info in enumerate(atoms_info):
                symbol = atom_info[0]
                x, y, z = map(float, atom_info[1:4])
                force_x, force_y, force_z = forces[i]
                of.write(f"{symbol:<2} {x:>20.10f} {y:>20.10f} {z:>20.10f} {force_x:>20.10f} {force_y:>20.10f} {force_z:>20.10f}\n")

def find_file(pattern):
    """Find a file with a given pattern."""
    files = glob.glob(pattern)
    if len(files) != 1:
        raise SystemError(f"Expected one file matching {pattern}, found {len(files)}.")
    return files[0]

# Parse command line arguments
args = sys.argv[1:]  # Skip the script name
shifted = "no"  # Default behavior is no energy shifting

# Check for the -shifted argument
if "-shifted" in args:
    shifted_index = args.index("-shifted")
    if shifted_index + 1 < len(args):
        shifted = args[shifted_index + 1].lower()
        del args[shifted_index:shifted_index + 2]

# Determine file names
if len(args) == 3:
    pos_file, frc_file, cell_file = args
else:
    pos_file = find_file("*-pos-1*")    #default file name
    frc_file = find_file("*-frc-1*")
    cell_file = find_file("*.cell")

if not all(os.path.exists(f) for f in [pos_file, frc_file, cell_file]):
    raise SystemError("One or more input files do not exist.")

# Read data from files
pos_data = list(extract_xyz_data(pos_file))
num_atoms_list = [num_atoms for num_atoms, _ in pos_data]  # Collect number of atoms for each frame
energies, forces_list = extract_forces_and_energy(frc_file, num_atoms_list)
lattices = extract_cell_data(cell_file)

if shifted == "yes":
    # Determine unique elements
    flatten_comprehension = lambda matrix: [item[0] for sublist in matrix for item in sublist]
    all_elements = sorted(set(flatten_comprehension([atoms_info for _, atoms_info in pos_data])))

    coeff_matrix = np.zeros((len(pos_data), len(all_elements)))
    energy_matrix = np.zeros((len(pos_data), 1))

    for idx, (num_atoms, atoms_info) in enumerate(pos_data):
        for j, element in enumerate(all_elements):
            coeff_matrix[idx][j] = sum(1 for atom in atoms_info if atom[0] == element)
        energy_matrix[idx][0] = energies[idx]

    print('Normalizing energy....')
    if np.linalg.matrix_rank(coeff_matrix) < len(all_elements):
        print("Warning! The coeff_matrix is underdetermined, adding constraints....")
        import itertools
        for i in itertools.combinations(range(len(all_elements)), 2):
            additional_matrix = np.zeros(len(all_elements))
            additional_matrix[i[0]], additional_matrix[i[1]] = 1, -1
            additional_energy = np.zeros(1)
            coeff_matrix = np.r_[coeff_matrix, [additional_matrix]]
            energy_matrix = np.r_[energy_matrix, [additional_energy]]

    # Solve for atomic energy shifts
    atomic_shifted_energy = SVD_A(coeff_matrix, energy_matrix)
    for i, element in enumerate(all_elements):
        print(f"{element}:{atomic_shifted_energy[i][0]:.10f} eV")

    # Calculate shifted energies
    shifted_energy = (energy_matrix - np.matmul(coeff_matrix, atomic_shifted_energy)).flatten()
    print("Averaged energies now: %f eV." % shifted_energy[:len(pos_data)].mean())
    print("Absolute maximum energy now: %f eV." % max(abs(shifted_energy[:len(pos_data)])))

    # Write shifted XYZ file
    write_xyz("shifted.xyz", pos_data, forces_list, shifted_energy.tolist(), lattices)

# Always write the original XYZ file
write_xyz("original.xyz", pos_data, forces_list, energies, lattices)

print("Done! 'original.xyz' file is generated.")
if shifted == "yes":
    print("Done! 'shifted.xyz' file is generated.")
