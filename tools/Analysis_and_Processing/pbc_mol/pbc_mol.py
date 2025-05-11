# This program processes XYZ trajectory files of molecular simulations to ensure that molecules remain intact
# across periodic boundaries in the simulation box. It uses covalent radii to determine bonding between atoms and adjusts
# molecular positions accordingly, considering periodic boundary conditions.

# Key components of the program:
# 1. **Reading and Writing XYZ Files**: The program reads trajectory data from an input XYZ file, preserving all additional columns
#    (e.g., forces, velocities). After processing, it writes the adjusted trajectory to an output XYZ file.
# 2. **Lattice Parsing**: Extracts lattice information from the simulation box to understand periodic boundary conditions.
# 3. **Cell List for Spatial Partitioning**: Divides the simulation box into smaller cells to optimize neighbor searching, 
#    reducing computational overhead.
# 4. **Bond Detection**: Uses covalent radii and bond threshold rules to determine whether two atoms are bonded, 
#    considering periodic boundary conditions.
# 5. **Depth-First Search (DFS)**: Identifies all atoms belonging to the same molecule by traversing a bonding graph.
# 6. **Molecular Adjustment**: Adjusts positions of atoms in each molecule to ensure the entire molecule stays intact 
#    across periodic boundaries.
# 7. **Parallel Processing**: Utilizes Python's multiprocessing module to process trajectory frames in parallel, increasing computational efficiency.

import numpy as np
import re
from multiprocessing import Pool, cpu_count

#Input and output file paths
INPUT_FILE = "dump.xyz"
OUTPUT_FILE = "output.xyz"

#If the distance between atoms is less than 1.15 times the sum of covalent radii, it is considered a bond.
COVALENT_BOND_MULTIPLIER = 1.5

#Covalent radius table (more elements can be added as needed)
#https://doi.org/10.1039/B801115J
COVALENT_RADII = {
"H"	     :   0.31,	
"He"	 :   0.28,	
"Li"	 :   1.28,	
"Be"	 :   0.96,	
"B"	     :   0.84,	
"C"      :   0.76,	
"N"	     :   0.71,	
"O"	     :   0.66,	
"F"	     :   0.57,	
"Ne"	 :   0.58,	
"Na"	 :   1.66,	
"Mg"	 :   1.41,	
"Al"	 :   1.21,	
"Si"	 :   1.11,	
"P"	     :   1.07,	
"S"	     :   1.05,	
"Cl"	 :   1.02,	
"Ar"	 :   1.06,	
"K"	     :   2.03,	
"Ca"	 :   1.76,	
"Sc"	 :   1.70,	
"Ti"	 :   1.60,	
"V"	     :   1.53,	
"Cr"	 :   1.39,	
"Mn"     :   1.39,	
"Fe"     :   1.32,	
"Co"     :   1.26,	
"Ni"	 :   1.24,	
"Cu"	 :   1.32,	
"Zn"	 :   1.22,	
"Ga"	 :   1.22,	
"Ge"	 :   1.20,	
"As"	 :   1.19,	
"Se"	 :   1.20,	
"Br"	 :   1.20,	
"Kr"	 :   1.16,	
"Rb"	 :   2.20,	
"Sr"	 :   1.95,	
"Y"	     :   1.90,	
"Zr"	 :   1.75,	
"Nb"	 :   1.64,	
"Mo"	 :   1.54,	
"Tc"	 :   1.47,	
"Ru"	 :   1.46,	
"Rh"	 :   1.42,	
"Pd"	 :   1.39,	
"Ag"	 :   1.45,	
"Cd"	 :   1.44,	
"In"	 :   1.42,	
"Sn"	 :   1.39,	
"Sb"	 :   1.39,	
"Te"	 :   1.38,	
"I"	     :   1.39,	
"Xe"	 :   1.40,	
"Cs"	 :   2.44,	
"Ba"	 :   2.15,	
"La"	 :   2.07,	
"Ce"	 :   2.04,	
"Pr"	 :   2.03,	
"Nd"	 :   2.01,	
"Pm"	 :   1.99,	
"Sm"	 :   1.98,	
"Eu"	 :   1.98,	
"Gd"	 :   1.96,	
"Tb"	 :   1.94,	
"Dy"	 :   1.92,	
"Ho"	 :   1.92,	
"Er"	 :   1.89,	
"Tm"	 :   1.90,	
"Yb"	 :   1.87,	
"Lu"	 :   1.87,	
"Hf"	 :   1.75,	
"Ta"	 :   1.70,	
"W"	     :   1.62,	
"Re"	 :   1.51,	
"Os"	 :   1.44,	
"Ir"	 :   1.41,	
"Pt"	 :   1.36,	
"Au"	 :   1.36,	
"Hg"	 :   1.32,	
"Tl"	 :   1.45,	
"Pb"	 :   1.46,	
"Bi"	 :   1.48,	
"Po"	 :   1.40,	
"At"	 :   1.50,	
"Rn"	 :   1.50,	
"Fr"	 :   2.60,	
"Ra"	 :   2.21,	
"Ac"	 :   2.15,	
"Th"	 :   2.06,	
"Pa"	 :   2.00,	
"U"	     :   1.96,	
"Np"	 :   1.90,	
"Pu"	 :   1.87,	
"Am"	 :   1.80,	
"Cm"	 :   1.69,	
}
# ==========================================

def read_xyz_trajectory(file_path):
    """
    Read XYZ trajectory file, including extra columns (force, velocity, etc.)
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    trajectory = []
    i = 0
    while i < len(lines):
        # Number of atoms
        n_atoms = int(lines[i].strip())
        # Box info (e.g., Lattice, time, etc.)
        box_info = lines[i + 1].strip()
        # Atom data
        atoms = []
        for j in range(n_atoms):
            atom_line = lines[i + 2 + j].strip().split()
            atom_type = atom_line[0]
            position = list(map(float, atom_line[1:4]))  # Assume 2nd, 3rd, 4th columns are coordinates
            extra_info = atom_line[4:]  # Keep additional columns (e.g., force, velocity)
            atoms.append((atom_type, np.array(position), extra_info))
        trajectory.append((box_info, atoms))
        i += 2 + n_atoms
    
    return trajectory

def write_xyz_trajectory(file_path, trajectory):
    """
    Write processed XYZ trajectory file, preserving all extra columns
    """
    with open(file_path, 'w') as f:
        for frame in trajectory:
            box_info, atoms = frame
            f.write(f"{len(atoms)}\n")
            f.write(f"{box_info}\n")
            for atom_type, position, extra_info in atoms:
                position_str = " ".join(f"{x:.6f}" for x in position)
                extra_info_str = " ".join(extra_info)  # Join extra columns as strings
                f.write(f"{atom_type} {position_str} {extra_info_str}\n")

def parse_lattice(box_info):
    """
    Parse Lattice information from box info
    """
    lattice_match = re.search(r'Lattice="([\d.\s\-]+)"', box_info)
    if lattice_match:
        lattice = list(map(float, lattice_match.group(1).split()))
        return np.array(lattice).reshape((3, 3))
    else:
        raise ValueError("Lattice information not found in box info")

def cell_list(atoms, lattice, cell_size):
    """
    Construct a cell list for spatial partitioning
    """
    atom_types, positions, _ = zip(*atoms)
    positions = np.array(positions)
    n_cells = np.floor(np.diag(lattice) / cell_size).astype(int)
    cell_index = np.floor(positions / cell_size).astype(int) % n_cells

    atom_cells = {}
    for idx, cell in enumerate(cell_index):
        cell_tuple = tuple(cell)
        if cell_tuple not in atom_cells:
            atom_cells[cell_tuple] = []
        atom_cells[cell_tuple].append(idx)
    
    return atom_cells, n_cells

def find_neighbors(cell, n_cells):
    """
    Find all neighboring cells for a given cell index
    """
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                neighbor = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                neighbor = tuple([n % dim for n, dim in zip(neighbor, n_cells)])
                neighbors.append(neighbor)
    return neighbors

def is_bonded(atom1, atom2, lattice):
    """
    Check if two atoms are bonded, considering periodic boundary conditions
    """
    type1, pos1, _ = atom1
    type2, pos2, _ = atom2
    radius1 = COVALENT_RADII.get(type1, 0)
    radius2 = COVALENT_RADII.get(type2, 0)
    bond_threshold = COVALENT_BOND_MULTIPLIER * (radius1 + radius2)

    delta = pos1 - pos2
    for i in range(3):
        delta[i] -= np.rint(delta[i] / lattice[i, i]) * lattice[i, i]
    distance = np.linalg.norm(delta)
    return distance < bond_threshold

def adjust_molecules_parallel(frame):
    """
    Process a single frame, adjust molecules to avoid periodic boundary crossing
    """
    box_info, atoms = frame
    lattice = parse_lattice(box_info)
    cell_size = np.min(np.diag(lattice)) / 10  # Define cell size
    atom_cells, n_cells = cell_list(atoms, lattice, cell_size)

    adjusted_atoms = atoms[:]
    visited = [False] * len(atoms)
    molecules = []

    def dfs(atom_idx, molecule):
        visited[atom_idx] = True
        molecule.append(atom_idx)
        for neighbor_idx in bonded_graph[atom_idx]:
            if not visited[neighbor_idx]:
                dfs(neighbor_idx, molecule)

    # Build bonding graph
    bonded_graph = {i: [] for i in range(len(atoms))}
    for cell, atom_indices in atom_cells.items():
        neighbors = find_neighbors(cell, n_cells)
        for neighbor in neighbors:
            if neighbor in atom_cells:
                for i in atom_indices:
                    for j in atom_cells[neighbor]:
                        if i < j and is_bonded(atoms[i], atoms[j], lattice):
                            bonded_graph[i].append(j)
                            bonded_graph[j].append(i)

    # Adjust molecular positions
    for i in range(len(atoms)):
        if not visited[i]:
            molecule = []
            dfs(i, molecule)
            molecules.append(molecule)

    for molecule in molecules:
        ref_position = adjusted_atoms[molecule[0]][1]
        for atom_idx in molecule:
            atom_type, position, extra_info = adjusted_atoms[atom_idx]
            delta = position - ref_position
            for i in range(3):
                delta[i] -= np.rint(delta[i] / lattice[i, i]) * lattice[i, i]
            adjusted_position = ref_position + delta
            adjusted_atoms[atom_idx] = (atom_type, adjusted_position, extra_info)

    return (box_info, adjusted_atoms)

def process_xyz_trajectory_parallel(input_file, output_file):
    """
    Main function: Parallel processing of the XYZ trajectory
    """
    trajectory = read_xyz_trajectory(input_file)
    n_cores = min(cpu_count(), len(trajectory))

    with Pool(n_cores) as pool:
        processed_trajectory = pool.map(adjust_molecules_parallel, trajectory)

    write_xyz_trajectory(output_file, processed_trajectory)

if __name__ == "__main__":
    # Run the main function
    process_xyz_trajectory_parallel(INPUT_FILE, OUTPUT_FILE)
