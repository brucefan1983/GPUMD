import os
import subprocess

# ------------------ Program Introduction ------------------ #
# This script automates the process of reading a multi-frame .xyz file,
# generating ORCA input files, executing ORCA calculations, parsing ORCA .engrad files,
# and creating a combined .xyz trajectory file with calculated energies and forces.

# Usage:
# 1. Prepare a multi-frame .xyz file containing molecular geometries (e.g., "input.xyz").
# 2. Prepare an ORCA input template file (e.g., "template.inp") with a "* xyz" block for atomic coordinates.
# 3. Update the `xyz_file`, `template_inp`, and other configuration variables below as needed.
# 4. Run the script. Make sure ORCA is correctly installed and its executable path is specified.

# Required Software:
# - ORCA (Quantum Chemistry Package)
# - Python 3.x


#template.inp#

#! B3LYP D3 def2-TZVP(-f) def2/J RIJCOSX noautostart miniprint EnGrad
#%maxcore     2000
#%pal nprocs   12 end
#* xyz   0   1
#O      5.42000000   10.68000000    7.42000000
#H      6.09000000   10.65000000    6.75000000
#H      5.23000000    9.76000000    7.61000000
#O     10.61000000    7.46000000    5.45000000
#H     10.98000000    7.06000000    4.66000000
#H     10.62000000    6.76000000    6.10000000
#*

import os
import subprocess

# ------------------ Input and Output Configuration ------------------ #
orca_path = "/home/chen/software/orca_6_0_0"  # Path to ORCA executable directory
xyz_file = "template.xyz"               # Input multi-frame .xyz file
template_inp = "templateinp"           # ORCA input template file
output_dir = "orca_calculations" # Directory for ORCA input/output files
combined_xyz = "output_trajectory.xyz"  # Output combined .xyz trajectory file
box_size = 50.0                  # Size of the box for the lattice
center_molecule = True           # Set to False if no centering is needed
create_molden = False            # Set to False to skip creation of molden files

# ------------------ Function Definitions ------------------ #
def read_multi_xyz(filename):
    """Read a multi-frame .xyz file and extract atomic data and frame information."""
    frames = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    i = 0
    while i < len(lines):
        # Read the number of atoms in the frame
        try:
            num_atoms = int(lines[i].strip())
        except ValueError:
            print("Unexpected format in .xyz file. Line:", lines[i])
            break
        
        # Read frame information (e.g., energy, metadata)
        frame_info = lines[i + 1].strip()
        
        # Read atomic data, extracting only the first four columns
        atoms = []
        for j in range(i + 2, i + 2 + num_atoms):
            atom_data = lines[j].strip().split()
            atom_symbol = atom_data[0]
            x, y, z = atom_data[1:4]  # Extract the first four columns
            atoms.append(f"{atom_symbol} {x} {y} {z}")
        
        # Save the frame
        frames.append({
            "num_atoms": num_atoms,
            "frame_info": frame_info,
            "atoms": atoms
        })
        
        # Move to the next frame
        i += 2 + num_atoms
    
    return frames


def write_inp_file(template_inp, atoms, output_inp):
    """Generate a new ORCA input file based on a template, inserting atomic data into the '* xyz' block."""
    with open(template_inp, 'r') as file:
        template_lines = file.readlines()
    
    # Find the '* xyz' block in the template
    start_idx = None
    end_idx = None
    for i, line in enumerate(template_lines):
        if line.strip().startswith("* xyz"):
            start_idx = i + 1
        elif start_idx is not None and line.strip().startswith("*"):
            end_idx = i
            break
    
    if start_idx is None or end_idx is None:
        raise ValueError("Template .inp file does not contain a valid '* xyz' coordinates block!")
    
    # Insert atomic data
    new_lines = template_lines[:start_idx] + [atom + "\n" for atom in atoms] + template_lines[end_idx:]
    
    # Write the new input file
    with open(output_inp, 'w') as file:
        file.writelines(new_lines)


def parse_engrad(engrad_file):
    """Parse the ORCA .engrad file and extract energy, gradients, and atomic coordinates."""
    if not os.path.exists(engrad_file):
        raise FileNotFoundError(f"Engrad file not found: {engrad_file}")
        
    with open(engrad_file, 'r') as file:
        lines = [line.strip() for line in file if line.strip() and not line.startswith("#")]

    num_atoms = int(lines[0])  # The first valid line contains the number of atoms
    energy_hartree = float(lines[1])  # The second valid line contains the energy in Hartree
    gradients = [float(lines[i]) for i in range(2, 2 + 3 * num_atoms)]  # Extract gradients (3*num_atoms)
    
    # Extract atomic numbers and coordinates
    atoms = []
    for i in range(2 + 3 * num_atoms, 2 + 3 * num_atoms + num_atoms):
        parts = lines[i].split()
        atomic_number = int(parts[0])
        x, y, z = map(float, parts[1:])
        atoms.append((atomic_number, x, y, z))
    
    return num_atoms, energy_hartree, gradients, atoms


def write_xyz_frame(num_atoms, energy_eV, gradients, atoms, box_size, center_molecule=False):
    """Write a single frame of an .xyz trajectory."""
    BOHR_TO_ANGSTROM = 0.529177210903    # Conversion factor: 1 Bohr = 0.529177210903 Angstrom
    HARTREE_BOHR_TO_EV_ANGSTROM = 27.211386245988 / BOHR_TO_ANGSTROM  # Conversion factor for gradient to force (eV/Angstrom)

    forces = [-g * HARTREE_BOHR_TO_EV_ANGSTROM for g in gradients]
    shift_x = shift_y = shift_z = 0.0
    if center_molecule:
        geo_center = [sum(atom[i + 1] for atom in atoms) / num_atoms for i in range(3)]
        shift_x = box_size / 2 - geo_center[0] * BOHR_TO_ANGSTROM
        shift_y = box_size / 2 - geo_center[1] * BOHR_TO_ANGSTROM
        shift_z = box_size / 2 - geo_center[2] * BOHR_TO_ANGSTROM

    frame_lines = []
    # Write the number of atoms
    frame_lines.append(f"{num_atoms}\n")
    # Write the box information and energy
    frame_lines.append(f'Lattice="{box_size:.1f} 0.0 0.0 0.0 {box_size:.1f} 0.0 0.0 0.0 {box_size:.1f}" '
                       f'Properties=species:S:1:pos:R:3:force:R:3 config_type=orca2xyz '
                       f'energy={energy_eV:.8f} pbc="T T T"\n')
    # Write atomic data
    for i, atom in enumerate(atoms):
        atomic_number, x, y, z = atom
        x = x * BOHR_TO_ANGSTROM + shift_x
        y = y * BOHR_TO_ANGSTROM + shift_y
        z = z * BOHR_TO_ANGSTROM + shift_z
        fx, fy, fz = forces[3 * i:3 * i + 3]
        element = atomic_symbol(atomic_number)
        frame_lines.append(f"{element} {x:.8f} {y:.8f} {z:.8f} {fx:.8f} {fy:.8f} {fz:.8f}\n")
    
    return frame_lines


def atomic_symbol(atomic_number):
    """Convert atomic number to element symbol."""
    periodic_table = {
     1: 'H',   2: 'He',  3: 'Li',  4: 'Be',  5: 'B',   6: 'C',   7: 'N',   8: 'O',   9: 'F',    10: 'Ne',
    11: 'Na',  12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',  16: 'S',  17: 'Cl', 18: 'Ar', 19: 'K',    20: 'Ca',
    21: 'Sc',  22: 'Ti', 23: 'V',  24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',   30: 'Zn',
    31: 'Ga',  32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y',    40: 'Zr',
    41: 'Nb',  42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In',   50: 'Sn',
    51: 'Sb',  52: 'Te', 53: 'I',  54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr',   60: 'Nd',
    61: 'Pm',  62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm',   70: 'Yb',
    71: 'Lu',  72: 'Hf', 73: 'Ta', 74: 'W',  75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au',   80: 'Hg',
    81: 'Tl',  82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac',   90: 'Th',
    91: 'Pa',  92: 'U',  93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es',  100: 'Fm',
   101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds',
   111: 'Rg', 112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'
    }
    return periodic_table.get(atomic_number, 'X')


def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frames = read_multi_xyz(xyz_file)
    print(f"Detected {len(frames)} frames in {xyz_file}")

    problematic_frames = []  # To record frames with issues

    with open(combined_xyz, 'w') as combined_file:
        for frame_idx, frame in enumerate(frames):
            frame_number = frame_idx + 1
            output_inp = os.path.join(output_dir, f"frame{frame_number}.inp")
            output_engrad = os.path.join(output_dir, f"frame{frame_number}.engrad")

            try:
                # Step 1: Run ORCA
                write_inp_file(template_inp, frame["atoms"], output_inp)
                orca_command = os.path.join(orca_path, "orca")  # Construct the full path to ORCA executable
                subprocess.run(f"{orca_command} {output_inp} > {output_inp.replace('.inp', '.out')}", shell=True, check=True)

                # Step 2 (Optional): Run orca_2mkl and rename molden file
                if create_molden:
                    orca_2mkl_command = os.path.join(orca_path, "orca_2mkl")
                    subprocess.run(f"{orca_2mkl_command} frame{frame_number} -molden", shell=True, check=True, cwd=output_dir)
                    molden_input_file = os.path.join(output_dir, f"frame{frame_number}.molden.input")
                    molden_output_file = os.path.join(output_dir, f"frame{frame_number}.molden")
                    os.rename(molden_input_file, molden_output_file)

                # Parse the .engrad file
                num_atoms, energy_hartree, gradients, atoms = parse_engrad(output_engrad)
                energy_eV = energy_hartree * 27.211386245988

                # Write the current frame to the combined .xyz file
                frame_lines = write_xyz_frame(num_atoms, energy_eV, gradients, atoms, box_size, center_molecule)
                combined_file.writelines(frame_lines)

            except Exception as e:
                print(f"Error processing frame {frame_number}: {e}")
                problematic_frames.append(frame_number)  # Record problematic frame number

    # Print summary of problematic frames
    if problematic_frames:
        print("\nThe following frames encountered issues and were skipped:")
        print(", ".join(map(str, problematic_frames)))
    else:
        print("\nAll frames processed successfully!")


if __name__ == "__main__":
    main()
