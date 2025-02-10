import sys
from ase.io import read

def convert_poscar_to_lammps_data(poscar_file, lammps_data_file):
    # Read the POSCAR file
    atoms = read(poscar_file, format='vasp')
    
    # Get element symbols and masses
    symbols = atoms.get_chemical_symbols()
    masses = atoms.get_masses()
    unique_symbols = sorted(set(symbols))
    
    # Create a dictionary to map element symbols to their type numbers
    symbol_to_type = {symbol: i + 1 for i, symbol in enumerate(unique_symbols)}
    
    # Create LAMMPS data file
    with open(lammps_data_file, 'w') as f:
        f.write("LAMMPS data file\n\n")
        
        # Write the number of atoms and atom types
        f.write(f"{len(atoms)} atoms\n")
        f.write(f"{len(unique_symbols)} atom types\n\n")
        
        # Write the box dimensions
        cell = atoms.get_cell()
        f.write(f"0.0 {cell[0][0]} xlo xhi\n")
        f.write(f"0.0 {cell[1][1]} ylo yhi\n")
        f.write(f"0.0 {cell[2][2]} zlo zhi\n")
        f.write(f"{cell[1][0]} {cell[2][0]} {cell[2][1]} xy xz yz\n\n")
        
        # Write the masses
        f.write("Masses\n\n")
        for i, symbol in enumerate(unique_symbols):
            mass = masses[symbols.index(symbol)]
            f.write(f"{i+1} {mass}\n")
        f.write("\n")
        
        # Write the atomic coordinates
        f.write("Atoms\n\n")
        for i, atom in enumerate(atoms):
            atom_id = i + 1
            atom_type = symbol_to_type[atom.symbol]
            x, y, z = atom.position
            f.write(f"{atom_id} {atom_type} {x} {y} {z}\n")
    
#    print(f"Conversion complete! {poscar_file} has been converted to {lammps_data_file}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python pos2lmp.py <poscar_file> <lammps_data_file>")
        sys.exit(1)
    
    poscar_file = sys.argv[1]
    lammps_data_file = sys.argv[2]
    
    convert_poscar_to_lammps_data(poscar_file, lammps_data_file)
