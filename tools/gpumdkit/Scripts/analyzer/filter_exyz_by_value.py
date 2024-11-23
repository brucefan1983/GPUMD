import sys
from ase.io import read, write

def main():
    # Command line arguments
    filename = sys.argv[1]  # Input .extxyz file
    property = sys.argv[2]  # Filtering property: 'energy', 'force', or 'virial'
    threshold = float(sys.argv[3])  # Threshold value for filtering
    
    # Read the structures from the .extxyz file
    images = read(filename, index=':')
    
    # Filter based on the specified property
    filtered_images = []
    for atoms in images:
        if property == 'energy':
            # Assuming energy is stored as a single value in the atoms.info dictionary
            if 'energy' in atoms.info and atoms.info['energy'] <= threshold:
                filtered_images.append(atoms)
        elif property in ["force", "forces"]:
            forces = atoms.get_forces()
            # Assuming we want to filter out structures where any force component is greater than the threshold
            if all(abs(force) <= threshold for force in forces.flatten()):
                filtered_images.append(atoms)
        elif property == 'virial':
            # Assuming virial is stored as a tensor in the atoms.info dictionary
            if 'virial' in atoms.info and all(v <= threshold for v in atoms.info['virial'].flatten()):
                filtered_images.append(atoms)
        else:
            raise ValueError("Unsupported property. Please use 'energy', 'force', or 'virial'.")

    # Write the filtered structures back to a new .extxyz file
    output_filename = f"filtered.xyz"
    write(output_filename, filtered_images)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_file> <property> <threshold>")
        sys.exit(1)
    main()