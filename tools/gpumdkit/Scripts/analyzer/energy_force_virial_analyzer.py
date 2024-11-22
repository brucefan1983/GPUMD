import sys
import numpy as np
from ase.io import read
import matplotlib.pyplot as plt

def calculate_range(frames, property_name):
    property_name = property_name.lower()
    values = []
    
    for frame in frames:
        info_lower = {k.lower(): v for k, v in frame.info.items()}
        
        if property_name == "energy":
            if 'energy' in info_lower:
                values.append(info_lower['energy'])
            else:
                raise ValueError("Energy information not found in frame info.")
        elif property_name in ["force", "forces"]:
            forces = frame.get_forces()
            values.extend(np.linalg.norm(forces, axis=1))
        elif property_name == "virial":
            if 'virial' in info_lower:
                virial = info_lower['virial']
                values.extend(virial)  
            else:
                raise ValueError("Virial information not found in frame info.")
        else:
            raise ValueError("Invalid property. Choose from 'energy', 'force', or 'virial'.")
    
    return np.min(values), np.max(values), values

def plot_histogram(values, property_name):
    plt.figure(figsize=(6,4), dpi=100)
    plt.hist(values, bins=30, edgecolor='black')
    plt.title(f'{property_name.capitalize()} Histogram')
    plt.xlabel(f'{property_name.capitalize()}')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    #plt.savefig(f'range_{property_name.capitalize()}.png')

if __name__ == "__main__":
    # Check if the required arguments are provided
    if len(sys.argv) < 3:
        print("Usage: python script.py <filename> <property> [hist]")
        sys.exit(1)
    
    filename = sys.argv[1]
    property_name = sys.argv[2]
    plot_hist = len(sys.argv) > 3 and sys.argv[3] == 'hist'
    
    # Read the extxyz file
    frames = read(filename, index=":")
    
    # Calculate the range of the specified property
    min_val, max_val, values = calculate_range(frames, property_name)
    
    # Print the range
    print(f"{property_name.capitalize()} range: {min_val} to {max_val}")
    
    # Plot histogram if requested
    if plot_hist:
        plot_histogram(values, property_name)

