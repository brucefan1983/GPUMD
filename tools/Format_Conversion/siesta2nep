#!/usr/bin/env python3
"""
Convert SIESTA outputs to NEP training format (xyz)
Usage: python siesta2nep.py -o output.xyz
"""

import argparse
import glob
import numpy as np
from pathlib import Path

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="Convert SIESTA outputs to NEP format")
parser.add_argument("-o", "--output", default="siesta2nep.xyz",
                    help="XYZ file to write (default: siesta2nep.xyz)")
args = parser.parse_args()
out_path = Path(args.output)

def dual_print(text, file=None):
    print(text)
    if file:
        file.write(text + "\n")

# ---------- 1. Free energy ----------
with open("BASIS_ENTHALPY") as f:
    energy = next(
        float(line.split(":")[-1])
        for line in f
        if "The above number is the electronic (free)energy:" in line
    )

# ---------- 2. Forces from *.FA ----------
fa_file = glob.glob("*.FA")[0]
with open(fa_file) as f:
    fa_lines = f.readlines()

natoms = int(fa_lines[0])
indices, forces = [], []
for line in fa_lines[1:natoms + 1]:
    idx, fx, fy, fz = line.split()
    indices.append(int(idx))
    forces.append([float(fx), float(fy), float(fz)])

indices = np.array(indices)
forces = np.array(forces)

# ---------- 3. Geometry from *STRUCT_OUT ----------
struct_file = glob.glob("*STRUCT_OUT")[0]
with open(struct_file) as f:
    s_lines = f.readlines()

supercell = np.array([[float(x) for x in s_lines[i].split()] for i in range(3)])
natoms_struct = int(s_lines[3])

atom_lines = s_lines[4:4 + natoms_struct]

# Extract full per-atom info
atom_types_raw = []
frac = []
atomic_numbers = []

for ln in atom_lines:
    tokens = ln.split()
    atom_types_raw.append(int(tokens[0]))  # original type (1-based)
    atomic_numbers.append(int(tokens[1]))  # atomic number
    frac.append([float(x) for x in tokens[-3:]])

frac = np.array(frac)
wrapped_frac = frac % 1.0
cart = wrapped_frac @ supercell

# Map atomic numbers to element symbols
element_map = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
    9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
    16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
    23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
    30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr'
}

atom_symbols = [element_map.get(z, f"X{z}") for z in atomic_numbers]

# ---------- 4. Stress tensor and volume from relax.out ----------
with open("relax.out") as f:
    r_lines = f.readlines()

# Find stress tensor
stress_indices = [
    idx for idx, line in enumerate(r_lines)
    if "siesta: Stress tensor (static) (eV/Ang**3):" in line
]

if not stress_indices:
    raise ValueError("Static stress tensor not found!")

stress_index = stress_indices[-1]

# Read the full 3x3 stress tensor
tensor_lines = r_lines[stress_index + 1 : stress_index + 4]
stress_tensor = []
for ln in tensor_lines:
    parts = ln.split()
    stress_tensor.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])

stress_tensor = np.array(stress_tensor)

# Find cell volume
volume_lines = [
    line for line in r_lines
    if "siesta: Cell volume =" in line
]

if not volume_lines:
    raise ValueError("Cell volume not found!")

volume_line = volume_lines[-1]
volume = float(volume_line.split("=")[-1].split()[0])

# Calculate virial = -stress * volume
virial_tensor = -stress_tensor * volume

# Flatten stress and virial tensors to 9 components
stress_flat = stress_tensor.flatten()
virial_flat = virial_tensor.flatten()

# ---------- 5. Read calculation time from CLOCK file ----------
time_value = 0.0  # Default value
try:
    with open("CLOCK") as f:
        for line in f:
            if "End of run" in line:
                parts = line.split()
                if len(parts) >= 4:
                    time_value = float(parts[3])
                break
except FileNotFoundError:
    print("Warning: CLOCK file not found, using default time value")

# ---------- 6. Write NEP format XYZ ----------
with out_path.open("w") as f:
    # First line: number of atoms
    dual_print(str(natoms_struct), f)
    
    # Second line: metadata
    lattice_str = " ".join([f"{x:.8f}" for x in supercell.flatten()])
    virial_str = " ".join([f"{x:.8f}" for x in virial_flat])
    stress_str = " ".join([f"{x:.8f}" for x in stress_flat])
    
    metadata = f'Config_type=siesta2nep Weight=1.0 Lattice="{lattice_str}" '
    metadata += f'Energy={energy:.8f} Virial="{virial_str}" '
    metadata += f'Stress="{stress_str}" '
    metadata += f'pbc="T T T" '
    metadata += f'Properties=species:S:1:pos:R:3:force:R:3'
    
    dual_print(metadata, f)
    
    # Atom lines: element, coordinates, forces
    for i in range(natoms_struct):
        elem = atom_symbols[i]
        x, y, z = cart[i]
        fx, fy, fz = forces[i]
        atom_line = f"{elem} {x:.8f} {y:.8f} {z:.8f} {fx:.8f} {fy:.8f} {fz:.8f}"
        dual_print(atom_line, f)

print(f"Conversion completed. Output written to {out_path}")
print(f"Structure: {natoms_struct} atoms")
print(f"Energy: {energy:.6f} eV")
print(f"Volume: {volume:.6f} Ang^3")
