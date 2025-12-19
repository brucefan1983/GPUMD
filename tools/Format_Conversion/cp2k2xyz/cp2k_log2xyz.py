#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert validated CP2K calculations into a extended XYZ file.

Scans for .inp files, interactively lets you select which to process, 
finds matching .log files, and extracts:
  - atomic coordinates (from .inp),
  - lattice vectors (from .inp &CELL),
  - total energy (eV), forces (eV/Å), and stress (GPa) (from .log),
  - converts stress → virial (eV) using cell volume.

Skips frames missing any of {coordinates, energy, forces, stress}.
Outputs all valid frames to 'cp2k_selected.xyz' in extended XYZ format
compatible with ASE/NEP/MACE, with 'dirID' metadata for tracking.

Usage:
    python cp2k_log2xyz.py

Requirements:
  - .inp: &COORD and &CELL blocks.
  - .log: 'ENERGY| Total FORCE_EVAL', 'ATOMIC FORCES', and 'Analytical stress tensor [GPa]'.

Output fields:
  Lattice="Ax Ay Az Bx By Bz Cx Cy Cz"
  energy=<eV> virial="<9×eV>" pbc="T T T"
  Properties=species:S:1:pos:R:3:force:R:3
  dirID="<parent_dir>"
"""

import re
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np

def natural_sort_key(name: str):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(name))]

def find_log_for_inp(inp: Path) -> Optional[Path]:
    candidates = [
        inp.with_suffix('.log'),
        inp.parent / f"cp2k-{inp.stem}.log",
        inp.parent / f"cp2k_{inp.stem}.log",
        inp.parent / "cp2k.log",
        inp.parent / "output.log",
        inp.parent / f"{inp.stem}_output.log",
    ]
    for log in candidates:
        if log.is_file():
            return log
    logs = list(inp.parent.glob("*.log"))
    return logs[0] if logs else None

def extract_lattice(content: str) -> List[float]:
    match = re.search(r'&CELL.*?A\s+([\d\.\-E\+\s]+)\s+B\s+([\d\.\-E\+\s]+)\s+C\s+([\d\.\-E\+\s]+).*?&END CELL', content, re.DOTALL | re.IGNORECASE)
    return [float(x) for group in match.groups() for x in group.split()] if match else [0.0] * 9

def compute_volume_from_lattice(lattice: List[float]) -> float:
    if len(lattice) != 9:
        return 1.0
    a = np.array(lattice[0:3])
    b = np.array(lattice[3:6])
    c = np.array(lattice[6:9])
    return abs(np.dot(a, np.cross(b, c)))

def extract_atoms(content: str) -> List[List]:
    match = re.search(r'&COORD\s*(.*?)&END COORD', content, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    atoms = []
    for line in match.group(1).strip().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 4:
                atoms.append([parts[0]] + [float(x) for x in parts[1:4]])
    return atoms

def extract_energy(content: str) -> float:
    match = re.search(r'ENERGY\| Total FORCE_EVAL.*?:\s+([\d\.\-E\+]+)', content)
    return float(match.group(1)) * 27.2113838565563 if match else float('nan')

def extract_stress_in_gpa(content: str) -> List[float]:
    match = re.search(r'STRESS\|\s+Analytical stress tensor\s+\[GPa\]', content)
    if not match:
        return []
    pattern = r'STRESS\|\s+[xyz]\s+([-\d\.E+\-]+)\s+([-\d\.E+\-]+)\s+([-\d\.E+\-]+)'
    matches = re.findall(pattern, content, re.IGNORECASE)
    if len(matches) >= 3:
        return [float(val) for row in matches[:3] for val in row]
    return []

def stress_to_virial(stress_gpa: List[float], volume_ang3: float) -> List[float]:
    if len(stress_gpa) != 9 or volume_ang3 <= 0:
        return [0.0] * 9
    factor = volume_ang3 / 160.2176634
    return [s * factor for s in stress_gpa]

def extract_forces(content: str) -> List[List[float]]:
    match = re.search(r'ATOMIC FORCES in \[a\.u\.\]\n\n # Atom\s+Kind\s+Element\s+X\s+Y\s+Z\n(.*?)(?=\n SUM OF ATOMIC FORCES)', content, re.DOTALL)
    if not match:
        return []
    forces = []
    for line in match.group(1).strip().splitlines():
        parts = line.split()
        if len(parts) >= 6:
            fx, fy, fz = [float(parts[i]) * 51.4220631857 for i in range(3, 6)]
            forces.append([fx, fy, fz])
    return forces

def format_xyz_frame(lattice, atoms, energy, virial, forces, mat_id: str) -> str:
    lines = [f"{len(atoms)}\n"]
    lattice_str = ' '.join(f"{x:.10f}" for x in lattice)
    virial_str = ' '.join(f"{x:.10f}" for x in virial)
    info = [
        f'Lattice="{lattice_str}"',
        f'energy={energy:.10f}',
        f'virial="{virial_str}"',
        'pbc="T T T"',
        'Properties=species:S:1:pos:R:3:force:R:3',
        f'dirID="{mat_id}"'
    ]
    lines.append(' '.join(info) + '\n')
    for i, atom in enumerate(atoms):
        el, x, y, z = atom
        fx = fy = fz = 0.0
        if i < len(forces):
            fx, fy, fz = forces[i]
        lines.append(f"{el:2s} {x:14.8f} {y:14.8f} {z:14.8f} {fx:14.8f} {fy:14.8f} {fz:14.8f}\n")
    return ''.join(lines)

def select_files_interactive(inp_files: List[Path]) -> List[Path]:
    print(f"\nFound {len(inp_files)} .inp files:\n")
    for i, f in enumerate(inp_files, 1):
        print(f"{i:3d}. {f.relative_to(Path.cwd())}")
    print("\nEnter your choice:")
    print("  - Single number (e.g., 3)")
    print("  - Multiple numbers separated by commas (e.g., 1,4,7)")
    print("  - Range (e.g., 2-5)")
    print("  - 'all' to select all")
    print("  - 'quit' to exit")
    while True:
        choice = input("\nYour selection: ").strip()
        if choice.lower() in ('quit', 'q'):
            sys.exit(0)
        elif choice.lower() == 'all':
            return inp_files
        else:
            selected = []
            try:
                for part in choice.split(','):
                    part = part.strip()
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        selected.extend(range(start-1, end))
                    else:
                        selected.append(int(part)-1)
                return [inp_files[i] for i in selected if 0 <= i < len(inp_files)]
            except (ValueError, IndexError):
                print("Invalid input. Please try again.")

def main():
    current = Path.cwd()
    inp_files = sorted(current.rglob("*.inp"), key=natural_sort_key)
    if not inp_files:
        print("No .inp files found in current directory or subdirectories.")
        return

    selected = select_files_interactive(inp_files)
    print(f"\nProcessing {len(selected)} selected .inp file(s)...\n")

    success_count = 0
    error_counts = {
        'inp: missing coordinates': 0,
        'log: file not found': 0,
        'log: missing energy': 0,
        'log: missing forces': 0,
        'log: missing stress': 0,
    }
    failed_details = []
    all_xyz_frames = []

    for inp in selected:
        rel_path = inp.relative_to(current)
        try:
            inp_text = inp.read_text(encoding='utf-8', errors='ignore')
            atoms = extract_atoms(inp_text)
            if not atoms:
                error = 'inp: missing coordinates'
                error_counts[error] += 1
                failed_details.append(f"{rel_path} → {error}")
                continue

            log_file = find_log_for_inp(inp)
            if not log_file:
                error = 'log: file not found'
                error_counts[error] += 1
                failed_details.append(f"{rel_path} → {error}")
                continue

            log_text = log_file.read_text(encoding='utf-8', errors='ignore')

            energy = extract_energy(log_text)
            forces = extract_forces(log_text)
            stress = extract_stress_in_gpa(log_text)

            # Validate presence
            missing = []
            if energy != energy:  # NaN check
                missing.append('energy')
            if not forces:
                missing.append('forces')
            if len(stress) != 9:
                missing.append('stress')

            if missing:
                for m in missing:
                    error_counts[f'log: missing {m}'] += 1
                failed_details.append(f"{rel_path} → log: missing {' / '.join(missing)}")
                continue

            # All valid → collect frame
            lattice = extract_lattice(inp_text)
            volume = compute_volume_from_lattice(lattice)
            virial = stress_to_virial(stress, volume)
            mat_id = inp.parent.name
            frame = format_xyz_frame(lattice, atoms, energy, virial, forces, mat_id)
            all_xyz_frames.append(frame)
            success_count += 1

        except Exception as e:
            msg = f"{rel_path} → exception: {e}"
            failed_details.append(msg)
            print(msg, file=sys.stderr)

    # Write unified XYZ if any success
    output_file = current / "cp2k_selected.xyz"
    if all_xyz_frames:
        output_file.write_text(''.join(all_xyz_frames), encoding='utf-8')
        print(f"\n Successfully wrote {success_count} structures to: {output_file}")

    # Summary
    total = len(selected)
    failed = total - success_count
    print("\n" + "=" * 60)
    print(f"Summary: {success_count} succeeded, {failed} failed")
    for category, count in error_counts.items():
        if count > 0:
            print(f"  - {category}: {count}")
    if failed_details:
        print("\nFailed entries:")
        for d in failed_details:
            print(f"  {d}")

if __name__ == '__main__':
    main()
