"""
Unified Atomic Energy Baseline Aligner for XYZ Trajectory Files
For xyz files generated under the Windows system, please run dos2unix *.xyz before running the program.

This script provides a flexible tool for aligning energies within multi-frame XYZ trajectory files.
It supports three distinct energy alignment modes, configurable via the 'ALIGNMENT_MODE' setting:

1.  'REF_GROUP_ALIGNMENT':
    Purpose: To align the energies of specified 'shift_groups' to the average energy baseline
             of a designated 'reference_group'. This is useful when you have a high-accuracy
             dataset (reference_group) and want to shift other datasets (shift_groups) to its scale. 
             Grouping is implemented using 'config_type'
    Method: For each shift group, atomic energy baselines (one per element type) are optimized
            by minimizing the mean squared error (MSE) between shifted group energies and the
            mean energy of the reference group.
    Settings Used: 'reference_group', 'shift_groups'. 'nep_model_file' is ignored.

2.  'ZERO_BASELINE_ALIGNMENT':
    Purpose: To shift all energies in the XYZ file such that the calculated energy of free atoms
             (based on optimized atomic baselines) would be zero. This is a common practice for
             standardizing energies to a vacuum reference.
    Method: For each detected 'config_type' group, atomic energy baselines are optimized
            by minimizing the MSE of the shifted energies themselves (effectively aligning to zero).
    Settings Used: None of 'reference_group', 'shift_groups', or 'nep_model_file' are used for logic.

3.  'DFT_TO_NEP_ALIGNMENT':
    Purpose: To align the original DFT energies (read from 'energy=' field) to the energies
             calculated by a Neuroevolution Potential (NEP) model. This is useful
             for standardizing DFT data to the scale of a force field or machine learning potential.
    Method: First, NEP energies are calculated for all structures using an external NEP model file.
            Then, for each 'config_type' group, atomic energy baselines are optimized to minimize
            the MSE between the shifted DFT energies and the calculated NEP energies.
    Settings Used: 'nep_model_file'. 'reference_group' and 'shift_groups' are ignored.

Key Features:
-   **Flexible Mode Selection**: Easily switch between alignment strategies.
-   **Robust XYZ Parsing**: Handles case-insensitive header keys (e.g., 'Energy=' vs 'energy=')
    and supports quoted or unquoted 'config_type' values.
-   **Parallel NEP Calculation**: Leverages multiprocessing for efficient NEP energy computation
    in 'DFT_TO_NEP_ALIGNMENT' mode.
-   **Optimized NES Algorithm**: Uses Natural Evolution Strategy for baseline optimization,
    featuring vectorized calculations, early stopping for efficiency, and full reproducibility
    (fixed random seeds).
-   **Preserves Original Data**: Only the 'energy=' field in the XYZ header is modified.
    Original atomic coordinates, forces (if present in input), and other header fields are preserved.
-   **Intelligent Group Handling**: Automatically detects all 'config_type' groups.
    Ignores irrelevant settings based on the chosen mode.

How to Use:
1.  **Configure File Paths**: Set 'input_xyz_file' and 'output_xyz_file'.
2.  **Select Alignment Mode**: Set 'ALIGNMENT_MODE' to one of the three options:
    'REF_GROUP_ALIGNMENT', 'ZERO_BASELINE_ALIGNMENT', or 'DFT_TO_NEP_ALIGNMENT'.
3.  **Adjust Mode-Specific Settings**:
    -   If 'REF_GROUP_ALIGNMENT': Define 'reference_group'. Optionally, define 'shift_groups'
        (if empty, all non-reference groups will be processed).
    -   If 'DFT_TO_NEP_ALIGNMENT': Provide the path to your NEP model file ('nep_model_file').
4.  **Run the Script**: Execute the Python script. Progress and optimized atomic baselines
    will be printed to the console. The final aligned energies will be written to the
    'output_xyz_file'.

Author: Chen Zherui (chenzherui0124@foxmail.com)

"""


import numpy as np
import re
import os
from collections import Counter, defaultdict
from ase.io import read
from calorine.calculators import CPUNEP
from multiprocessing import Pool, cpu_count

#######################
#      SETTINGS       #
#######################

# --- Core File Paths ---
input_xyz_file = 'train.xyz'
output_xyz_file = 'output_aligned.xyz'

# --- ALIGNMENT MODE SELECTION ---
# Choose one of the following modes:
# 'REF_GROUP_ALIGNMENT': Align energies of 'shift_groups' to a 'reference_group' mean. It can be used for training set preprocessing and nep.restart
#                        
# 'ZERO_BASELINE_ALIGNMENT': Align all energies to a zero atomic baseline. It can only be used for initial training set preprocessing, and cannot be used for nep.restart. This feature is similar to shift_energy_to_zero. py in the tools folder of gpumd, but allows the use of different DFT programs for calculations
#                         
# 'DFT_TO_NEP_ALIGNMENT': Calculate NEP energies, then align DFT energies to NEP energies. It can be used for training set preprocessing and nep.restart. It is recommended to use this feature when fine-tuning the existing nep.txt (such as nep89)
#  
ALIGNMENT_MODE = 'REF_GROUP_ALIGNMENT' # 'REF_GROUP_ALIGNMENT', 'ZERO_BASELINE_ALIGNMENT', 'DFT_TO_NEP_ALIGNMENT'

# --- Mode-Specific Settings ---
reference_group = "cp2k2xyz" # For 'REF_GROUP_ALIGNMENT' mode ONLY. Ignored otherwise. Grouping is implemented using 'config_type'
shift_groups = []  # For 'REF_GROUP_ALIGNMENT' mode ONLY. If empty, auto-detects. Ignored otherwise.

nep_model_file = 'nep.txt' # For 'DFT_TO_NEP_ALIGNMENT' mode ONLY. Ignored otherwise.

# --- NES Hyperparameters ---
max_generations = 10000
population_size = 40
convergence_tol = 1e-8
random_seed = 42

# --- Batch Size for NEP Calculation (only for DFT_TO_NEP_ALIGNMENT mode) ---
nep_batch_size = 32 # Tune this value to balance memory usage and performance. The larger the batch value, the faster the speed, but the more memory is required

##
#   Helper Functions  #
##

def parse_xyz_frames_optimized(xyz_filename):
    """
    Parses XYZ file efficiently for metadata. Does NOT store all atom lines in memory.
    Instead, it records file offsets and lengths for later retrieval.
    Returns:
      frames_metadata: list of dicts with 'primary_energy', 'config_type', 'elem_counts', etc.
      frame_io_info: list of dicts with 'n_atoms', 'header_offset', 'header_len', 'atom_lines_offset', 'atom_lines_len'
    """
    frames_metadata = []
    frame_io_info = []
    
    with open(xyz_filename, "r") as fin:
        line_offsets = [] # Stores (line_start_offset, line_length) for each line
        current_offset = 0
        for line in fin:
            line_offsets.append((current_offset, len(line)))
            current_offset += len(line)
        
        # Reset file pointer to beginning for parsing
        fin.seek(0)

        i = 0
        frame_idx = 0
        while i < len(line_offsets):
            # Line 1: n_atoms
            n_atoms_line_offset = line_offsets[i][0]
            n_atoms = int(fin.readline().strip())
            
            # Line 2: header
            header_line_offset = line_offsets[i+1][0]
            header = fin.readline().strip()

            # Atom lines: n_atoms lines
            atom_lines_start_offset = line_offsets[i+2][0]
            for _ in range(n_atoms):
                fin.readline() # Skip atom lines
            atom_lines_end_offset = fin.tell() 

            atom_lines_len = atom_lines_end_offset - atom_lines_start_offset

            # Extract metadata
            m_energy = re.search(r'energy=([-\d\.Ee]+)', header, re.IGNORECASE)
            primary_energy = float(m_energy.group(1)) if m_energy else None
            if primary_energy is None:
                raise ValueError(f"No 'energy=' found in header: {header} for frame {frame_idx}. This field is required.")

            m_config = re.search(r'config_type="?([^\s"]+)"?', header, re.IGNORECASE)
            config_type = m_config.group(1) if m_config else "default_group"

            # Parse atom symbols for element counts (re-reading these specific lines for counts)
            current_pos = fin.tell() # Store current position
            fin.seek(atom_lines_start_offset) # Seek back to read atom lines for symbol counting
            atom_symbols = []
            for _ in range(n_atoms):
                atom_line = fin.readline()
                atom_symbols.append(atom_line.split()[0])
            fin.seek(current_pos) # Restore file pointer
            elem_counts = Counter(atom_symbols)
            
            frames_metadata.append({
                'primary_energy': primary_energy,
                'config_type': config_type,
                'elem_counts': elem_counts,
                'original_header': header,
                'n_atoms': n_atoms,
                'frame_idx': frame_idx
            })
            
            frame_io_info.append({
                'n_atoms': n_atoms,
                'header_offset': header_line_offset,
                'header_len': line_offsets[i+1][1],
                'atom_lines_offset': atom_lines_start_offset,
                'atom_lines_len': atom_lines_len
            })
            
            i += 2 + n_atoms
            frame_idx += 1
            
    return frames_metadata, frame_io_info

def calculate_nep_batch(batch_start_idx, batch_end_idx, nep_model_file_path, input_xyz_file_path):
    """
    Calculates NEP energies for a batch of structures.
    Reads structures using an ASE slice string for contiguous blocks.
    """
    energies = []
    
    # Read a batch of structures using a slice string
    # E.g., index='0:10' reads frames 0 through 9
    atoms_batch = read(input_xyz_file_path, index=f"{batch_start_idx}:{batch_end_idx}")

    # Initialize calculator
    calc = CPUNEP(nep_model_file_path)
    for atoms_obj in atoms_batch:
        atoms_obj.calc = calc
        energies.append(atoms_obj.get_potential_energy())

    return energies

def atomic_baseline_cost(param_population, source_energies, element_counts, target_energies=None):
    """
    Generic cost function for NES: minimize MSE of ( (source_energy - sum(n_X * E_X)) - target_energy )^2.
    If target_energies is None, it minimizes MSE of (shifted_source_energy)^2 (i.e., targets zero).
    """
    shifted_source_energies = source_energies[None, :] - np.dot(param_population, element_counts.T)
    
    if target_energies is not None:
        cost = np.mean((shifted_source_energies - target_energies[None, :]) ** 2, axis=1).reshape(-1, 1)
    else: # Align to zero baseline
        cost = np.mean(shifted_source_energies ** 2, axis=1).reshape(-1, 1)
    return cost

def nes_optimize_atomic_baseline(
        num_variables,
        max_generations,
        source_energies,
        element_counts,
        target_energies=None,
        pop_size=40,
        tol=1e-8,
        seed=42,
        print_every=100
):
    """
    NES optimizer for finding atomic reference energies.
    """
    np.random.seed(seed)
    import random
    random.seed(seed)

    best_fitness_history = np.ones((max_generations, 1))
    elite_solutions = np.zeros((max_generations, num_variables))
    mean = -1 * np.random.rand(1, num_variables) 
    stddev = 0.1 * np.ones((1, num_variables)) 
    lr_mean = 1.0 
    lr_stddev = (3 + np.log(num_variables)) / (5 * np.sqrt(num_variables)) / 2 
    selection_weights = np.maximum(0, np.log(pop_size / 2 + 1) - np.log(np.arange(1, pop_size + 1)))
    selection_weights = selection_weights / np.sum(selection_weights) - 1 / pop_size

    for gen in range(max_generations):
        z_samples = np.random.randn(pop_size, num_variables) 
        population = mean + stddev * z_samples 
        fitness = atomic_baseline_cost(population, source_energies, element_counts, target_energies) 
        sorted_idx = np.argsort(fitness.flatten()) 
        fitness = fitness[sorted_idx]
        z_samples = z_samples[sorted_idx, :]
        population = population[sorted_idx, :]
        
        best_fitness_history[gen] = fitness[0, 0]
        elite_solutions[gen, :] = population[0, :]
        
        mean += lr_mean * stddev * (selection_weights @ z_samples)
        stddev *= np.exp(lr_stddev * (selection_weights @ (z_samples ** 2 - 1)))
        
        if gen % print_every == 0:
            print(f'Generation = {gen}, best fitness = {fitness[0,0]:.8f}')
        
        if gen > 0 and abs(best_fitness_history[gen] - best_fitness_history[gen-1]) < tol:
            print(f"Converged at generation {gen}.")
            best_fitness_history = best_fitness_history[:gen + 1]
            elite_solutions = elite_solutions[:gen + 1]
            break
    return best_fitness_history, elite_solutions

##
#        MAIN LOGIC      #
##

# Ensure full reproducibility by setting seeds globally
np.random.seed(random_seed)
import random
random.seed(random_seed)

# 1. Parse XYZ file efficiently to get metadata and file I/O info
print(f"Parsing '{input_xyz_file}' for energies and structure metadata (memory-optimized)...")
frames_metadata, frame_io_info = parse_xyz_frames_optimized(input_xyz_file)

# 2. Identify all unique elements and config_types
all_elements = sorted(list(set(e for f in frames_metadata for e in f['elem_counts'])))
num_elements = len(all_elements)
print(f"Unique elements found: {all_elements}")

all_config_types = sorted(list(set(f['config_type'] for f in frames_metadata if f['config_type'] is not None)))
print(f"Detected config_types (groups): {all_config_types}")

# Store the optimized atomic baselines for each group
group_to_optimized_baseline = {}

# --- Conditional Logic based on ALIGNMENT_MODE ---

# Determine which groups to process based on the selected mode
groups_to_process = []
target_energies_for_nes = None # Default target for NES

if ALIGNMENT_MODE == 'REF_GROUP_ALIGNMENT':
    print(f"\n--- Mode: Reference Group Alignment ---")
    
    # Process groups explicitly in shift_groups or auto-detected, excluding reference group
    if not shift_groups:
        groups_to_process = [g for g in all_config_types if g != reference_group]
        print(f"Detected shift_groups (auto): {groups_to_process}")
    else:
        groups_to_process = [g for g in shift_groups if g != reference_group] 
        print(f"Shift_groups (user-defined, excluding reference): {groups_to_process}")
    
    # Get reference group mean energy
    ref_frames = [f for f in frames_metadata if f['config_type'] == reference_group]
    if len(ref_frames) == 0:
        raise RuntimeError(f"No reference group ('{reference_group}') found in input for REF_GROUP_ALIGNMENT mode!")
    ref_energies = np.array([f['primary_energy'] for f in ref_frames])
    ref_mean_energy = np.mean(ref_energies)
    print(f"Mean energy of reference group ('{reference_group}'): {ref_mean_energy:.8f}")

    # The actual target_energies_for_nes will be np.full_like(group_primary_energies, ref_mean_energy) inside the loop.

elif ALIGNMENT_MODE == 'ZERO_BASELINE_ALIGNMENT':
    print(f"\n--- Mode: Align to Zero Atomic Baseline ---")
    # For this mode, ALL detected config_types are processed. reference_group and shift_groups are ignored.
    groups_to_process = all_config_types
    print(f"All detected groups will be processed: {groups_to_process}")
    # Target for this mode is implicitly zero, target_energies_for_nes will remain None.

elif ALIGNMENT_MODE == 'DFT_TO_NEP_ALIGNMENT':
    print(f"\n--- Mode: DFT to NEP Alignment ---")
    # For this mode, ALL detected config_types are processed. reference_group and shift_groups are ignored.
    groups_to_process = all_config_types
    print(f"All detected groups will be processed: {groups_to_process}")

    # Check NEP model file existence
    if not nep_model_file or not os.path.exists(nep_model_file):
        raise FileNotFoundError(f"NEP model file '{nep_model_file}' not found. Required for DFT_TO_NEP_ALIGNMENT mode.")

    # Calculate NEP energies for all structures in parallel (memory efficient)
    print(f"Calculating NEP energies in batches...")
    num_frames = len(frames_metadata)
    num_batches = int(np.ceil(num_frames / nep_batch_size))
    
    # Use a list to store NEP energies in order of frame index
    all_nep_calculated_energies = [None] * num_frames 

    with Pool(cpu_count()) as pool:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * nep_batch_size
            end_idx = min((batch_idx + 1) * nep_batch_size, num_frames)
            
            # Pass the batch range (start_idx, end_idx) to worker
            # The worker will read this slice from the file.
            batch_nep_energies = pool.apply(calculate_nep_batch, 
  args=(start_idx, end_idx, nep_model_file, input_xyz_file))
            
            # Store results back into the main list
            for i, energy in enumerate(batch_nep_energies):
                all_nep_calculated_energies[start_idx + i] = energy
            
            print(f"  Processed NEP batch {batch_idx+1}/{num_batches} (frames {start_idx}-{end_idx-1})")

    # Integrate calculated NEP energies back into frames_metadata
    for i, nep_energy in enumerate(all_nep_calculated_energies):
        frames_metadata[i]['nep_energy_calculated'] = nep_energy

else:
    raise ValueError(f"Unknown ALIGNMENT_MODE: '{ALIGNMENT_MODE}'. Please choose from 'REF_GROUP_ALIGNMENT', 'ZERO_BASELINE_ALIGNMENT', or 'DFT_TO_NEP_ALIGNMENT'.")


# --- Optimization Loop ---
for group_name in groups_to_process:
    group_frames = [f for f in frames_metadata if f['config_type'] == group_name]
    if not group_frames:
        print(f"Warning: No frames found for config_type '{group_name}'. Skipping optimization.")
        continue

    group_primary_energies = np.array([f['primary_energy'] for f in group_frames]) # Energy to be shifted
    
    group_element_counts = np.array([
        [f['elem_counts'].get(e, 0) for e in all_elements] 
        for f in group_frames
    ], dtype=float)

    # Set specific target_energies_for_nes based on mode
    if ALIGNMENT_MODE == 'REF_GROUP_ALIGNMENT':
        target_energies_for_nes = np.full_like(group_primary_energies, ref_mean_energy)
        print_opt_msg = f"\n--- Optimizing atomic baseline for group: '{group_name}' to align with '{reference_group}' ---"
    elif ALIGNMENT_MODE == 'ZERO_BASELINE_ALIGNMENT':
        target_energies_for_nes = None # Align to zero
        print_opt_msg = f"\n--- Optimizing atomic baseline for group: '{group_name}' to align to zero ---"
    elif ALIGNMENT_MODE == 'DFT_TO_NEP_ALIGNMENT':
        target_energies_for_nes = np.array([f['nep_energy_calculated'] for f in group_frames]) # Target is calculated NEP energy
        print_opt_msg = f"\n--- Optimizing DFT-to-NEP atomic baselines for group: '{group_name}' ---"
    
    print(print_opt_msg)
    
    best_fitness_history, elite_solutions = nes_optimize_atomic_baseline(
        num_elements, max_generations, group_primary_energies, group_element_counts, target_energies_for_nes,
        pop_size=population_size, tol=convergence_tol, seed=random_seed, print_every=100
    )
    
    optimized_baseline = elite_solutions[-1, :]
    group_to_optimized_baseline[group_name] = optimized_baseline
    
    print(f"  Optimized atomic baselines for group '{group_name}':")
    for e, val in zip(all_elements, optimized_baseline):
        print(f"    {e}: {val:.8f}")

# --- Final Output Writing ---
print(f"\n--- Writing final aligned energies to '{output_xyz_file}' ---")
# Open input file in read mode to seek for atom lines
with open(input_xyz_file, "r") as fin_orig, open(output_xyz_file, "w") as fout:
    for i, f_meta in enumerate(frames_metadata):
        original_primary_energy = f_meta['primary_energy'] 
        
        shifted_energy = original_primary_energy # Default: no shift if group was not processed
        
        # Apply shift if the group was part of the optimization process for this mode
        if f_meta['config_type'] in group_to_optimized_baseline:
            optimized_baseline_for_group = group_to_optimized_baseline[f_meta['config_type']]
            frame_element_counts = np.array([f_meta['elem_counts'].get(e, 0) for e in all_elements], dtype=float)
            baseline_sum = np.dot(frame_element_counts, optimized_baseline_for_group)
            shifted_energy = original_primary_energy - baseline_sum
        else:
            # Provide specific warning/info for cases not processed
            if ALIGNMENT_MODE == 'REF_GROUP_ALIGNMENT' and f_meta['config_type'] == reference_group:
                print(f"Note: Reference group '{f_meta['config_type']}' energy is kept as original (no shift applied).")
            else:
                print(f"Warning: config_type '{f_meta['config_type']}' was not optimized/processed. Its energy will remain as original.")
            
        # Reconstruct header with updated energy. 
        header_new = re.sub(r'energy=([-\d\.Ee]+)', f'energy={shifted_energy:.8f}', f_meta['original_header'], flags=re.IGNORECASE)
        
        fout.write(f"{f_meta['n_atoms']}\n")
        fout.write(f"{header_new}\n")
        
        # Write original atom lines by seeking and reading from the input file
        io_info = frame_io_info[i]
        fin_orig.seek(io_info['atom_lines_offset'])
        atom_lines_data = fin_orig.read(io_info['atom_lines_len'])
        fout.write(atom_lines_data)

print(f"\nEnergy alignment process complete based on mode '{ALIGNMENT_MODE}'.")
print(f"Output written to {output_xyz_file}.")
