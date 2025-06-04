"""
Unified Atomic Energy Baseline Aligner for XYZ Trajectory Files

This script provides a flexible tool for aligning energies within multi-frame XYZ trajectory files.
It supports three distinct energy alignment modes, configurable via the 'ALIGNMENT_MODE' setting:

1.  'REF_GROUP_ALIGNMENT':
    Purpose: To align the energies of specified 'shift_groups' to the average energy baseline
             of a designated 'reference_group'. This is useful when you have a high-accuracy
             dataset (reference_group) and want to shift other datasets (shift_groups) to its scale.
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
             calculated by a Neural Exchange-correlation Potential (NEP) model. This is useful
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
ALIGNMENT_MODE = 'REF_GROUP_ALIGNMENT' 

# --- Mode-Specific Settings ---

# For 'REF_GROUP_ALIGNMENT' mode ONLY:
# These settings will be ignored if ALIGNMENT_MODE is not 'REF_GROUP_ALIGNMENT'.
reference_group = "aimd"
shift_groups = []  # If empty, all config_types except reference_group will be optimized.

# For 'DFT_TO_NEP_ALIGNMENT' mode ONLY:
# This setting will be ignored if ALIGNMENT_MODE is not 'DFT_TO_NEP_ALIGNMENT'.
nep_model_file = 'nep.txt' # Path to your NEP model file (e.g., 'nep.txt')

# --- NES Hyperparameters (apply to all modes using NES) ---
max_generations = 10000
population_size = 40
convergence_tol = 1e-8
random_seed = 42

#######################
#   Helper Functions  #
#######################

def parse_xyz_frames(xyz_filename):
    """
    Parses XYZ file to extract primary energy, config_type, element counts,
    original header, and original atom lines (including positions and forces).
    Also returns ASE Atoms objects for NEP calculation (if needed by mode).
    Assumes 'energy=' is the primary energy field.
    """
    frames_metadata = []
    # ase_atoms_list is only needed for DFT_TO_NEP_ALIGNMENT mode, but we parse it always for simplicity.
    ase_atoms_list = read(xyz_filename, index=':') 
    
    with open(xyz_filename, "r") as fin:
        lines = fin.readlines()
    
    i = 0
    frame_idx = 0
    while i < len(lines):
        n_atoms = int(lines[i].strip())
        header = lines[i + 1].strip()

        # Extract primary energy (e.g., DFT energy)
        m_energy = re.search(r'energy=([-\d\.Ee]+)', header, re.IGNORECASE)
        primary_energy = float(m_energy.group(1)) if m_energy else None
        if primary_energy is None:
            raise ValueError(f"No 'energy=' found in header: {header} for frame {frame_idx}. This field is required.")

        # Extract config_type
        m_config = re.search(r'config_type="?([^\s"]+)"?', header, re.IGNORECASE)
        config_type = m_config.group(1) if m_config else "default_group"

        elem_counts = Counter()
        original_atom_lines = [] 
        
        for j in range(i + 2, i + 2 + n_atoms):
            line_content = lines[j]
            elem_symbol = line_content.split()[0]
            elem_counts[elem_symbol] += 1 
            original_atom_lines.append(line_content) # Store the full line as-is
        
        frames_metadata.append({
            'primary_energy': primary_energy, # This is the energy that will be shifted
            'config_type': config_type,
            'elem_counts': elem_counts,
            'original_header': header, 
            'n_atoms': n_atoms,
            'atoms_obj_idx': frame_idx, # Link to ase_atoms_list for NEP calculation if needed
            'original_atom_lines': original_atom_lines 
        })
        
        i += 2 + n_atoms
        frame_idx += 1
    
    return frames_metadata, ase_atoms_list

def calculate_nep_worker(ase_atoms_obj, nep_model_file_path):
    """
    Worker function for parallel NEP calculation.
    Initializes CPUNEP calculator within the worker process.
    """
    worker_calc = CPUNEP(nep_model_file_path)
    ase_atoms_obj.calc = worker_calc
    energy = ase_atoms_obj.get_potential_energy()
    forces = ase_atoms_obj.get_forces() 
    return energy, forces

def atomic_baseline_cost(param_population, source_energies, element_counts, target_energies=None):
    """
    Generic cost function for NES: minimize MSE of ( (source_energy - sum(n_X * E_X)) - target_energy )^2.
    If target_energies is None, it minimizes MSE of (shifted_source_energy)^2 (i.e., targets zero).
    Args:
        param_population: [n_pop, n_elem] -- current population of atomic baseline parameters
        source_energies: [n_samples,] -- energies to be shifted
        element_counts: [n_samples, n_elem] -- element counts for each structure
        target_energies: [n_samples,] or None -- energies to align *to*. If None, aligns to zero.
    Returns:
        cost: [n_pop, 1] -- MSE cost for each set of parameters in the population
    """
    shifted_source_energies = source_energies[None, :] - np.dot(param_population, element_counts.T)
    
    if target_energies is not None:
        cost = np.mean((shifted_source_energies - target_energies[None, :]) ** 2, axis=1).reshape(-1, 1)
    else: # Align to zero baseline
        cost = np.mean(shifted_source_energies ** 2, axis=1).reshape(-1, 1)
    return cost

def nes_optimize_atomic_baseline(
        num_variables,              # Number of atomic species (parameters to optimize)
        max_generations,            # Maximum NES iterations
        source_energies,            # [n_samples,] -- Energies to be shifted
        element_counts,             # [n_samples, n_variables] -- Element counts for each structure
        target_energies=None,       # [n_samples,] or None -- Energies to align *to*. If None, aligns to zero.
        pop_size=40,                # NES population size
        tol=1e-8,                   # Early stopping threshold
        seed=42,                    # Random seed for reproducibility
        print_every=100             # Print progress every N generations
):
    """
    NES optimizer for finding atomic reference energies.
    Returns:
        best_fitness_history: [actual_gens, 1] -- best fitness per generation
        elite_solutions: [actual_gens, num_variables] -- best solution per generation
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

##########################
#        MAIN LOGIC      #
##########################

# Ensure full reproducibility by setting seeds globally
np.random.seed(random_seed)
import random
random.seed(random_seed)

# 1. Parse XYZ file to get metadata and ASE Atoms objects
print(f"Parsing '{input_xyz_file}' for energies and structure metadata...")
frames_metadata, ase_atoms_list = parse_xyz_frames(input_xyz_file)

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
    print(f"Reference Group: '{reference_group}'")
    
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

    # Set target for NES to the mean energy of the reference group
    # This will be passed to nes_optimize_atomic_baseline for each group
    # A placeholder will be created for each group's specific array.
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

    # Calculate NEP energies for all structures in parallel
    print(f"Calculating NEP energies and forces using '{nep_model_file}'...")
    num_cores = cpu_count()
    print(f"Using {num_cores} cores for parallel NEP calculation...")

    # Prepare arguments for multiprocessing: pass ASE Atoms objects directly
    nep_calculation_args = [(atoms_obj, nep_model_file) for atoms_obj in ase_atoms_list]

    with Pool(num_cores) as pool:
        nep_results = pool.starmap(calculate_nep_worker, nep_calculation_args)

    # Integrate calculated NEP energies back into frames_metadata. Forces are ignored for output.
    for i, (nep_energy, _) in enumerate(nep_results): 
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
with open(output_xyz_file, "w") as fout:
    for f_meta in frames_metadata:
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
                # This warning applies if a config_type was detected but not in groups_to_process (e.g., if it was an empty group)
                print(f"Warning: config_type '{f_meta['config_type']}' was not processed by NES in this mode. Its energy will remain as original.")
            
        # Reconstruct header with updated energy. 
        # The 'energy=' field in the output now contains the ALIGNED primary energy.
        # CORRECTED TYPO: re.re.sub -> re.sub
        header_new = re.sub(r'energy=([-\d\.Ee]+)', f'energy={shifted_energy:.8f}', f_meta['original_header'], flags=re.IGNORECASE)
        
        fout.write(f"{f_meta['n_atoms']}\n")
        fout.write(f"{header_new}\n")
        
        # Write original atom lines (including original forces/coordinates)
        for atom_line in f_meta['original_atom_lines']:
            fout.write(atom_line)

print(f"\nEnergy alignment process complete based on mode '{ALIGNMENT_MODE}'.")
print(f"Output written to {output_xyz_file}.")