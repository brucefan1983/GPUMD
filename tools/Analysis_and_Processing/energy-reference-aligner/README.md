## Unified Atomic Energy Baseline Aligner for XYZ Trajectory Files

---
For xyz files generated under the Windows system, please run dos2unix *.xyz before running the program.
This script provides a flexible tool for aligning energies within multi-frame XYZ trajectory files. It supports three distinct energy alignment modes, configurable via the ALIGNMENT_MODE setting:

## Alignment Modes

### 1. REF_GROUP_ALIGNMENT

- **Purpose**: Aligns the energies of specified shift_groups to the average energy baseline of a designated reference_group. This is useful when you have a high-accuracy dataset (reference_group) and want to shift other datasets (shift_groups) to its scale.
- **Method**: For each shift group, atomic energy baselines (one per element type) are optimized by minimizing the mean squared error (MSE) between shifted group energies and the mean energy of the reference group.
- **Settings Used**: reference_group, shift_groups. nep_model_file is ignored.

### 2. ZERO_BASELINE_ALIGNMENT

- **Purpose**: Shifts all energies in the XYZ file such that the calculated energy of free atoms (based on optimized atomic baselines) would be zero. This is a common practice for standardizing energies to a vacuum reference.
- **Method**: For each detected config_type group, atomic energy baselines are optimized by minimizing the MSE of the shifted energies themselves (effectively aligning to zero).
- **Settings Used**: None of reference_group, shift_groups, or nep_model_file are used for logic.

### 3. DFT_TO_NEP_ALIGNMENT

- **Purpose**: Aligns the original DFT energies (read from energy= field) to the energies calculated by a Neuroevolution Potential (NEP) model. This is useful for standardizing DFT data to the scale of a force field or machine learning potential.
- **Method**: First, NEP energies are calculated for all structures using an external NEP model file. Then, for each config_type group, atomic energy baselines are optimized to minimize the MSE between the shifted DFT energies and the calculated NEP energies.
- **Settings Used**: nep_model_file. reference_group and shift_groups are ignored.

## Key Features

- **Flexible Mode Selection**: Easily switch between alignment strategies.
- **Robust XYZ Parsing**: Handles case-insensitive header keys (e.g., Energy= vs energy=) and supports quoted or unquoted config_type values.
- **Parallel NEP Calculation**: Leverages multiprocessing for efficient NEP energy computation in DFT_TO_NEP_ALIGNMENT mode.
- **Optimized NES Algorithm**: Uses Natural Evolution Strategy for baseline optimization, featuring vectorized calculations, early stopping for efficiency, and full reproducibility (fixed random seeds).
- **Preserves Original Data**: Only the energy= field in the XYZ header is modified. Original atomic coordinates, forces (if present in input), and other header fields are preserved.
- **Intelligent Group Handling**: Automatically detects all config_type groups. Ignores irrelevant settings based on the chosen mode.

## How to Use

1. **Configure File Paths**: Set input_xyz_file and output_xyz_file.
2. **Select Alignment Mode**: Set ALIGNMENT_MODE to one of the three options: REF_GROUP_ALIGNMENT, ZERO_BASELINE_ALIGNMENT, or DFT_TO_NEP_ALIGNMENT.
3. **Adjust Mode-Specific Settings**:
   - If REF_GROUP_ALIGNMENT: Define reference_group. Optionally, define shift_groups (if empty, all non-reference groups will be processed).
   - If DFT_TO_NEP_ALIGNMENT: Provide the path to your NEP model file (nep_model_file).
4. **Run the Script**: Execute the Python script. Progress and optimized atomic baselines will be printed to the console. The final aligned energies will be written to the output_xyz_file.

## Author

Chen Zherui (chenzherui0124@foxmail.com)
