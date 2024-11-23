#!/bin/bash

#Read all frames of exyz and then calculate

# CP2K environment setup
source /home/chen/software/cp2k-2024.1/tools/toolchain/install/setup
export PATH=$PATH:/home/chen/software/cp2k-2024.1/exe/local

xyz_file="trj.xyz"
template_inp="test.inp"

dos2unix ${xyz_file} ${template_inp}

# Set the number of cores
num_cores=48  # You can change this value as needed

# Debugging: Check the value of num_cores
echo "Number of cores set to: $num_cores"

frame_number=0  # Current frame counter

while read -r line; do
    if [[ $line =~ ^[0-9]+$ ]]; then
        frame_number=$((frame_number + 1))
        atom_count=$line
        read -r frame_info
        # Extract Lattice information and split into three triplets
        lattice_values=$(echo $frame_info | grep -oP 'Lattice="\K[^"]+')
        IFS=' ' read -r lattice_a1 lattice_a2 lattice_a3 lattice_b1 lattice_b2 lattice_b3 lattice_c1 lattice_c2 lattice_c3 <<< "$lattice_values"
        
        # Create a new input file
        new_inp="frame${frame_number}.inp"
        cp $template_inp $new_inp

        # Modify the PROJECT line
        sed -i "s/^  PROJECT.*/  PROJECT frame${frame_number}/" $new_inp

        # Modify the CELL line, only replace between &SUBSYS and &COORD
        sed -i "/&SUBSYS/,/&COORD/c\  &SUBSYS\n    &CELL\n      A    ${lattice_a1}    ${lattice_a2}    ${lattice_a3}\n      B    ${lattice_b1}    ${lattice_b2}    ${lattice_b3}\n      C    ${lattice_c1}    ${lattice_c2}    ${lattice_c3}\n      PERIODIC XYZ #Direction(s) of applied PBC (geometry aspect)\n    &END CELL\n    &COORD" $new_inp
        
        # Clear the COORD content
        sed -i '/&COORD/,/&END COORD/c\    &COORD\n    &END COORD' $new_inp
        
        # Read atom coordinates and write to the new input file
        for ((i=0; i<atom_count; i++)); do
            read -r atom_line
            # Extract element symbol and coordinate information
            IFS=' ' read -r element x y z _ <<< "$atom_line"
            sed -i "/&COORD/a\      $element $x $y $z" $new_inp
        done
    fi
done < $xyz_file

# Running CP2K for each input file
for inp_file in frame*.inp; do
    # Debugging: Print the command that will be run
    echo "Running: mpirun -np $num_cores cp2k.popt $inp_file"
    mpirun -np $num_cores cp2k.popt $inp_file | tee "${inp_file%.inp}.out"
done

# Merging output files
merged_cell_file="merged.cell"
merged_frc_file="merged-frc-1.xyz"
merged_pos_file="merged-pos-1.xyz"

# Remove old merged files if they exist
rm -f $merged_cell_file $merged_frc_file $merged_pos_file

# Concatenate each type of file
for frame_number in $(seq 1 $frame_number); do
    cat "frame${frame_number}-1.cell" >> $merged_cell_file
    cat "frame${frame_number}-frc-1.xyz" >> $merged_frc_file
    cat "frame${frame_number}-pos-1.xyz" >> $merged_pos_file
done

# Process merged.cell to retain only the first comment line
awk '!/^#/ || !found { if (/^#/) found=1; print }' $merged_cell_file > temp && mv temp $merged_cell_file

# Clean up individual frame files and input files
for frame_number in $(seq 1 $frame_number); do
    rm -f "frame${frame_number}.inp" "frame${frame_number}-1.cell" "frame${frame_number}-frc-1.xyz" "frame${frame_number}-pos-1.xyz" "frame${frame_number}.out" "frame${frame_number}-1.restart" "frame${frame_number}-1.ener"
done