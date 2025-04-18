#!/bin/bash
### HOW TO USE #################################################################################
### SYNTAX: ./outcar2nep-exyz.sh dire_name
###     NOTE: 1).'dire_name' is the directory containing OUTCARs
### Email: yanzhowang@gmail.com if any questions
### Modified by Yuwen Zhang
### Modified by Shunda Chen
### Modified by Zihan Yan
################################################################################################
#--- DEFAULT ASSIGNMENTS ---------------------------------------------------------------------
isol_ener=0     # Shifted energy, specify the value?
viri_logi=1     # Logical value for virial, true=1, false=0
#--------------------------------------------------------------------------------------------
read_dire=$1
if [ -z "$read_dire" ]; then
        echo "Your syntax is illegal, please try again"
        exit 1
fi

writ_dire="NEPdataset-multiple_frames"
writ_file="NEP-dataset.xyz"
error_file="non_converged_files.txt"

rm -rf "$writ_dire"
mkdir "$writ_dire"
rm -f "$error_file"

root_path=$(pwd)

# Count the total number of OUTCAR files
total_outcar=$(find -L "$read_dire" -name "OUTCAR" | wc -l)
# Check for converged and non-converged OUTCAR files
converged_files=()
non_converged_files=()

echo "Checking the convergence of OUTCARs ..."

for file in $(find "$read_dire" -name "OUTCAR"); do
    NSW=$(grep "number of steps for IOM" "$file" | awk '{print $3}')
    
    if [ "$NSW" -ne 0 ]; then
        converged_files+=("$file")
        continue
    fi
    
    NELM=$(grep "of ELM steps" "$file" | awk '{print $3}' | tr -d ';')
    actual_steps=$(grep -c "Iteration" "$file")

    if grep -q "aborting loop because EDIFF is reached" "$file"; then
        if [ "$actual_steps" -lt "$NELM" ]; then
            converged_files+=("$file")
        else
            non_converged_files+=("$file")
        fi
    else
        non_converged_files+=("$file")
    fi
done

total_converged=${#converged_files[@]}
total_non_converged=${#non_converged_files[@]}

echo "Total OUTCAR files: $total_outcar"
echo "Converged OUTCAR files: $total_converged"
echo "Non-converged OUTCAR files: $total_non_converged"

# Write non-converged OUTCAR file paths to the error file
if [ $total_non_converged -gt 0 ]; then
    printf "%s\n" "${non_converged_files[@]}" > "$error_file"
fi

# Process converged OUTCAR files
N_count=1
for file in "${converged_files[@]}"; do
    configuration=$(basename "$(dirname "$file")")
    start_lines=($(sed -n '/aborting loop because EDIFF is reached/=' "$file"))
    end_lines=($(sed -n '/[^ML] energy  without entropy/=' "$file"))
    ion_numb_arra=($(grep "ions per type" "$file" | tail -n 1 | awk -F"=" '{print $2}'))
    ion_symb_arra=($(grep "POTCAR:" "$file" | awk '{print $3}' | awk -F"_" '{print $1}' | awk '!seen[$0]++'))
    syst_numb_atom=$(grep "number of ions" "$file" | awk '{print $12}')

    k=0
    for ((i=0; i<${#start_lines[@]}; i++)); do
        for ((j=k; j<${#end_lines[@]}; j++)); do
            if [ ${start_lines[i]} -lt ${end_lines[j]} ]; then
                k=$j
                break
            fi
        done
        start_line=${start_lines[i]}
        end_line=${end_lines[k]}

        sed -n "${start_line},${end_line}p" "$file" > temp.file
        echo "$syst_numb_atom" >> "$writ_dire/$writ_file"
        latt=$(grep -A 7 "VOLUME and BASIS-vectors are now" temp.file | tail -n 3 | sed 's/-/ -/g' | awk '{print $1,$2,$3}' | xargs)
        ener=$(grep "free  energy   TOTEN" temp.file | tail -1 | awk '{printf "%.10f\n", $5 - '"$syst_numb_atom"' * '"$isol_ener"'}')

        if [[ $viri_logi -eq 1 ]]; then
            viri=$(grep -A 20 "FORCE on cell =-STRESS" temp.file | grep "Total " | tail -n 1 | awk '{print $2,$5,$7,$5,$3,$6,$7,$6,$4}')
            echo "Config_type=$configuration Weight=1.0 Lattice=\"$latt\" Energy=$ener Virial=\"$viri\" pbc=\"T T T\" Properties=species:S:1:pos:R:3:forces:R:3" >> "$writ_dire/$writ_file"
        else
            echo "Config_type=$configuration Weight=1.0 Lattice=\"$latt\" Properties=species:S:1:pos:R:3:forces:R:3 Energy=$ener pbc=\"T T T\"" >> "$writ_dire/$writ_file"
        fi

        for((j=0;j<${#ion_numb_arra[*]};j++));do
            printf ''${ion_symb_arra[j]}'%.0s\n' $(seq 1 1 ${ion_numb_arra[j]}) >> $writ_dire/symb.tem
        done

        grep -A $((syst_numb_atom + 1)) "TOTAL-FORCE (eV/Angst)" temp.file | tail -n $syst_numb_atom > "$writ_dire/posi_forc.tem"
        paste "$writ_dire/symb.tem" "$writ_dire/posi_forc.tem" >> "$writ_dire/$writ_file"
        rm -f "$writ_dire"/*.tem
    done
    rm -f temp.file

    # Display progress bar
    progress=$((N_count * 100 / total_converged))
    echo -ne "Progress: ["
    for ((p=0; p<progress/2; p++)); do echo -ne "#"; done
    for ((p=progress/2; p<50; p++)); do echo -ne "."; done
    echo -ne "] $progress% ($N_count/$total_converged)\r"
    N_count=$((N_count + 1))
done

echo -ne "\nConversion complete.\n"
dos2unix "$writ_dire/$writ_file"
echo "All done."
