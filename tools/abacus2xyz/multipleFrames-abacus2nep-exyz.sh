#!/bin/bash
### HOW TO USE #################################################################################
### SYNTAX: ./multipleFrames-abacus2nep-exyz.sh dire_name
###     NOTE: 1).'dire_name' is the directory containing running_md.log and MD_dump file.
### Email: tang070205@proton.me if any questions
### Modified by Yuwen Zhang, Shunda Chen, Zihan Yan
### Modified by Benrui Tang
################################################################################################
#--- DEFAULT ASSIGNMENTS ---------------------------------------------------------------------
isol_ener=0     # Shifted energy, specify the value?
stress_logi=0     # Logical value for virial, true=1, false=0
#--------------------------------------------------------------------------------------------
read_dire=$1
if [ -z "$read_dire" ]; then
        echo "Your syntax is illegal, please try again"
        exit 1
fi

writ_dire="NEPdataset"; writ_file="NEP-dataset.xyz";
rm -rf $writ_dire; mkdir $writ_dire

configuration=$(pwd | awk -F'/' '{print $(NF-2)"/"$(NF-1)"/"$NF}')
syst_numb_atom=$(grep "TOTAL ATOM NUMBER" running_md.log |awk '{print $5}')
ener_values=($(grep 'etot' running_md.log |awk '{print $4}'))
if [[ $stress_logi -eq 1 ]]; then
    mdstep_lines=($(grep -n 'MDSTEP' MD_dump | awk -F: '{print $1+12+'$syst_numb_atom'}'))
else
    mdstep_lines=($(grep -n 'MDSTEP' MD_dump | awk -F: '{print $1+9+'$syst_numb_atom'}'))
fi
N_counts=$(( ${#mdstep_lines[@]} - 1 ))


for ((i=1; i<${#mdstep_lines[@]}; i++)); do
    start_line=${mdstep_lines[i-1]}
    end_line=${mdstep_lines[i]}
    ener=${ener_values[i]}

    sed -n "${start_line},${end_line}p" MD_dump > temp.file
    echo "$syst_numb_atom" >> "$writ_dire/$writ_file"
    latt=$(grep -A 3 "LATTICE_VECTORS" temp.file | tail -n 3 | awk '{for (i = 1; i <= NF; i++) {printf "%.8f ", $i}}' |xargs)
    if [[ $stress_logi -eq 1 ]]; then
        stress=$(grep -A 3 "VIRIAL (kbar)" temp.file | tail -n 3 | awk '{for (i = 1; i <= NF; i++) {printf "%.8f ", $i * 0.1}}' |xargs)
        echo "Config_type=$configuration Weight=1.0 Lattice=\"$latt\" Energy=$ener Stress=\"$stress\" pbc=\"T T T\" Properties=species:S:1:pos:R:3:forces:R:3" >> "$writ_dire/$writ_file"
    else
        echo "Config_type=$configuration Weight=1.0 Lattice=\"$latt\" Energy=$ener Properties=species:S:1:pos:R:3:forces:R:3 pbc=\"T T T\"" >> "$writ_dire/$writ_file"
    fi
    grep -A $(($syst_numb_atom)) "INDEX" temp.file | tail -n $syst_numb_atom | awk '{print $2}' >$writ_dire/symb.tem
    grep -A $(($syst_numb_atom)) "INDEX" temp.file | tail -n $syst_numb_atom | awk '{for (i=3; i<=8; i++) printf "%.8f ", $i; printf "\n"}' > $writ_dire/posi_force.tem
    paste $writ_dire/symb.tem $writ_dire/posi_force.tem >> $writ_dire/$writ_file
    #grep -A $(($syst_numb_atom)) "INDEX" temp.file | tail -n $syst_numb_atom | awk '{print $2,$3,$4,$5,$6,$7,$8}' >$writ_dire/$writ_file
    rm -f $writ_dire/*.tem
    echo -ne "Process: ${i}/${N_counts}\r"
    
done
rm -f temp.file

echo
dos2unix "$writ_dire/$writ_file"
echo "All done."


