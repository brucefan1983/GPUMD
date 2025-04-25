#!/bin/bash
### HOW TO USE #################################################################################
### SYNTAX: ./multipleFrames-abacus2nep-exyz.sh dire_name
###     NOTE: 1).'dire_name' is the directory containing running_md.log and MD_dump file.
### Email: yanzhowang@gmail.com if any questions
### Modified by multipleFrames-outcars2nep-exyz.sh
### Modified by Benrui Tang
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

writ_dire="NEPdataset"; writ_file="NEP-dataset.xyz";
rm -rf $writ_dire; mkdir $writ_dire

configuration=$(pwd | awk -F'/' '{print $(NF-2)"/"$(NF-1)"/"$NF}')
syst_numb_atom=$(grep "TOTAL ATOM NUMBER" running_md.log |awk '{print $5}')
ener_values=($(grep 'etot' running_md.log |awk '{print $4}'))
scf_lines=($(grep -n 'STEP OF MOLECULAR DYNAMICS' running_md.log | awk -F: '{print $1}'))
scf_last=$(wc -l < running_md.log)
scf_lines+=($scf_last)
scf_nmax=$(grep 'scf_nmax' INPUT | awk '{print $2}')
mdstep_lines=($(grep -n 'MDSTEP' MD_dump | awk -F: '{print $1}'))
mdstep_last=$(wc -l < MD_dump)
mdstep_lines+=($mdstep_last)
N_counts=$(( ${#mdstep_lines[@]} - 2 ))


for ((i=1; i<$(( ${#mdstep_lines[@]} - 1 )); i++)); do
    ener=${ener_values[i]}

    scf_start=${scf_lines[i]}
    scf_end=${scf_lines[i+1]}
    scf_act=$(sed -n "${scf_start},${scf_end}p" running_md.log | grep -c "ALGORITHM")
    if [ "$scf_act" -eq "$scf_nmax" ]; then
        echo "Skipping the $i structure due to non convergence"
        echo -ne "Process: ${i}/${N_counts}\r"
        continue
    fi

    md_start=${mdstep_lines[i]}
    md_end=${mdstep_lines[i+1]}
    sed -n "${md_start},${md_end}p" MD_dump > temp.file
    echo "$syst_numb_atom" >> "$writ_dire/$writ_file"
    latt=$(grep -A 3 "LATTICE_VECTORS" temp.file | tail -n 3 | awk '{for (i = 1; i <= NF; i++) {printf "%.8f ", $i}}' |xargs)
    conversion_value=$(echo "$latt" | awk '{a1=$1; a2=$2; a3=$3; b1=$4; b2=$5; b3=$6; c1=$7; c2=$8; c3=$9;
        V=a1*(b2*c3 - b3*c2) + a2*(b3*c1 - b1*c3) + a3*(b1*c2 - b2*c1); if (V < 0) V=-V; printf "%.8f", V/1602.1766208}')
    if [[ $viri_logi -eq 1 ]]; then
        viri=$(grep -A 3 "VIRIAL (kbar)" temp.file | tail -n 3 | awk '{for (i = 1; i <= NF; i++) {printf "%.8f ", $i * '$conversion_value'}}' |xargs)
        echo "Energy=$ener Lattice=\"$latt\" Virial=\"$viri\" Config_type=$configuration-$i Weight=1.0 Properties=species:S:1:pos:R:3:forces:R:3" >> "$writ_dire/$writ_file"
    else
        echo "Energy=$ener Lattice=\"$latt\" Config_type=$configuration-$i Weight=1.0 Properties=species:S:1:pos:R:3:forces:R:3" >> "$writ_dire/$writ_file"
    fi
    grep -A $syst_numb_atom "INDEX" temp.file | tail -n $syst_numb_atom | awk '{print $2,$3,$4,$5,$6,$7,$8}' >> $writ_dire/$writ_file
    echo -ne "Process: ${i}/${N_counts}\r"
    rm -f temp.file
done

echo
dos2unix "$writ_dire/$writ_file"
echo "All done."


