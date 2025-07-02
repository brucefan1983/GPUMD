#!/bin/bash
### HOW TO USE #################################################################################
### SYNTAX: ./singleFrame-abacus2nep-exyz.sh dire_name   
###     NOTE: 1).'dire_name' is the directory containing running_scf.logs
### Email: tang070205@proton.me if have questions
### Modified by Benrui Tang
################################################################################################
#--- DEFAULT ASSIGNMENTSts ---------------------------------------------------------------------
isol_ener=0     # Shifted energy, specify the value?
#--------------------------------------------------------------------------------------------
read_dire=$1
if [ -z $read_dire ]; then
        echo "Your syntax is illegal, please try again"
        exit
fi
writ_dire="NEPdataset"; writ_file="NEP-dataset.xyz"; py_file="extract_positions.py"
rm -rf $writ_dire $py_file; mkdir $writ_dire
cat > "$py_file" <<EOF
from ase.io import read
import sys
atoms = read(sys.argv[1])
positions = atoms.get_positions()
with open(sys.argv[2], "w") as f:
    for pos in positions:
        f.write(f"{pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}\n")
EOF

N_case=$(find -L $read_dire -name "running_scf.log" | wc -l)
N_count=1
for i in `find -L "$read_dire" -name "running_scf.log"`; do
    configuration=$(echo "$i" | sed 's/\/running_scf.log//g' | awk -F'/' '{print $(NF-2)"/"$(NF-1)"/"$NF}')
    scf_act=$(grep -c "ALGORITHM" "$i")
    scf_nmax=$(grep 'scf_nmax' "$(dirname "$i")/INPUT" | awk '{print $2}')
    if [ "$scf_act" -eq "$scf_nmax" ] || ! grep -q "FINAL_ETOT_IS" "$i"; then
    echo "Directory of incomplete or non-converged locations: $(dirname "$(realpath "$i")")"
    continue
    fi

    ase_version=$(pip list | grep ase) && ase_available="yes" || ase_available="no"
    if [ "$ase_available" == "yes" ]; then
        python3 "$py_file" "$(dirname "$i")/STRU.cif" "$writ_dire/position.tem"
        latt=$(python3 -c "from ase.io import read; atoms = read('$(dirname "$i")/STRU.cif'); print(' '.join(['{:.10f}'.format(x) for x in atoms.cell.flatten()]))")
    else
        if [ -f "$(dirname "$i")/../STRU" ]; then
            stru_file_path="$(dirname "$i")/../STRU"
        else
            stru_file_path=$(grep 'stru_file' INPUT | awk '{print $2}')
        fi
        latt=$(grep -A 3 "LATTICE_VECTORS" "$stru_file_path" | tail -n 3 | awk '{ for (i=1; i<=NF; i++) printf "%.10f ", $i}')
        if grep -q "taud" "$i"; then
            grep "taud" "$i" | awk -v l="$latt" '{split(l,v," "); printf "%.10f %.10f %.10f\n", $2*v[1]+$3*v[4]+$4*v[7], $2*v[2]+$3*v[5]+$4*v[8], $2*v[3]+$3*v[6]+$4*v[9]}' > "$writ_dire/position.tem"
        else
            grep "tauc" "$i" | awk '{printf "%.10f %.10f %.10f\n", $2, $3, $4}' > "$writ_dire/position.tem"
        fi
    fi

    syst_numb_atom=$(grep "TOTAL ATOM NUMBER" "$i" | awk '{print $5}')
    echo "$syst_numb_atom" >> "$writ_dire/$writ_file"
    ener=$(grep "FINAL_ETOT_IS" "$i" | awk '{printf "%.6f\n", $2 - '$syst_numb_atom' * '$isol_ener'}')

    if grep -q "TOTAL-STRESS" "$i"; then
        conversion_value=$(grep "Volume (A^3)" "$i" | awk '{print $4/1602.1766208}')
        viri=$(grep -A 4 "TOTAL-STRESS" "$i" | tail -n 3 | awk '{for (i = 1; i <= NF; i++) {printf "%.10f ", $i * '$conversion_value'}}' | xargs)
        echo "Energy=$ener Lattice=\"$latt\" Virial=\"$viri\" Config_type=$configuration Weight=1.0 Properties=species:S:1:pos:R:3:forces:R:3" >> "$writ_dire/$writ_file"
    else
        echo "Energy=$ener Lattice=\"$latt\" Config_type=$configuration Weight=1.0 Properties=species:S:1:pos:R:3:forces:R:3" >> "$writ_dire/$writ_file"
        echo "Warning: No virial found in $(dirname "$i")."
    fi

    grep -A $(($syst_numb_atom + 1)) "TOTAL-FORCE" "$i" | tail -n "$syst_numb_atom" | awk '{print $1}' | sed 's/[0-9]//g' >> "$writ_dire/symb.tem"
    grep -A $(($syst_numb_atom + 1)) "TOTAL-FORCE" "$i" | tail -n "$syst_numb_atom" | awk '{printf "     %.10f %.10f %.10f\n", $2,$3,$4}' > "$writ_dire/force.tem"
    paste -d'	    ' "$writ_dire/symb.tem" "$writ_dire/position.tem" "$writ_dire/force.tem" >> "$writ_dire/$writ_file"

    rm -f "$writ_dire"/*.tem
    echo -ne "Progress: $N_count/$N_case \r"
    N_count=$((N_count + 1))
done
rm -f "$py_file"
echo
dos2unix $writ_dire/$writ_file
echo "All done."
