#!/bin/bash
### HOW TO USE #################################################################################
### SYNTAX: ./singleFrame-abacus2nep-exyz.sh dire_name   
###     NOTE: 1).'dire_name' is the directory containing running_scf.logs
### Email: tang070205@proton.me if have questions
### Modified by Yanzhou Wang 
### Modified by Shunda Chen 
### Modified by Benrui Tang
################################################################################################
#--- DEFAULT ASSIGNMENTSts ---------------------------------------------------------------------
isol_ener=0     # Shifted energy, specify the value?
viri_logi=1     # Logical value for virial, true=1, false=0
#--------------------------------------------------------------------------------------------
read_dire=$1
if [ -z $read_dire ]; then
        echo "Your syntax is illegal, please try again"
        exit
fi
writ_dire="NEPdataset"; writ_file="NEP-dataset.xyz";
rm -rf $writ_dire; mkdir $writ_dire

N_case=$(find -L $read_dire -name "running_scf.log" | wc -l)
N_count=1
for i in `find -L $read_dire -name "running_scf.log"`
do
	     configuration=$(echo "$i" |sed 's/\/running_scf.log//g' | awk -F'/' '{print $(NF-2)"/"$(NF-1)"/"$NF}')
             syst_numb_atom=$(grep "TOTAL ATOM NUMBER" $i |awk '{print $5}')
             echo $syst_numb_atom >> $writ_dire/$writ_file
             latt=$(grep -A 3 "Lattice vectors" $i | tail -n 3 | sed 's/+//g' | awk '{ for (i=1; i<=NF; i++) printf "%.6f ", $i}')
             ener=$(grep "FINAL_ETOT_IS" $i | awk '{printf "%.6f\n", $2 - '$syst_numb_atom' * '$isol_ener'}')
             if [ $viri_logi -eq 1 ]
             then
                   conversion_value=$(grep "Volume (A^3)" $i |awk '{print $4/(1602.1766208 * '$syst_numb_atom')}')
                   viri=$(grep -A 4 "TOTAL-STRESS" $i | tail -n 3 | awk '{for (i = 1; i <= NF; i++) {printf "%.6f ", $i * '$conversion_value'}}' |xargs)
                   echo Energy=$ener Lattice=\"$latt\" Virial=\"$viri\" "Config_type=$configuration Weight=1.0 Properties=species:S:1:pos:R:3:forces:R:3" >> $writ_dire/$writ_file
             else
                   echo Energy=$ener Lattice=\"$latt\" "Config_type=$configuration Weight=1.0 Properties=species:S:1:pos:R:3:forces:R:3" >> $writ_dire/$writ_file
             fi
             taucx_values=$(grep "tauc" running_scf.log | awk '{print $2}')
             max_taucx=0
             for value in $tauc_values; do
               if (( $(echo "$value > $max_taucx" | bc -l) )); then
                 max_taucx=$value
                 fi
             done
             if (( $(echo "$max_tauc < 1" | bc -l) ))
             then
               scf_dir=$(echo "$i" | sed 's/\/running_scf.log//g')
               cif_file="${scf_dir}/STRU.cif"
               cell_a=$(grep "_cell_length_a" "$cif_file" | awk '{print $2}')
               cell_b=$(grep "_cell_length_b" "$cif_file" | awk '{print $2}')
               cell_c=$(grep "_cell_length_c" "$cif_file" | awk '{print $2}')
               grep "tauc" $i | awk '{printf "%.6f %.6f %.6f\n", $2*'$cell_a', $3*'$cell_b', $4*'$cell_c'}' > $writ_dire/position.tem
             else
               grep "tauc" $i | awk '{printf "%.6f %.6f %.6f\n", $2, $3, $4}' > $writ_dire/position.tem
             fi
             grep -A $(($syst_numb_atom + 1)) "TOTAL-FORCE" $i | tail -n $syst_numb_atom | awk '{print $1}' | sed 's/[0-9]//g' >>$writ_dire/symb.tem
             grep -A $(($syst_numb_atom + 1)) "TOTAL-FORCE" $i | tail -n $syst_numb_atom |awk '{printf "%.6f %.6f %.6f\n", $2,$3,$4}' > $writ_dire/force.tem
             paste $writ_dire/symb.tem $writ_dire/position.tem $writ_dire/force.tem >> $writ_dire/$writ_file
             rm -f $writ_dire/*.tem
	     echo -ne "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\bProgress: $N_count/$N_case "
             N_count=$((N_count + 1))
done
echo
dos2unix $writ_dire/$writ_file
