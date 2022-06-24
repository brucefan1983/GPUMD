#!/bin/bash
### HOW TO USE #################################################################################
### SYNTAX: ./outcars2nepDataset_v5.sh dire_name 	
###	NOTE: 1).'dire_name' is the directory containing OUTCARs
###	      2). The 'outcar2nepDataset_v5' is executable
### An example: https://github.com/Yanzhou-Wang/bash_tools
### Email: yanzhowang@gmail.com if any questions
################################################################################################
#--- DEFAULT ASSIGNMENTSts ---------------------------------------------------------------------
isol_ener=0     # Shifted energy, specify the value ???????????????????????????????????
viri_logi=1     # Logical value for virial, true=1, false=0 ????????????????????????????
#--------------------------------------------------------------------------------------------
#receive positonal parameter
read_dire=$1
if [ -z $read_dire ]; then
	echo "Your syntax is illegal, please try again"
	exit
fi
writ_dire="NEPdataset"; writ_file="dataset.nep"; writ_file_body="body_dataset.nep"; writ_file_head="head_dataset.nep"
rm -rf $writ_dire; mkdir $writ_dire
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for i in `find $read_dire -name "OUTCAR"`
do
	# Part 1. print sequentially the list containing size and virial info for evergy single confs
	conf_size=$(grep "number of ions" $i |awk '{print $12}')
        echo "$conf_size $viri_logi" >> $writ_dire/$writ_file_head
	# Part 2. Collect information like energy, virial, lattice, element symbol, atomic position and corresponding force  
        # -----2.1 Extract energy and virial
	ener=$(grep "free  energy   TOTEN" $i | tail -1 | awk '{printf "%.6f\n", $5 - '$conf_size' * '$isol_ener'}')
	if [[ $viri_logi -eq 1 ]]
	then
        	viri=$(grep -A 13 "FORCE on cell =-STRESS" $i |tail -n 1 | awk '{print $2,$3,$4,$5,$6,$7}')
        	echo "$ener     $viri" >> $writ_dire/$writ_file_body
	else
		echo "$ener" >> $writ_dire/$writ_file_body
	fi
	# ----2.2 Extract lattice
        grep -A 7 "VOLUME and BASIS-vectors are now" $i |tail -n 3 |awk '{print $1,$2,$3}' |xargs >> $writ_dire/$writ_file_body
	# ----2.3 Prepare with symbol list to record all elements involed
	# --------2.3.1 Extract seqentially number of ions per type in a row
	grep "ions per type"  $i | tail -1 | awk -F"=" '{print $2}'> $writ_dire/ions.tem
	# --------2.3.2 Extract sequence of symbols for elements and creat symbol list
	index=1
	for symb in $(grep "VRHFIN" $i | awk -F"=" '{print $2}' |awk -F":" '{print $1}')
	do
		numb=$(awk '{print $'$index'}' $writ_dire/ions.tem)
		for((j=1;j<=$numb;j++))
		do
			echo "$symb" >> $writ_dire/symb_list.tem
		done 
		index=$(($index + 1))
	done
	# ----2.4 Extract position cocordinates and corresponding forces 
        grep -A $((conf_size + 1)) "TOTAL-FORCE (eV/Angst)" $i | tail -n $conf_size > $writ_dire/posi_forc.tem
	# Assemble energy, virial, lattice, symbol,positons and forces into one file
	paste $writ_dire/symb_list.tem $writ_dire/posi_forc.tem >> $writ_dire/$writ_file_body
	rm -f $writ_dire/*.tem
	echo "-------- case: $i done, next ... -------------------"
done
# Part 3. line 1 of nep dataset, record total number of confs
data_set_size=$(wc -l < $writ_dire/$writ_file_head)
sed -i '1i '$data_set_size'' $writ_dire/$writ_file_head

# Finally concatenate head and body parts
cat $writ_dire/$writ_file_head $writ_dire/$writ_file_body > $writ_dire/$writ_file
rm $writ_dire/$writ_file_head $writ_dire/$writ_file_body
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
echo "All done, bye."
