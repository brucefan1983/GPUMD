#!/bin/bash
############################# How to use #######################################################
### Syntax: ./script dire_name	################################################################
### 1. Assume the 'script' is exectable	########################################################	
### 2.'dire_name' is the directory where one stores vasp cases for nep-dataset #################
### Email: yanzhowang@gmail.com if any problem	################################################
### Yanzhou Wang created: 25.11.2021 ###########################################################
### Script version: 3		###############################################################
################################################################################################
#============================ Caveates =========================================================
# 1. vasp cases in specified 'dire_name' must contain 'OUTCAR'	================================
# 2. Orgnizations of vasp cases in 'dire_name' must be as follows: =============================
#       'dire_name' 						================================	
#	     |__case1 (case directory)	          		================================
#	     |__case2 (case directory)		        	================================	
#	     |__....  (...)			                ================================
#===============================================================================================
#-----------------------Assignments-----------------------------------------------------------
read_dire=$1
if [ -z $read_dire ]; then
	echo "Your systax is illegal, please try again"
	exit
fi

writ_dire="nep-dataset"; writ_file="dataset.nep"; writ_file_body="body_dataset.nep"; writ_file_head="head_dataset.nep"
rm -rf $writ_dire; mkdir $writ_dire

isol_ener=-1.37104835	# Shifted energy, one can specify the value
viri_logi=1	# Logical value for virial, here means virial contains in the dataset
#---------------------------------------------------------------------------------------------
#+++++++++++++++++++++++ code region (don't edit it) ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for i in `ls $read_dire`
do
	# Part 1. Collect all configuration size and whether virial or not
	conf_size=$(grep "number of ions" $read_dire/$i/OUTCAR |awk '{print $12}')
        echo "$conf_size $viri_logi" >> $writ_dire/$writ_file_head

	# Part 2. Collect information like energy, virial, lattice, element sysbol, atomic position and corresponding force  
        ener=$(grep "free  energy   TOTEN" $read_dire/$i/OUTCAR | tail -1 | awk '{printf "%.6f\n", $5 - '$conf_size' * '$isol_ener'}')
        viri=$(grep -A 14 "FORCE on cell =-STRESS" $read_dire/$i/OUTCAR |tail -n 1 | awk '{print $2,$3,$4,$5,$6,$7}')
        echo "$ener     $viri" >> $writ_dire/$writ_file_body
        grep -A 7 "VOLUME and BASIS-vectors are now" $read_dire/$i/OUTCAR |tail -n 3 |awk '{print $1,$2,$3}' |xargs >> $writ_dire/$writ_file_body
	
	index=1
	grep "ions per type"  $read_dire/$i/OUTCAR | tail -1 | awk -F"=" '{print $2}'> $writ_dire/ions.tem
	for symb in $(grep "VRHFIN" $read_dire/$i/OUTCAR | awk -F"=" '{print $2}' |awk -F":" '{print $1}')
	do
		numb=$(awk '{print $'$index'}' $writ_dire/ions.tem)
		for((j=1;j<=$numb;j++))
		do
			echo "$symb" >> $writ_dire/symb_list.tem
		done 
		index=$(($index + 1))
	done

        grep -A $((conf_size + 1)) "TOTAL-FORCE (eV/Angst)" $read_dire/$i/OUTCAR | tail -n $conf_size > $writ_dire/posi_forc.tem
	paste $writ_dire/symb_list.tem $writ_dire/posi_forc.tem >> $writ_dire/$writ_file_body
	rm -f $writ_dire/*.tem

	echo "-------- case: $i done, next ... -------------------"
done
# Part 3. line 1 of nep dataset
data_set_size=$(wc -l < $writ_dire/$writ_file_head)
sed -i '1i '$data_set_size'' $writ_dire/$writ_file_head

# Finally concatenate two head and body parts
cat $writ_dire/$writ_file_head $writ_dire/$writ_file_body > $writ_dire/$writ_file
rm $writ_dire/$writ_file_head $writ_dire/$writ_file_body
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
echo "All done, bye."
