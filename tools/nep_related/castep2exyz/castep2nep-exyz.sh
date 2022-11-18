#!/bin/bash
### HOW TO USE #################################################################################
### SYNTAX: ./castep2nep-exyz dire_name   
###     NOTE: 1).'dire_name' is the directory containing OUTCARs
### Email: yanzhowang@gmail.com if any questions
################################################################################################
#--- DEFAULT ASSIGNMENTSts ---------------------------------------------------------------------
isol_ener=0     # Shifted energy, specify the value?
#-----------------------------------------------------------------------------------------------
#grep "[ ]*Current cell volume" CN_relaxiation/nhg.castep |awk -F"=" '{print $2}' |awk '{print $1}'

read_dire=$1
if [ -z $read_dire ]; then
        echo "Your syntax is illegal, please try again"
        exit
fi
writ_dire="NEPdataset"; writ_file="NEP-dataset.xyz";
rm -rf $writ_dire; mkdir $writ_dire

N_case=$(find -L $read_dire -name "*.castep" | wc -l)
N_count=1
for i in `find -L $read_dire -name "*.castep"`
do
	dos2unix $i > /dev/null 2>&1
	N_lattice_vector=$(grep -A 5 "Unit Cell" $i |tail -n 3 |xargs |wc -w)
	if [[ $N_lattice_vector -eq 18 ]]
	then
		syst_numb_atom=$(grep "Total number of ions in cell" $i |awk '{print $8}')
		echo $syst_numb_atom >> $writ_dire/$writ_file
		latt=$(grep -A 5 "Unit Cell" $i |tail -n 3 |awk '{print $1,$2,$3}' |xargs)
		ener=$(grep "Final energy" $i | tail -n 1 | awk '{printf "%.6f\n", $4 - '$syst_numb_atom' * '$isol_ener'}')
		viri=$(grep -A 8 "^[* ][* ]*Stress Tensor" $i)
		if [[ -n "$viri" ]]
		then
			viri=$(grep -A 8 "^[* ][* ]*Stress Tensor" $i |tail -n 3 |awk '{print $3,$4,$5}' |xargs)
			volume=$(grep "[ ]*Current cell volume" $i |tail -n 1 |awk -F"=" '{print $2}' |awk '{print $1}')
			factor_GPa2eV=160.217662
			viri=$(echo $viri | awk '{printf("%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", $1*'$volume'/'$factor_GPa2eV', $2*'$volume'/'$factor_GPa2eV', $3*'$volume'/'$factor_GPa2eV', $4*'$volume'/'$factor_GPa2eV', $5*'$volume'/'$factor_GPa2eV', $6*'$volume'/'$factor_GPa2eV', $7*'$volume'/'$factor_GPa2eV', $8*'$volume'/'$factor_GPa2eV', $9*'$volume'/'$factor_GPa2eV')}')
			echo "Lattice=\"$latt\" Energy=$ener Virial=\"$viri\" Properties=species:S:1:pos:R:3:force:R:3" >> $writ_dire/$writ_file
		else
			echo "Lattice=\"$latt\" Energy=$ener Properties=species:S:1:pos:R:3:force:R:3" >> $writ_dire/$writ_file
		fi
		N_after_coor=$(($syst_numb_atom + 2))
		grep -A $N_after_coor "Fractional coordinates of atoms" $i | tail -n $syst_numb_atom |awk '{print $2}' > $writ_dire/symb.tem
		grep -A $N_after_coor "Fractional coordinates of atoms" $i | tail -n $syst_numb_atom |awk '{print $4,$5,$6}' > $writ_dire/posi_frac.tem
		#fraction to direct coordinates
		ax=$(echo $latt | awk '{print $1}')
		ay=$(echo $latt | awk '{print $2}')
		az=$(echo $latt | awk '{print $3}')
		bx=$(echo $latt | awk '{print $4}')
		by=$(echo $latt | awk '{print $5}')
		bz=$(echo $latt | awk '{print $6}')
		cx=$(echo $latt | awk '{print $7}')
		cy=$(echo $latt | awk '{print $8}')
		cz=$(echo $latt | awk '{print $9}')
		awk '{printf("%12.6f\n", $1 * '$ax' + $2 * '$bx' + $3 * '$cx')}' $writ_dire/posi_frac.tem > $writ_dire/posi_cart_x.tem
		awk '{printf("%12.6f\n", $1 * '$ay' + $2 * '$by' + $3 * '$cy')}' $writ_dire/posi_frac.tem > $writ_dire/posi_cart_y.tem
		awk '{printf("%12.6f\n", $1 * '$az' + $2 * '$bz' + $3 * '$cz')}' $writ_dire/posi_frac.tem > $writ_dire/posi_cart_z.tem
		paste $writ_dire/posi_cart_x.tem $writ_dire/posi_cart_y.tem $writ_dire/posi_cart_z.tem > $writ_dire/posi_cart.tem
	
		N_after_force=$(($syst_numb_atom + 5))
		grep -A $N_after_force "^[* ][* ]*.*Forces" $i | tail -n $syst_numb_atom |awk '{print $4,$5,$6}' > $writ_dire/forc.tem
		paste $writ_dire/symb.tem $writ_dire/posi_cart.tem $writ_dire/forc.tem >> $writ_dire/$writ_file
echo "$i"
#rm -f $writ_dire/*.tem
		echo -n "$N_count/$N_case "
		N_count=$((N_count + 1))
       else
             echo ""
             echo "!!!!!!!!!!!!!!!!!!!!! WARNING WARNING WARNING !!!!!!!!!!!!!!!!!!"
             echo "!!! Something wrong happened casue the number of direct or reciprocal lattice vector components is not 9 in $i !!!"
             echo "!!! So the script is skipping this situation and continue ... !!!"
             echo "!!! It is caused by the ugly formated OUTCAR, user please check it in person.!!!"
             echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
       fi
done
echo "Directory \"$writ_dire\" has been renewed. bye."
