#!/bin/bash

# You need to set the path of GPUMD and GPUMDkit in your ~/.bashrc, for example
# export GPUMD_path=/d/Westlake/GPUMD
# export GPUMDkit_path=${GPUMD_path}/tools/gpumdkit

VERSION="0.0.1 (dev) (2024-11-22)"

function f1_format_conversion(){
echo " ------------>>"
echo " 101) Convert OUTCAR to extxyz"
echo " 102) Convert mtp to extxyz"
echo " 103) Convert cp2k to extxyz"
echo " 104) Convert castep to extxyz"
echo " 105) Convert extxyz to POSCAR"
echo " 106) Developing ... "
echo " 000) Return to the main menu"
echo " ------------>>"
echo " Input the function number:"

arry_num_choice=("000" "101" "102" "103" "104" ) 
read -p " " num_choice
while ! echo "${arry_num_choice[@]}" | grep -wq "$num_choice" 
do
  echo " ------------>>"
  echo " Please reinput function number..."
  read -p " " num_choice
done

case $num_choice in
    "101")
        echo " >-------------------------------------------------<"
        echo " | This function calls the script in GPUMD's tools |"
        echo " | Script: multipleFrames-outcars2nep-exyz.sh      |"
        echo " | Developer: Yanzhou WANG (yanzhowang@gmail.com ) |"
        echo " >-------------------------------------------------<"
        echo " Input the directory containing OUTCARs"
        echo " ------------>>"
        read -p " " dir_outcars
        echo " >-------------------------------------------------<"
        bash ${GPUMD_path}/tools/vasp2xyz/outcar2xyz/multipleFrames-outcars2nep-exyz.sh ${dir_outcars}
        echo " >-------------------------------------------------<"
        echo " Code path: ${GPUMD_path}/tools/vasp2xyz/outcar2xyz/multipleFrames-outcars2nep-exyz.sh"
        ;;
    "102")
        echo " >-------------------------------------------------<"
        echo " | This function calls the script in GPUMD's tools |"
        echo " | Script: mtp2xyz.py                              |"
        echo " | Developer: Ke XU (kickhsu@gmail.com)            |"
        echo " >-------------------------------------------------<"
        echo " Input <filename.cfg> <Symbol1 Symbol2 Symbol3 ...>"
        echo " Examp: train.cfg Pd Ag"
        echo " ------------>>"
        read -p " " mtp_variables
        echo " ---------------------------------------------------"
        python ${GPUMD_path}/tools/mtp2xyz/mtp2xyz.py ${mtp_variables}
        echo " Code path: ${GPUMD_path}/tools/mtp2xyz/mtp2xyz.py"
        echo " ---------------------------------------------------"
        ;;
    "103")
        echo " >-------------------------------------------------<"
        echo " | This function calls the script in GPUMD's tools |"
        echo " | Script: cp2k2xyz.py                             |"
        echo " | Developer: Ke XU (kickhsu@gmail.com)            |"
        echo " >-------------------------------------------------<"
        echo " Input <dir_cp2k> "
        echo " Examp: ./cp2k "
        echo " ------------>>"
        read -p " " dir_cp2k
        echo " ---------------------------------------------------"
        python ${GPUMD_path}/tools/cp2k2xyz/cp2k2xyz.py ${dir_cp2k}
        echo " Code path: ${GPUMD_path}/tools/cp2k2xyz/cp2k2xyz.py"
        echo " ---------------------------------------------------"
        ;;
    "104")
        echo " >-------------------------------------------------<"
        echo " | This function calls the script in GPUMD's tools |"
        echo " | Script: castep2nep-exyz.sh                      |"
        echo " | Developer: Yanzhou WANG (yanzhowang@gmail.com ) |"
        echo " >-------------------------------------------------<"
        echo " Input <dir_castep>"
        echo " Examp: ./castep "
        echo " ------------>>"
        read -p " " dir_castep
        echo " ---------------------------------------------------"
        bash ${GPUMD_path}/tools/castep2exyz/castep2nep-exyz.sh ${dir_castep}
        echo " Code path: ${GPUMD_path}/tools/castep2exyz/castep2nep-exyz.sh"
        echo " ---------------------------------------------------"
        ;;
    "105")
        echo " >-------------------------------------------------<"
        echo " | This function calls the script in Scripts       |"
        echo " | Script: exyz2pos.py                             |"
        echo " | Developer: Zihan YAN (yanzihan@westlake.edu.cn) |"
        echo " >-------------------------------------------------<"
        echo " Input the name of extxyz"
        echo " Examp: ./train.xyz "
        echo " ------------>>"
        read -p " " filename
        echo " ---------------------------------------------------"
        python ${GPUMDkit_path}/Scripts/format_conversion/exyz2pos.py ${filename}
        echo " Code path: ${GPUMDkit_path}/Scripts/format_conversion/exyz2pos.py"
        echo " ---------------------------------------------------"
        ;; 
    "106")
        echo " Developing ... "
        ;;
       
    "000")
        menu
        main
        ;;
esac

}

function f2_sample_structures(){
echo " ------------>>"
echo " 201) Sample structures from extxyz"
echo " 202) Sample structures by pynep"
echo " 203) Find the outliers in training set"
echo " 204) Perturb structure"
echo " 205) Developing ... "
echo " 000) Return to the main menu"
echo " ------------>>"
echo " Input the function number:"

arry_num_choice=("000" "201" "202" "203" "204" "205" "206") 
read -p " " num_choice
while ! echo "${arry_num_choice[@]}" | grep -wq "$num_choice" 
do
  echo " ------------>>"
  echo " Please reinput function number..."
  read -p " " num_choice
done

case $num_choice in
    "201")
        echo " >-------------------------------------------------<"
        echo " | This function calls the script in Scripts       |"
        echo " | Script: sample_structures.py                    |"
        echo " | Developer: Zihan YAN (yanzihan@westlake.edu.cn) |"
        echo " >-------------------------------------------------<"
        echo " Input <extxyz_file> <sampling_method> <num_samples> [skip_num]"
        echo " [skip_num]: number of initial frames to skip, default value is 0"
        echo " Sampling_method: 'uniform' or 'random'"
        echo " Examp: train.xyz uniform 50 "
        echo " ------------>>"
        read -p " " sample_choice
        echo " ---------------------------------------------------"
        python ${GPUMDkit_path}/Scripts/sample_structures/sample_structures.py ${sample_choice}
        echo " Code path: ${GPUMDkit_path}/Scripts/sample_structures/sample_structures.py"
        echo " ---------------------------------------------------"
        ;;
    "202")
        echo " >-------------------------------------------------<"
        echo " | This function calls the script in Scripts       |"
        echo " | Script: pynep_select_structs.py                 |"
        echo " | Developer: Zihan YAN (yanzihan@westlake.edu.cn) |"
        echo " >-------------------------------------------------<"
        echo " Input <sample.xyz> <train.xyz> <nep_model> <min_dist>"
        echo " Examp: dump.xyz train.xyz ./nep.txt 0.01 "
        echo " ------------>>"
        read -p " " sample_choice
        echo " ---------------------------------------------------"
        python ${GPUMDkit_path}/Scripts/sample_structures/pynep_select_structs.py ${sample_choice}
        echo " Code path: ${GPUMDkit_path}/Scripts/sample_structures/pynep_select_structs.py"
        echo " ---------------------------------------------------"
        ;;
    "203")
        echo " >-------------------------------------------------<"
        echo " | This function calls the script in GPUMD's tools |"
        echo " | Script: get_max_rmse_xyz.py                     |"
        echo " | Developer: Ke XU (kickhsu@gmail.com)            |"
        echo " >-------------------------------------------------<"
        echo " Input <extxyz_file> <*_train.out> <num_outliers>"
        echo " Examp: train.xyz energy_train.out 13 "
        echo " ------------>>"
        read -p " " maxrmse_choice
        echo " ---------------------------------------------------"
        python ${GPUMD_path}/tools/get_max_rmse_xyz/get_max_rmse_xyz.py ${maxrmse_choice}
        echo " Code path: ${GPUMD_path}/tools/get_max_rmse_xyz/get_max_rmse_xyz.py"
        echo " ---------------------------------------------------"
        ;;
    "204")
        echo " >-------------------------------------------------<"
        echo " | This function calls the script in Scripts       |"
        echo " | Script: perturb_structure.py                    |"
        echo " | Developer: Zihan YAN (yanzihan@westlake.edu.cn) |"
        echo " >-------------------------------------------------<"
        echo " Input <input.vasp> <pert_num> <cell_pert_fraction> <atom_pert_distance> <atom_pert_style>"
        echo " The default paramters for perturb are 20 0.03 0.2 uniform"
        echo " Examp: POSCAR 20 0.03 0.2 uniform"
        echo " ------------>>"
        read -p " " perturb_choice
        echo " ---------------------------------------------------"
        python ${GPUMDkit_path}/Scripts/sample_structures/perturb_structure.py ${perturb_choice}
        echo " Code path: ${GPUMDkit_path}/Scripts/sample_structures/perturb_structure.py"
        echo " ---------------------------------------------------"
        ;;
    "205")
        echo " Developing ... "
        ;;
    "000")
        menu
        main
        ;;
esac

}

function f3_workflow_dev(){
echo " ------------>>"
echo " 301) SCF batch pretreatment"
echo " 302) MD sample batch pretreatment (gpumd)"
echo " 303) MD sample batch pretreatment (lmp)"
echo " 304) Developing ... "
echo " 000) Return to the main menu"
echo " ------------>>"
echo " Input the function number:"

arry_num_choice=("000" "301" "302" "303" "304") 
read -p " " num_choice
while ! echo "${arry_num_choice[@]}" | grep -wq "$num_choice" 
do
  echo " ------------>>"
  echo " Please reinput function number..."
  read -p " " num_choice
done

case $num_choice in
    "301")
        source ${GPUMDkit_path}/Scripts/workflow/scf_batch_pretreatment.sh
        f301_scf_batch_pretreatment
        ;;
    "302")
        source ${GPUMDkit_path}/Scripts/workflow/md_sample_batch_pretreatment_gpumd.sh
        f302_md_sample_batch_pretreatment_gpumd
        ;;
    "303")
        source ${GPUMDkit_path}/Scripts/workflow/md_sample_batch_pretreatment_lmp.sh
        f303_md_sample_batch_pretreatment_lmp
        ;; 
    "304")
        echo " Developing ... "
        ;;         
    "000")
        menu
        main
        ;;
esac
}

#--------------------- main script ----------------------
# Show the menu
function menu(){
echo " ----------------------- GPUMD -----------------------"
echo " 1) Format Conversion          2) Sample Structures   "
echo " 3) Workflow (dev)             4) Developing ...      "
echo " 0) Quit!"
}

# Function main
function main(){
    echo " ------------>>"
    echo ' Input the function number:'
    array_choice=("0" "1" "2" "3" "4") 
    read -p " " choice
    while ! echo "${array_choice[@]}" | grep -wq "$choice" 
    do
      echo " ------------>>"
      echo " Please reinput function number:"
      read -p " " choice
    done

    case $choice in
        "0")
            echo " Thank you for using GPUMDkit. Have a great day!"
            exit 0
            ;;
        "1")
            f1_format_conversion
            ;;
        "2")
            f2_sample_structures
            ;;
        "3")
            f3_workflow_dev
            ;;
        "4")
            echo "Developing ..."
            ;;
        *)
            echo "Incorrect Options"
            ;;

    esac
    echo " Thank you for using GPUMDkit. Have a great day!"
}

######### Custom functional area ###############
function help_info_table(){
    echo "+==================================================================================================+"
    echo "|                              GPUMDkit ${VERSION} Usage                             |"
    echo "|                                                                                                  |"
    echo "+======================================== Conversions =============================================+"
    echo "| -outcar2exyz   Convert OUTCAR to extxyz       | -pos2exyz     Convert POSCAR to extxyz           |"
    echo "| -castep2exyz   Convert castep to extxyz       | -pos2lmp      Convert POSCAR to LAMMPS           |"
    echo "| -cp2k2exyz     Convert cp2k output to extxyz  | -lmp2exyz     Convert LAMMPS-dump to extxyz      |"
    echo "| -addgroup      Add group label                | -addweight    Add weight to the struct in extxyz |"
    echo "| Developing...                                 | Developing...                                    |"
    echo "+========================================= Analysis ===============================================+"
    echo "| -range         Print range of energy etc.     | -max_rmse     Get max RMSE from XYZ              |"
    echo "| -min_dist      Get min_dist between atoms     | -filter_dist  Filter struct by min_dist          |"
    echo "| -filter_box    Filter struct by box limits    | -filter_value Filter struct by value (efs)       |"
    echo "+=========================================    Misc  ==============+================================+"
    echo "| -plt           Plot scripts                   | -get_frame     Extract the specified frame       |"
    echo "| -h, -help      Show this help message         | Developing...                                    |"
    echo "+==================================================================================================+"
    echo "| For detailed usage and examples, use: gpumdkit.sh -<option> -h                                   |"
    echo "+==================================================================================================+"
}


if [ ! -z "$1" ]; then
    case $1 in
        -h|-help)
            help_info_table
            ;;

        -plt)
            if [ ! -z "$2" ] && [ "$2" != "-h" ]; then
                case $2 in
                    "thermo")
                        python ${GPUMDkit_path}/Scripts/plt_scripts/plt_nep_thermo.py $3
                        ;;
                    "train")
                        python ${GPUMDkit_path}/Scripts/plt_scripts/plt_nep_train_results.py $3
                        ;;  
                    "prediction"| "valid"| "test")
                        python ${GPUMDkit_path}/Scripts/plt_scripts/plt_nep_prediction_results.py $3
                        ;;
                    "msd")
                        python ${GPUMDkit_path}/Scripts/plt_scripts/plt_msd.py $3
                        ;;
                    "vac")
                        python ${GPUMDkit_path}/Scripts/plt_scripts/plt_vac.py $3
                        ;;                
                    *)
                        echo "Usage: -plt thermo/train/prediction/msd/vac [save]"
                        echo "Examp: gpumdkit.sh -plt thermo save"
                        exit 1
                        ;;
                esac
            else
                echo " Usage: -plt thermo/train/prediction/msd/vac [save] (eg. gpumdkit.sh -plt thermo)"
                echo " See the codes in plt_scripts for more details"
                echo " Code path: ${GPUMDkit_path}/Scripts/plt_scripts"
            fi
            ;;

        -range)
            if [ ! -z "$2" ] && [ ! -z "$3" ] && [ "$2" != "-h" ]  ; then
                echo " Calling script by Zihan YAN. "
                echo " Code path: ${GPUMDkit_path}/Scripts/analyzer/energy_force_virial_analyzer.py"
                python ${GPUMDkit_path}/Scripts/analyzer/energy_force_virial_analyzer.py $2 $3 ${@:4}
            else
                echo " Usage: -range <exyzfile> <property> [hist] (eg. gpumdkit.sh -range train.xyz energy hist)" 
                echo " See the source code of energy_force_virial_analyzer.py for more details"
                echo " Code path: Code path: ${GPUMDkit_path}/Scripts/analyzer/energy_force_virial_analyzer.py"
            fi
            ;;

        -out2xyz|-outcar2exyz)
            if [ ! -z "$2" ] && [ "$2" != "-h" ] ; then
                echo " Calling script by Yanzhou WANG et al. "
                echo " Code path: ${GPUMD_path}/tools/vasp2xyz/outcar2xyz/multipleFrames-outcars2nep-exyz.sh"
                bash ${GPUMD_path}/tools/vasp2xyz/outcar2xyz/multipleFrames-outcars2nep-exyz.sh $2
            else
                echo " Usage: -out2xyz|-outcar2exyz dir_name (eg. gpumdkit.sh -outcar2exyz .)"
                echo " See the source code of multipleFrames-outcars2nep-exyz.sh for more details"
                echo " Code path: ${GPUMD_path}/tools/vasp2xyz/outcar2xyz/multipleFrames-outcars2nep-exyz.sh"
            fi
            ;;

        -cast2xyz|-castep2exyz)
            if [ ! -z "$2" ] && [ "$2" != "-h" ] ; then
                echo " Calling script by Yanzhou WANG et al. "
                echo " Code path: ${GPUMD_path}/tools/castep2exyz/castep2nep-exyz.sh"
                bash ${GPUMD_path}/tools/castep2exyz/castep2nep-exyz.sh $2
            else
                echo " Usage: -cast2xyz|-castep2exyz dir_name (eg. gpumdkit.sh -castep2exyz .)"
                echo " See the source code of castep2nep-exyz.sh for more details"
                echo " Code path: ${GPUMD_path}/tools/castep2exyz/castep2nep-exyz.sh"
            fi
            ;;

        -cp2k2xyz|-cp2k2exyz)
            if [ ! -z "$2" ] && [ "$2" != "-h" ] ; then
                echo " Calling script by Ke XU et al. "
                echo " Code path: ${GPUMD_path}/tools/cp2k2xyz/cp2k2xyz.py"
                python ${GPUMD_path}/tools/cp2k2xyz/cp2k2xyz.py $2
            else
                echo " Usage: -cp2k2xyz|-cp2k2exyz dir_name (eg. gpumdkit.sh -cp2k2exyz .)"
                echo " See the source code of cp2k2xyz.py for more details"
                echo " Code path: ${GPUMD_path}/tools/cp2k2xyz/cp2k2xyz.py"
            fi
            ;;

        -mtp2xyz|-mtp2exyz)
            if [ ! -z "$2" ] && [ "$2" != "-h" ] ; then
                echo " Calling script by Ke XU et al. "
                echo " Code path: ${GPUMD_path}/tools/mtp2xyz/mtp2xyz.py"
                python ${GPUMD_path}/tools/mtp2xyz/mtp2xyz.py train.cfg $2 ${@:3}
            else
                echo " Usage: -mtp2xyz|-mtp2exyz train.cfg Symbol1 Symbol2 Symbol3 ..."
                echo "   Examp: gpumdkit.sh -mtp2exyz train.cfg Pd Ag"
                echo " See the source code of mtp2xyz.py for more details"
                echo " Code path: ${GPUMD_path}/tools/mtp2xyz/mtp2xyz.py"
            fi
            ;;

        -pos2exyz)
            if [ ! -z "$2" ] && [ ! -z "$3" ] && [ "$2" != "-h" ] ; then
                echo " Calling script by Zihan YAN "
                echo " Code path: ${GPUMDkit_path}/Scripts/format_conversion/pos2exyz.py"
                python ${GPUMDkit_path}/Scripts/format_conversion/pos2exyz.py $2 $3
            else
                echo " Usage: -pos2exyz POSCAR model.xyz"
                echo " See the source code of pos2exyz.py for more details"
                echo " Code path: ${GPUMDkit_path}/Scripts/format_conversion/pos2exyz.py"
            fi
            ;;

        -exyz2pos)
            if [ ! -z "$2" ] && [ "$2" != "-h" ] ; then
                echo " Calling script by Zihan YAN "
                echo " Code path: ${GPUMDkit_path}/Scripts/format_conversion/exyz2pos.py"
                python ${GPUMDkit_path}/Scripts/format_conversion/exyz2pos.py $2
            else
                echo " Usage: -exyz2pos model.xyz"
                echo " See the source code of exyz2pos.py for more details"
                echo " Code path: ${GPUMDkit_path}/Scripts/format_conversion/exyz2pos.py"
            fi
            ;;

        -pos2lmp)
            if [ ! -z "$2" ] && [ "$2" != "-h" ] && [ ! -z "$3" ] ; then
                echo " Calling script by Zihan YAN "
                echo " Code path: ${GPUMDkit_path}/Scripts/format_conversion/pos2lmp.py"
                python ${GPUMDkit_path}/Scripts/format_conversion/pos2lmp.py $2 $3
            else
                echo " Usage: -pos2lmp POSCAR lammps.data"
                echo " See the source code of pos2lmp.py for more details"
                echo " Code path: ${GPUMDkit_path}/Scripts/format_conversion/pos2lmp.py"
            fi
            ;;

        -lmp2exyz|-lmpdump2exyz)
            if [ ! -z "$2" ] && [ "$2" != "-h" ] && [ ! -z "$3" ] ; then
                echo " Calling script by Zihan YAN "
                echo " Code path: ${GPUMDkit_path}/Scripts/format_conversion/lmp2exyz.py"
                python ${GPUMDkit_path}/Scripts/format_conversion/lmp2exyz.py $2 ${@:3}
            else
                echo " Usage: -lmp2exyz <dump_file> <element1> <element2> ..."
                echo " See the source code of lmp2exyz.py for more details"
                echo " Code path: ${GPUMDkit_path}/Scripts/format_conversion/lmp2exyz.py"
            fi
            ;;

        -addgroup|-addlabel)
            if [ ! -z "$2" ] && [ "$2" != "-h" ] && [ ! -z "$3" ] ; then
                echo " Calling script by Zihan YAN "
                echo " Code path: ${GPUMDkit_path}/Scripts/format_conversion/add_groups.py"
                python ${GPUMDkit_path}/Scripts/format_conversion/add_groups.py $2 ${@:3}
            else
                echo " Usage: -addgroup <POSCAR> <element1> <element2> ..."
                echo " See the source code of add_groups.py for more details"
                echo " Code path: ${GPUMDkit_path}/Scripts/format_conversion/add_groups.py"
            fi
            ;;

        -addweight)
            if [ ! -z "$2" ] && [ "$2" != "-h" ] && [ ! -z "$3" ] && [ ! -z "$4" ]; then
                echo " Calling script by Zihan YAN "
                echo " Code path: ${GPUMDkit_path}/Scripts/format_conversion/add_weight.py"
                python ${GPUMDkit_path}/Scripts/format_conversion/add_weight.py $2 $3 $4
            else
                echo " Usage: -addweight <input.xyz> <output.xyz> <weight> "
                echo " See the source code of add_groups.py for more details"
                echo " Code path: ${GPUMDkit_path}/Scripts/format_conversion/add_weight.py"
            fi
            ;;

        -max_rmse|-get_max_rmse_xyz)
            if [ ! -z "$2" ] && [ "$2" != "-h" ] && [ ! -z "$3" ] && [ ! -z "$4" ]; then
                echo " Calling script by Ke XU "
                echo " Code path: ${GPUMD_path}/tools/get_max_rmse_xyz/get_max_rmse_xyz.py"
                python ${GPUMD_path}/tools/get_max_rmse_xyz/get_max_rmse_xyz.py $2 $3 $4
            else
                echo " Usage: -getmax|-get_max_rmse_xyz train.xyz force_train.out 13"
                echo " See the source code of get_max_rmse_xyz.py for more details"
                echo " Code path: ${GPUMD_path}/tools/get_max_rmse_xyz/get_max_rmse_xyz.py"
            fi
            ;;

        -min_dist)
            if [ ! -z "$2" ] && [ "$2" != "-h" ]; then
                echo " Calling script by Zihan YAN "
                echo " Code path: ${GPUMDkit_path}/Scripts/analyzer/get_min_dist.py"
                python ${GPUMDkit_path}/Scripts/analyzer/get_min_dist.py $2
            else
                echo " Usage: -min_dist <exyzfile>"
                echo " See the source code of get_min_dist.py for more details"
                echo " Code path: ${GPUMDkit_path}/Scripts/analyzer/get_min_dist.py"
            fi
            ;;

        -filter_dist)
            if [ ! -z "$2" ] && [ "$2" != "-h" ] && [ ! -z "$3" ]; then
                echo " Calling script by Zihan YAN "
                echo " Code path: ${GPUMDkit_path}/Scripts/analyzer/filter_structures_by_distance.py"
                python ${GPUMDkit_path}/Scripts/analyzer/filter_structures_by_distance.py $2 $3
            else
                echo " Usage: -filter_xyz <exyzfile> <min_dist>"
                echo " See the source code of filter_structures_by_distance.py for more details"
                echo " Code path: ${GPUMDkit_path}/Scripts/analyzer/filter_structures_by_distance.py"
            fi
            ;;

        -filter_box)
            if [ ! -z "$2" ] && [ "$2" != "-h" ] && [ ! -z "$3" ] ; then
                echo " Calling script by Zihan YAN "
                echo " Code path: ${GPUMDkit_path}/Scripts/analyzer/filter_exyz_by_box.py"
                python ${GPUMDkit_path}/Scripts/analyzer/filter_exyz_by_box.py $2 $3
            else
                echo " Usage: -filter_box <exyzfile> <lattice limit>"
                echo " See the source code of filter_exyz_by_box.py for more details"
                echo " Code path: ${GPUMDkit_path}/Scripts/analyzer/filter_exyz_by_box.py"
            fi
            ;;

        -filter_value)
            if [ ! -z "$2" ] && [ "$2" != "-h" ] && [ ! -z "$3" ] ; then
                echo " Calling script by Zihan YAN "
                echo " Code path: ${GPUMDkit_path}/Scripts/analyzer/filter_exyz_by_value.py"
                python ${GPUMDkit_path}/Scripts/analyzer/filter_exyz_by_value.py $2 $3 $4
            else
                echo " Usage: -filter_value <exyzfile> <property> <value>"
                echo " See the source code of filter_exyz_by_value.py for more details"
                echo " Code path: ${GPUMDkit_path}/Scripts/analyzer/filter_exyz_by_value.py"
            fi
            ;;

        -get_frame)
            if [ ! -z "$2" ] && [ "$2" != "-h" ] && [ ! -z "$3" ] ; then
                echo " Calling script by Zihan YAN "
                echo " Code path: ${GPUMDkit_path}/Scripts/format_conversion/get_frame.py"
                python ${GPUMDkit_path}/Scripts/format_conversion/get_frame.py $2 $3
            else
                echo " Usage: -get_frame <exyzfile> <frame_index>"
                echo " See the source code of get_frame.py for more details"
                echo " Code path: ${GPUMDkit_path}/Scripts/format_conversion/get_frame.py"
            fi
            ;;

        *)
            echo " Unknown option: $1 "
            help_info_table
            exit 1
            ;;
    esac
    exit
fi

## logo
echo -e "\
           ____ ____  _   _ __  __ ____  _    _ _   
          / ___|  _ \| | | |  \/  |  _ \| | _(_) |_ 
         | |  _| |_) | | | | |\/| | | | | |/ / | __|
         | |_| |  __/| |_| | |  | | |_| |   <| | |_ 
          \____|_|    \___/|_|  |_|____/|_|\_\_|\__|
                                            
          GPUMDkit Version ${VERSION}
        Developer: Zihan YAN (yanzihan@westlake.edu.cn)
      "
menu
main