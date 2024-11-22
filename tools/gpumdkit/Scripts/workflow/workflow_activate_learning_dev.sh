#!/bin/bash -l
#SBATCH -p intel-sc3,intel-sc3-32c
#SBATCH -q huge
#SBATCH -N 1
#SBATCH -J workflow
#SBATCH -o workflow.log
#SBATCH --ntasks-per-node=1
cd $SLURM_SUBMIT_DIR

#---------------------------------  ATTENTION  -----------------------------------------#
# 1. This is a workflow script for Neuroevolution potential (NEP).                      #
# 2. You need to set up some varibles to run it correctly.                              #
# 3. Please contact me if you have any questions. (E-mail: yanzihan@westlake.edu.cn)    #
#---------------------------------------------------------------------------------------#

source ${GPUMDkit_path}/Scripts/workflow/submit_template.sh  # source the submit_template.sh script
python_pynep=/storage/zhuyizhouLab/yanzhihan/soft/conda/envs/gpumd/bin/python  # python executable for pynep

work_dir=${PWD}  # work directory
prefix_name=LiF_iter01  # prefix name for this workflow, used for the scf calculations
min_dist=1.4    # minimum distance between two atoms
box_limit=13    # box limit for the simulation box
max_fp_num=50  # maximum number of single point calculations
sample_method=pynep  # sampling method 'uniform' 'random' 'pynep'
pynep_sample_dist=0.01  # distance for pynep sampling

#print some info
echo "********************************************" 
echo $(date -d "2 second" +"%Y-%m-%d %H:%M:%S")  
echo "WORK_DIR =" ${work_dir} 
echo "********************************************" 

# Check if the required files exist

if [ -f nep.txt ] && [ -f nep.in ] && [ -f train.xyz ] && [ -f run.in ] && [ -f INCAR ] && [ -f POTCAR ] ; then
    if [ $(find . -maxdepth 1 -name "*.xyz" | wc -l) -eq 2 ]; then
        sample_xyz_file=$(ls *.xyz | grep -v "train.xyz")
        sample_struct_num=$(grep -c Lat ${sample_xyz_file})
        echo "Found the exyz file: $sample_xyz_file"
        echo "There are ${sample_struct_num} structs in the ${sample_xyz_file}."
    else
        echo "Error: There should be exactly one exyz file (except for train.xyz) in the current directory."
        exit 1
    fi
    echo "All required files exist."
    echo "Starting the workflow:"
else
    echo "Please put nep.in nep.txt train.xyz run.in INCAR POTCAR [KPOINTS] and the sample_struct.xyz in the current directory."
fi

cd ${work_dir}
mkdir 00.modev common
mv ${work_dir}/{nep.txt,nep.in,*.xyz,run.in,INCAR,KPOINTS,POTCAR} ./common
cp ${work_dir}/common/$sample_xyz_file ${work_dir}/00.modev
cd ${work_dir}/00.modev
(echo 3; echo 302) | gpumdkit.sh >> /dev/null
ln -s ${work_dir}/common/{nep.txt,run.in} ${work_dir}/00.modev/md
echo $(date -d "2 second" +"%Y-%m-%d %H:%M:%S") "Starting 00.modev step ..." 
submit_gpumd_array modev ${sample_struct_num}
sbatch submit.slurm
echo $(date -d "2 second" +"%Y-%m-%d %H:%M:%S") "${sample_struct_num} tasks had been submitted."

# Wait for all tasks to finish
while true; do
    logs=$(find "${work_dir}/00.modev/" -type f -name log -path "*/sample_*/log")
    finished_tasks_md=$(grep "Finished running GPUMD." $logs | wc -l)
    error_tasks_md=$(grep "Error" $logs | wc -l)

    if [ "$error_tasks_md" -ne 0 ]; then
        echo $(date -d "2 second" +"%Y-%m-%d %H:%M:%S") "Error: MD simulation encountered an error. Exiting..." 
        grep "Error" sample_*/log 
        exit 1
    fi
    if [ $finished_tasks_md -eq $sample_struct_num ]; then
        break
    fi
    sleep 30
done

echo $(date -d "2 second" +"%Y-%m-%d %H:%M:%S") "All modev tasks have finished. Starting analysis ..." 

mkdir ${work_dir}/01.select
ln -s ${work_dir}/common/{train.xyz,nep.txt} ${work_dir}/01.select
cat sample_*/dump.xyz >> ${work_dir}/01.select/modev_sampled_structs.xyz

echo $(date -d "2 second" +"%Y-%m-%d %H:%M:%S") "Analysis the min_dist in modev_sampled_structs.xyz" 
actual_min_dist=$(python ${GPUMDkit_path}/Scripts/analyzer/get_min_dist.py ${work_dir}/01.select/modev_sampled_structs.xyz | awk '{print $4}')

if [ $(awk 'BEGIN {print ('$actual_min_dist' < '$min_dist')}') -eq 1 ]; then
    echo "The actual minimum distance ($actual_min_dist) between two atoms is less than the specified value ($min_dist)."
    echo "Filtering the structs based on the min_dist you specified."
    cd ${work_dir}/01.select
    python ${GPUMDkit_path}/Scripts/analyzer/filter_structures_by_distance.py modev_sampled_structs.xyz ${min_dist}
    echo $(date -d "2 second" +"%Y-%m-%d %H:%M:%S") "Analysis the box in modev_sampled_structs.xyz" 
    mv filtered_modev_sampled_structs.xyz modev_sampled_structs.xyz
    python ${GPUMDkit_path}/Scripts/analyzer/filter_exyz_by_box.py modev_sampled_structs.xyz ${box_limit}
    echo "The box limit is $box_limit. filtered structs are saved in filtered_by_box.xyz"
else
    echo "The actual minimum distance ($actual_min_dist) between two atoms is greater than the specified value ($min_dist)."
    echo $(date -d "2 second" +"%Y-%m-%d %H:%M:%S") "Analysis the box in filtered_modev_sampled_structs.xyz" 
    cd ${work_dir}/01.select
    python ${GPUMDkit_path}/Scripts/analyzer/filter_exyz_by_box.py modev_sampled_structs.xyz ${box_limit}
    echo "The box limit is $box_limit. filtered structs are saved in filtered_by_box.xyz"
fi

# Check the value of sample_method
case $sample_method in
    "uniform")
        echo $(date -d "2 second" +"%Y-%m-%d %H:%M:%S") "Performing uniform sampling..." 
        (echo 2; echo 201; echo "filtered_by_box.xyz ${sample_method} ${max_fp_num}") | gpumdkit.sh >> /dev/null
        mv sampled_structures.xyz selected.xyz
        selected_struct_num=$(grep -c Lat selected.xyz)
        ;;
    "random")
        echo $(date -d "2 second" +"%Y-%m-%d %H:%M:%S") "Performing random sampling..." 
        (echo 2; echo 201; echo "filtered_by_box.xyz ${sample_method} ${max_fp_num}") | gpumdkit.sh >> /dev/null
        mv sampled_structures.xyz selected.xyz
        selected_struct_num=$(grep -c Lat selected.xyz)
        ;;
    "pynep")
        echo $(date -d "2 second" +"%Y-%m-%d %H:%M:%S") "Performing pynep sampling..." 
        ${python_pynep} ${GPUMDkit_path}/Scripts/sample_structures/pynep_select_structs.py filtered_by_box.xyz train.xyz nep.txt ${pynep_sample_dist}
        # Check the number of structures in selected.xyz
        selected_struct_num=$(grep -c Lat selected.xyz)
        if [ $selected_struct_num -gt $max_fp_num ]; then
            (echo 2; echo 201; echo "selected.xyz uniform ${max_fp_num}") | gpumdkit.sh >> /dev/null
            mv sampled_structures.xyz selected.xyz
        fi
        selected_struct_num=$(grep -c Lat selected.xyz)
        ;;
    *)
        echo "Invalid sample_method value. Please choose 'uniform', 'random', or 'pynep'." 
        exit 1
        ;;
esac

echo $(date -d "2 second" +"%Y-%m-%d %H:%M:%S") "The number of selected structures is $selected_struct_num." 

# Continue 02.scf step
echo $(date -d "2 second" +"%Y-%m-%d %H:%M:%S") "01.select have finished. Start 02.scf:" 
mkdir ${work_dir}/02.scf
cd ${work_dir}/02.scf
mv ${work_dir}/01.select/selected.xyz .
(echo 3; echo 301; echo "${prefix_name}") | gpumdkit.sh >> /dev/null
ln -s ${work_dir}/common/{INCAR,POTCAR,KPOINTS} ./fp
submit_vasp_array scf ${selected_struct_num} ${prefix_name}
sbatch submit.slurm
echo $(date -d "2 second" +"%Y-%m-%d %H:%M:%S") "${selected_struct_num} tasks had been submitted."

# Wait for all tasks to finish
while true; do
    logs=$(find "${work_dir}/02.scf/" -type f -name log -path "*/${prefix_name}_*/log")
    finished_tasks_scf=$(grep "F=" $logs | wc -l)
    if [ $finished_tasks_scf -eq $selected_struct_num ]; then
        break
    fi
    sleep 30
done

echo $(date -d "2 second" +"%Y-%m-%d %H:%M:%S") "All scf tasks have finished. Starting prediction ..." 
echo "---------------------------------------"
gpumdkit.sh -out2xyz .
echo "---------------------------------------"
cd NEPdataset-multiple_frames
mkdir prediction
cd prediction
ln -s ${work_dir}/common/nep.txt .
ln -s ../NEP-dataset.xyz ./train.xyz
cp ${work_dir}/common/nep.in .

if ! grep -q "prediction" nep.in; then
    echo "prediction 1" >> nep.in
fi

submit_nep_prediction
sbatch submit.slurm
gpumdkit.sh -plt prediction save
echo $(date -d "2 second" +"%Y-%m-%d %H:%M:%S") "Prediction finished."

