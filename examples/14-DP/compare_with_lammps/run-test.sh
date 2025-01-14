#!/bin/bash
set -e
set -u

# ________________ modify these variables. ________________
gmd_exe="/root/autodl-tmp/GPUMD-master/src/gpumd"
lmp_exe="/root/autodl-tmp/deepmd-kit/bin/lmp"
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# for gpumd run in 1 cpu core.
export TF_INTRA_OP_PARALLELISM_THREADS=1
export OMP_NUM_THREADS=1
export TF_INTER_OP_PARALLELISM_THREADS=1
# DP librays path.
# export LD_LIBRARY_PATH=/home/ke/deepmd-kit/lib:$LD_LIBRARY_PATH

for i in 0 1 2 3 ; do

    gmd_dir="gmd-wat-${i}"
    if [ ! -d ${gmd_dir} ] ; then mkdir ${gmd_dir}
    else rm -rf ${gmd_dir} ; mkdir ${gmd_dir} ; fi
    lmp_dir="lmp-wat-${i}"
    if [ ! -d ${lmp_dir} ] ; then mkdir ${lmp_dir}
    else rm -rf ${lmp_dir} ; mkdir ${lmp_dir} ; fi

    cd ${gmd_dir}
    echo "Run in ${gmd_dir}, model ${i}.xyz"
    gmd_xyz="../Models/${i}.xyz"
    run_in="../Models/run.in"
    echo -e "dp 2 O H \n" > dp.txt
    cp ${gmd_xyz} model.xyz
    cp ${run_in} run.in
    if [ -f dump.xyz ] ; then rm dump.xyz ; fi
    ${gmd_exe} > gmd-${i}.out 2>info
    cd ..

    cd ${lmp_dir}
    source /root/autodl-tmp/deepmd-kit/bin/activate /root/autodl-tmp/deepmd-kit
    echo "Run in ${lmp_dir}, model ${i}-h2o"
    lmp_data="../Models/${i}.data"
    lmp_in="../Models/lmp.in"
    cp ${lmp_data} water.data
    cp ${lmp_in} lmp.in
    ${lmp_exe} -in lmp.in > lmp-${i}.out 2>info
    cd ..
    python Models/compare_force.py ${i}

    echo -e "GPUMD with gpumd-gpu \c" ; grep Speed ${gmd_dir}/gmd-${i}.out | tail -n 1
    echo -e "LAMMPS with dp-cpu \c" ; grep katom-step/s ${lmp_dir}/lmp-${i}.out | tail -n 1

done
