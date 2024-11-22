#!bin/bash

function submit_gpumd_array(){
cat > submit.slurm <<-EOF
#!/bin/bash -l
#SBATCH -p rt-2080ti-short,v100,v100-af,a40-tmp,a40-quad,a100-40g
#SBATCH -q gpu-huge
#SBATCH -N 1
#SBATCH -J $1
#SBATCH -o log
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=1-$2

module purge
module load cuda/10.2

cd \$SLURM_SUBMIT_DIR

cd sample_\${SLURM_ARRAY_TASK_ID}
gpumd > log
EOF
}

function submit_nep_prediction(){
cat > submit.slurm <<-EOF
#!/bin/bash -l
#SBATCH -p rt-2080ti-short,v100,v100-af,a40-tmp,a40-quad,a100-40g
#SBATCH -q gpu-huge
#SBATCH -N 1
#SBATCH -J prediction
#SBATCH -o log
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:1

module purge
module load cuda/10.2

cd \$SLURM_SUBMIT_DIR

nep
EOF
}

function submit_vasp_array(){
cat > submit.slurm <<-EOF
#!/bin/bash -l
#SBATCH -p intel-sc3,intel-sc3-32c
#SBATCH -q huge
#SBATCH -N 1
#SBATCH -J $1
#SBATCH -o log
#SBATCH --ntasks-per-node=32
#SBATCH --array=1-$2
cd \$SLURM_SUBMIT_DIR

module purge
#module load vasp/6.3.1-intel-sc3
module load vasp/5.4.4-intel-sc3

date
cd ${3}_\${SLURM_ARRAY_TASK_ID}
mpirun -n 32 vasp_std > log
cd ..
date
EOF
}