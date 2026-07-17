#!/bin/bash
#SBATCH --job-name=fill_main_k_kernel
##SBATCH -t 20:00:00           # time limit hh:mm:ss
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
##SBATCH --qos=2h              # reduce the queuing time
#SBATCH --nodes=1             # nb of nodes requested
#SBATCH -p cidbn              # partition requested
#SBATCH -o ./slurm-files/%x-%J.out
#SBATCH -e ./slurm-files/%x-%J.err

source ./venv/bin/activate
srun python ./src/main.py
