#!/bin/bash
#SBATCH --job-name=fill_kernels
##SBATCH -t 00:30:00          # time limit hh:mm:ss
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
##SBATCH --qos=2h             # reduce the queuing time
#SBATCH --nodes=1             # nb of nodes requested
#SBATCH --mem-per-cpu=5G      # ram requested per cpu
#SBATCH -p cidbn              # partition requested
#SBATCH -o ./slurm-files/%x-%J.out
#SBATCH -e ./slurm-files/%x-%J.err

source ./venv/bin/activate
srun ./venv/bin/python ./src/main.py
