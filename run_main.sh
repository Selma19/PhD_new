#!/bin/bash
# custom name
#SBATCH --job-name=fill_kernels
# partition
#SBATCH -p cidbn
# nb of nodes (note that multiprocessing does not allow for multiple nodes to be used)
#SBATCH --nodes=1
# time limit hh:mm:ss
#SBATCH -t 10:00:00
# reduce the queuing time if time is less than 2 hours
##SBATCH --qos=2h

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=100
# ram per cpu
#SBATCH --mem-per-cpu=1G

# redirection of standard error and output
#SBATCH -o ./slurm-files/%x-%J.out
#SBATCH -e ./slurm-files/%x-%J.err

# limit the nb of OpenMP threads tp 1 to avoid multiple levels
# of parallelism (e.g. numpy uses OpenMP)
export OMP_NUM_THREADS=1

source ./venv/bin/activate
srun ./venv/bin/python ./src/main.py
