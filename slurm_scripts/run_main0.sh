#!/bin/bash
# custom name
#SBATCH --job-name=fill_stimulus
# partition
#SBATCH -p cidbn
# nb of nodes
#SBATCH --nodes=1
# time limit hh:mm:ss
#SBATCH -t 00:20:00
# reduce the queuing time if time is less than 2 hours
#SBATCH --qos=2h

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=38
# ram per cpu
#SBATCH --mem-per-cpu=4G

# redirection of standard error and output
#SBATCH -o ./../slurm-files/%x-%J.out
#SBATCH -e ./../slurm-files/%x-%J.err

# limit the nb of OpenMP threads tp 1 to avoid multiple levels
# of parallelism (e.g. numpy uses OpenMP)
export OMP_NUM_THREADS=1

source ./../venv/bin/activate

srun  ./../venv/bin/python ./../src/main0.py

# to check ram and cpu usage of the job, you may want to run:
# module load py-reportseff
# reportseff -u u24849
