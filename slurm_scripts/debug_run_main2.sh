#!/bin/bash
# custom name
#SBATCH --job-name=debug
#SBATCH -p scc-cpu
#SBATCH --nodes=3
#SBATCH -t 00:03:00
#SBATCH --qos=2h

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G

#SBATCH -o ../slurm-files/%x-%J.out
#SBATCH -e ../slurm-files/%x-%J.err

export OMP_NUM_THREADS=1

source ../venv/bin/activate

#n_scripts=$(( $SLURM_JOB_NUM_NODES - 1 ))
n_scripts=$SLURM_JOB_NUM_NODES

for ((i=0; i<$n_scripts; i++))
do
    srun --nodes=1 --exclusive bash _debug.sh $i $n_scripts &
done

wait
