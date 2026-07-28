#!/bin/bash
# custom name
#SBATCH --job-name=fill_kernels_kernel
# partition
#SBATCH -p cidbn
# nb of nodes
#SBATCH --nodes=6
# time limit hh:mm:ss
#SBATCH -t 01:40:00
# reduce the queuing time if time is less than 2 hours
#SBATCH --qos=2h

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50

# ram per cpu:
# if the ram requested is not large enough when multiple nodes
# are used, you may not get an OOM error, just some of the processes will wait,
# doing nothing until the time limit is reached
#SBATCH --mem-per-cpu=4G

# redirection of standard error and output
#SBATCH -o ../slurm-files/%x-%J.out
#SBATCH -e ../slurm-files/%x-%J.err

# limit the nb of OpenMP threads tp 1 to avoid multiple levels
# of parallelism (e.g. numpy uses OpenMP)
export OMP_NUM_THREADS=1

source ../venv/bin/activate

# this will be the database where all data will be aggregated after computation
cp ../data/solo/kernel.db $SHARED_TMPDIR

echo "kernel transferred to shared location"

n_scripts=$SLURM_JOB_NUM_NODES

for ((i=0; i<$n_scripts; i++))
do
    srun --nodes=1 --exclusive bash _run_main2.sh $i $n_scripts &
done

wait

echo "aggregating all pieces into a single database..."

srun --nodes=1 --exclusive ../venv/bin/python ../src/merge_kernel_dbs.py

echo "transferring back the kernel database..."

cp ${SHARED_TMPDIR}/kernel.db ../data/solo/new_kernel.db
