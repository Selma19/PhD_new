#!/bin/bash

###########################################################################################

# The first chunk, managed by the script 0 (so 100 rows) as spawn by run_main2.sh, did not complete.
# Hopefully, we had transferred the completed chunks to data/solo before the job expired.
# So here we divide the 100 remaining rows among 10 cpu cores per node and 10 nodes.


# what we observe:
# is that the total time needed to process 100 rows divided as 1 row per cpu core
# takes 125 min, whereas the total time to process the exact same
# 100 rows divided as 2 rows per cpu core takes MORE than 1200 min.
# It clearly does not add up. The only differences btw the two settings:
# - 10 cores per node in the 1 row setting vs 50 cores per node in the 2 rows setting
# - 5 G of ram in the 1 row setting vs 3 G in the 2 rows setting

###########################################################################################

# custom name
#SBATCH --job-name=fill_kernels_kernel_last_chunk
# partition
#SBATCH -p cidbn
# nb of nodes
#SBATCH --nodes=10
# time limit hh:mm:ss
#SBATCH -t 20:00:00
# reduce the queuing time if time is less than 2 hours
##SBATCH --qos=2h

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10

# ram per cpu:
# if the ram requested is not large enough when multiple nodes
# are used, you may not get an OOM error, just some of the processes will wait,
# doing nothing until the time limit is reached
#SBATCH --mem-per-cpu=5G

# redirection of standard error and output
#SBATCH -o ../slurm-files/%x-%J.out
#SBATCH -e ../slurm-files/%x-%J.err

# limit the nb of OpenMP threads to 1 to avoid multiple levels
# of parallelism (e.g. numpy uses OpenMP)
export OMP_NUM_THREADS=1

source ../venv/bin/activate

# this is the database where data from the different nodes are aggregated after computation
cp ../data/solo/kernel.db $SHARED_TMPDIR

echo "kernel transferred to shared location"

n_scripts=$SLURM_JOB_NUM_NODES

for ((i=0; i<$n_scripts; i++))
do
    srun --nodes=1 --exclusive bash _run_main3.sh $i $n_scripts &
done

wait

echo "aggregating all pieces into a single database..."

srun --nodes=1 --exclusive ../venv/bin/python ../src/merge_kernel_dbs.py

echo "transferring back the kernel database..."

cp ${SHARED_TMPDIR}/kernel.db ../data/solo/kernel_with_last_chunk.db
