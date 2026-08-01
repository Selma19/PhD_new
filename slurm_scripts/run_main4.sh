#!/bin/bash

###########################################################################################

# After executing run_main4.sh, we got 6 chunks which we merge here to get the filled kernel table.

###########################################################################################

# custom name
#SBATCH --job-name=fill_kernels_kernel_merge_chunks
# partition
#SBATCH -p cidbn
# nb of nodes
#SBATCH --nodes=1
# time limit hh:mm:ss
#SBATCH -t 01:00:00
# reduce the queuing time if time is less than 2 hours
#SBATCH --qos=2h

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# ram per cpu:
# if the ram requested is not large enough when multiple nodes
# are used, you may not get an OOM error, just some of the processes will wait,
# doing nothing until the time limit is reached
#SBATCH --mem-per-cpu=5G

# redirection of standard error and output
#SBATCH -o ../slurm-files/%x-%J.out
#SBATCH -e ../slurm-files/%x-%J.err

# this is the database where data from the different nodes are aggregated after computation
cp ../data/solo/kernel.db $SHARED_TMPDIR
for ((i=0; i<6; i++))
do
    cp ../data/solo/kernel_$i.db $SHARED_TMPDIR
done

echo "chunks transferred"

source ../venv/bin/activate
srun ../venv/bin/python ../src/merge_kernel_dbs.py

echo "chunks merged"

cp ${SHARED_TMPDIR}/kernel.db ../data/solo/filled_kernel.db
