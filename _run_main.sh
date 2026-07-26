#!/bin/bash

# This script is useful to move data from the project directory to the ram filesystem local to
# the compute node.
# This transfer should take place after the compute node has been decided (so that
# $SHM_TMPDIR is defined) and before the Python program is run.
# Once the python program has completed, these data should be transferred back to
# the project directory.

cp ./data/solo/kernel.db $LOCAL_TMPDIR
cp ./data/solo/stimulus.db $LOCAL_TMPDIR

echo "databases transferred to compute node local ssd"

./venv/bin/python ./src/main.py $1 $2

echo "transferring the database chunk to the ssd shared by all compute nodes"

cp ${LOCAL_TMPDIR}/kernel.db $SHARED_SSD_TMPDIR/kernel_$1.db
