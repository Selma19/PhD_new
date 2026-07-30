#!/bin/bash
export OMP_NUM_THREADS=1
source venv/bin/activate
venv/bin/python src/main2_local.py

# in local, takes 26 minutes to process 1 row from the Main table of the Kernel db,
# when curve_fit is removed from the parametrized kernels
