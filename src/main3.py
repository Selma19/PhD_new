"""Completes filling the kernel table of the kernel database,
because `main2.py` failed due to a time limit.
"""
from database import Kernel_db
import sys, os

if __name__ == "__main__":
    # collect arguments sent to the program
    args = sys.argv[1:3]
    script_num = int(args[0])
    n_scripts = int(args[1])

    db = Kernel_db(location='memory')
    db._fill_kernels(
        max_rows_per_cpu=1,
        n_cpus_max=int(os.environ["SLURM_CPUS_PER_TASK"]),
        script_num=script_num,
        n_scripts=n_scripts,
        debug=False
    )
