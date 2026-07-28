"""Fills the kernel table of the kernel database.

Notes
-----
There are only 568 rows in the Main table of the Kernel database, so
if parallelization is done at the level of the rows, we do not
take advantage of the SCC resources.
So instead, we parallelize at the level of the triples (17 triples per row,
so 17 * 568 tasks that can be handled independently).
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
        max_rows_per_cpu=2,
        n_cpus_max=int(os.environ["SLURM_CPUS_PER_TASK"]),
        script_num=script_num,
        n_scripts=n_scripts,
        debug=False
    )
