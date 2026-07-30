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

if __name__ == "__main__":
    # collect arguments sent to the program
    db = Kernel_db()
    db._fill_kernels(
        max_rows_per_cpu=1,
        n_cpus_max=10
    )
