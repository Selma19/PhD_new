"""Merges the tables Kernels from various databases,
given that these tables are all disjoint and have an autoincrementing primary key.
"""

import sqlite3, os

db_dir = os.environ['SHARED_TMPDIR']

# connect to the aggregation database
main_conn = sqlite3.connect(
    os.path.join(db_dir, 'kernel.db')
)
main_conn.execute("PRAGMA foreign_keys = ON;")

db_names = [name for name in os.listdir(db_dir) if 'kernel_' in name and '.db' in name]
for db_name in db_names:
    conn = sqlite3.connect(
        os.path.join(db_dir, db_name)
    )
    conn.execute("PRAGMA foreign_keys = ON;")

    # select all columns but the primary key
    rows = conn.execute("""
        SELECT
            main_key, kernel_output,
            train_error, test_error,
            kernel_type, kernel_method,
            method_param
        FROM Kernels
    """).fetchall()
    conn.close()

    main_conn.executemany(
        """
        INSERT INTO Kernels (
            main_key, kernel_output, train_error, test_error,
            kernel_type, kernel_method, method_param
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, rows
    )
    main_conn.commit()
main_conn.close()
