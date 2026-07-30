"""Defines the kernel database."""

import subprocess, os, json
import multiprocessing as mp

from .fill_db import crossVal, fit_kernel, evaluate, read_kernel, load_dataset, load_fragments
from .._dbType import Database
from ..stimulus import Stimulus_db

class Kernel_db(Database):
    """The kernel database has three tables:
    - Main
    - Kernels (makes ref to Main)
    - Fit_kernels (makes ref to Kernels)

    The Main table has 3 columns (other than the primary key):
    - agent
    - coherence
    - filtering_method: for now Literal['unfiltered', 'remove_after_tgt']

    The Kernels table has 7 columns:
    - main_key INTEGER
    - kernel_output TEXT
    - train_error REAL
    - test_error REAL
    - kernel_type TEXT
    - kernel_method TEXT
    - method_param: either 'no_param' or a tuple of floats converted into str
    """

    def __init__(self, location=None):
        super().__init__()
        if location is None:
            self.db_path = __file__
            for _ in range(4):
                self.db_path = os.path.dirname(self.db_path)
            self.db_path = os.path.join(
                self.db_path, 'data', 'solo'
            )
        elif location == 'memory':
            self.db_path = os.environ['LOCAL_TMPDIR']
        self.db_name = "kernel.db"

    def visu(self):
        """We have 2 figures:
        - one where we visualize various kernel functions (implement only multiselect buttons)
        - one where we visualize the train and test errors of the plotted kernels
        """
        path_to_script = __file__
        for _ in range(3):
            path_to_script = os.path.dirname(path_to_script)
        path_to_script = os.path.join(
            path_to_script,
            "visu_db.py"
        )
        subprocess.run(["streamlit", "run", path_to_script])

    def _create_main(self):
        """Creates the `Main` table."""
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS Main (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent TEXT,
            coh REAL,
            filtering_method TEXT,
            UNIQUE(agent, coh, filtering_method)
        )
        """)

    def _create_kernels(self):
        """Creates the `Kernels` table."""
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS Kernels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            main_key INTEGER,
            kernel_output TEXT,
            train_error REAL,
            test_error REAL,
            kernel_type TEXT,
            kernel_method TEXT,
            method_param TEXT,
            FOREIGN KEY (main_key) REFERENCES main(id),
            UNIQUE(main_key, kernel_type, kernel_method, method_param)
        )
        """)

    def _create_fits(self):
        """Creates the `Fit_kernels` table."""
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS Fit_kernels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kernel_key INTEGER,
            fit_output TEXT,
            error REAL,
            fit_type TEXT,
            fit_method,
            FOREIGN KEY (kernel_key) REFERENCES Kernels(id),
            UNIQUE(kernel_key, fit_type, fit_method)
        )
        """)

    def create(self):
        self.connect()

        self._create_main()
        self._create_kernels()
        self._create_fits()
        self.conn.commit()
        print("Kernel database created")
        print()

    def _fill_main(self):
        """Fills the `Main` table.
        """
        # connect to the stimulus database
        stim_db = Stimulus_db()
        stim_db.connect()

        # load all pairs (agent, coh) occurring in the stimulus db
        # note that some agents have distinct coherence values
        keys = stim_db.cur.execute("""
            SELECT DISTINCT
                agent, coh
            FROM
                Main
        """).fetchall()
        stim_db.close()

        # define the filtering methods:
        # they will define what data to load from the stimulus db as a dataset
        # to train the kernels
        filter_meths = ['unfiltered', 'remove_after_tgt']

        # fill the main table
        rows = []
        for agent, coh in keys:
            for filter_meth in filter_meths:
                rows.append( (agent, coh, filter_meth) )

        self.connect()
        self.cur.executemany("""
            INSERT OR IGNORE INTO Main (agent, coh, filtering_method)
            VALUES (?, ?, ?)
        """, rows)
        self.conn.commit()
        self.close()

        print("Main table filled")
        print()

    def _kernel_single_row(self, row_main: tuple):
        """Processes a single row of the `Main` table from the `Stimulus` database.
        
        Returns a list of rows to be inserted in the `Kernels` table of the `Kernel` database.
        """
        # rows to insert into the Kernels table
        rows = []

        dataset = load_dataset(*row_main[1:])

        for triple in self._gen_triples():
            kernel_output, train_error, test_error = crossVal(
                dataset, *triple[:2], json.loads(triple[2])
            )

            # make sure the data format of the row is correct
            row = (
                row_main[0], json.dumps(kernel_output),
                train_error, test_error, *triple
            )
            rows.append(row)
        return rows

    def _gen_triples(self):
        """Generates the triples for computing the `Kernels` table."""
        # all possible values
        kernel_types = ['raw', 'param1']
        kernel_methods = ['linear_reg', 'lasso', 'ridge', 'curve_fit', 'nested_sampling']
        method_params = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

        # specify a subset of triples (kernel_type, kernel_method, method_param)
        triples = [('raw', 'linear_reg', json.dumps('no_param'))]
        for kernel_method in ['lasso', 'ridge']:
            triples.extend(
                [('raw', kernel_method, json.dumps(param)) for param in method_params]
            )
        for kernel_method in ['nested_sampling']:
            triples.append( ('param1', kernel_method, json.dumps('no_param')) )
        return triples

    def _fill_kernels_chunk(self, main_rows: list, n_cpus: int):
        """Processes in parallel each row contained in `main_rows`.
        """
        # make sure to close the connection to the db to avoid multiprocessing to
        # raise an error about not being able to pickle a sqlite3.Connection object
        self.close()

        with mp.Pool( processes=min(n_cpus, len(main_rows)) ) as pool:
            # list_rows is a list of list of tuples
            list_rows = pool.map(self._kernel_single_row, main_rows)

        rows = []
        for el in list_rows:
            rows.extend(el)

        # restore the connection to the db to insert the rows
        self.connect()

        # insert the accumulated rows into the Kernels table
        self.cur.executemany(
            """
            INSERT OR IGNORE INTO Kernels (
                main_key, kernel_output, train_error, test_error,
                kernel_type, kernel_method, method_param
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows
        )
        self.conn.commit()
        self.close()

    def _get_main_keys(self,
        n_cpus: int,
        max_rows_per_cpu: int,
        n_scripts: int,
        script_num: int,
        nn_rows: int
    ):
        if max_rows_per_cpu * n_scripts * n_cpus < nn_rows:
            raise ValueError("too many rows, choose a higher value for `max_rows_per_cpu`")

        # as many scripts as possible will receive max_rows_per_cpu
        max_rows_per_node = n_cpus * max_rows_per_cpu
        max_num = nn_rows // max_rows_per_node
        if script_num < max_num:
            first_row = 1 + script_num * max_rows_per_node
            return (first_row, first_row + max_rows_per_node)

        # if there is at least 1 node without any row to process
        if max_num < n_scripts - 1:
            raise ValueError("choose a lower value for `n_scripts`,"
            " because one or more scripts will have no row to process")
        first_row = (n_scripts - 1) * max_rows_per_node + 1
        return (first_row, nn_rows + 1)

    def _fill_kernels(self,
        max_rows_per_cpu: int,
        n_cpus_max: int,
        script_num: int=0,
        n_scripts: int=1
    ):
        self.connect()
        # for each row of Main and each triple, compute the kernels and evaluate them
        n_rows = len(self.cur.execute("""SELECT * FROM Main""").fetchall())
        n_rows = 10
        self.close()

        n_cpus = min(n_cpus_max, mp.cpu_count())
        key1, key2 = self._get_main_keys(
            n_cpus=n_cpus, max_rows_per_cpu=max_rows_per_cpu, n_scripts=n_scripts,
            script_num=script_num, nn_rows=n_rows
        )

        print(f"processing rows {key1} to {key2 - 1}, script nb {script_num}")

        self.connect()
        main_rows = self.cur.execute("""
            SELECT *
            FROM Main
            WHERE
                id >= ?
                AND id < ?
        """, (key1, key2)).fetchall()
        self.close()
        self._fill_kernels_chunk(main_rows, n_cpus=n_cpus)

        print(f"Kernels table filled, script nb {script_num}")
        print()

    def _read_kernel_fct(self, kernel_key: int):
        """Returns the kernel fct (its 300 complex values) as a 1d numpy.ndarray.

        Parameters
        ----------
        kernel_key : int
            identifies the row of the `Kernels` table that identifies the kernel
        """
        kernel_output, kernel_type, kernel_method = self.cur.execute(
            """SELECT kernel_output, kernel_type, kernel_method FROM Kernels
            WHERE id = ?""", (kernel_key,)
        ).fetchone()
        return read_kernel(json.loads(kernel_output), kernel_type, kernel_method)

    def _fill_fit_kernels(self):
        """Fills the `Fit_kernels` table.
        """
        # all possible values
        fit_types = ['param1', 'param2', 'param3']
        fit_methods = ['curve_fit', 'nested_sampling']

        # specify a subset of couples (fit_type, fit_method)
        # couples = [('param2', 'curve_fit'), ('param2', 'nested_sampling')]
        couples = [('param3', 'curve_fit')]

        # for each row of Kernels and each couple, fit the kernel and evaluate
        # the reconstruction error made by the fit
        keys = self.cur.execute("""SELECT id, main_key FROM Kernels""").fetchall()
        for kernel_key, main_key in keys:
            print(f"processing kernel number {kernel_key}")

            # the kernel function (its 300 complex values)
            kernel = self._read_kernel_fct(kernel_key)

            # the dataset that was used to extract the kernel
            # (the fit will be evaluated on the same dataset)
            agent, coh, filtering_method = self.cur.execute(
                """SELECT agent, coh, filtering_method FROM Main WHERE id = ?""",
                (main_key,)
            ).fetchone()
            dataset = load_dataset(agent, coh, filtering_method)

            for couple in couples:
                # if the couple had already been processed, do not process it again
                # (nested_sampling takes time)
                testRow = self.cur.execute("""
                    SELECT
                        kernel_key, fit_type, fit_method
                    FROM
                        Fit_kernels
                    WHERE
                        kernel_key = ?
                        AND fit_type = ?
                        AND fit_method = ?
                """, (kernel_key, *couple)).fetchall()
                
                # if the row is not present, insert it
                if not testRow:
                    print(f"processing couple {couple}")

                    fit_output, kernel_fct = fit_kernel(kernel, *couple)
                    error = evaluate(kernel_fct, dataset)

                    # make sure the data format of the row is correct
                    row = (
                        kernel_key, json.dumps(fit_output),
                        error, *couple
                    )

                    # insert the row into the Fit_kernels table
                    self.cur.execute(
                        """
                        INSERT OR IGNORE INTO Fit_kernels (
                            kernel_key, fit_output, error,
                            fit_type, fit_method
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """, row
                    )
                    print()
                    self.conn.commit()
        print("Fit_kernels table filled")
        print()

    def fill(self):
        # fill the main table:
        # it contains the agent names, coherence values and filtering method
        # the names and coherence values are imported from the stimulus database
        self._fill_main()

        # generate all kernels using various models, optimization and regularization methods
        self._fill_kernels()

        # generate all fits using various models and optimization methods
        self._fill_fit_kernels()
