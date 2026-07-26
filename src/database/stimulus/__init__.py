"""Defines the stimulus database."""

from typing import Literal
import os, json
import multiprocessing as mp

from .fill_db import from_folders, from_h5
from .loadData import get_list_block
from .._dbType import Database

class Stimulus_db(Database):
    """The stimulus database has a single table:
    - Main

    A row of this table is completely determined by the tuple (agent, coh, xp_name, block).
    
    Each row of the column nominal_angle is a (json) list of 10 couples
    (time of change in nominal direction, new nominal direction).
    Time steps are expressed in ms and angles in radians.
    """

    def __init__(self, location: str| None=None):
        super().__init__()
        if location is None:
            self.db_path = __file__
            for _ in range(4):
                self.db_path = os.path.dirname(self.db_path)
            self.db_path = os.path.join(self.db_path, 'data', 'solo')

        elif location == 'memory':
            self.db_path = os.environ['LOCAL_TMPDIR']

        self.db_name = "stimulus.db"

    def create(self):
        self.connect()
        
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS Main (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent TEXT,
            coh REAL,
            xp TEXT,
            block INTEGER,
            times TEXT,
            nominal_angle TEXT,
            target_times TEXT,
            joystick TEXT,
            mean_dot TEXT,
            UNIQUE(agent, coh, xp, block)     
        )
        """)
        self.conn.commit()
        print("Stimulus database created")
        print()

    def _fill_single_folders(self, agent: str):
        # get the list of coherence values
        cohs = [
            0, 0.079, 0.131, 0.217,
            0.359, 0.592, 0.978
        ]
        
        rows = []

        for coh in cohs:
            blocks = get_list_block(coh, agent)
            for block in blocks:
                # wrap the arguments for readability
                args = (agent, coh, block)

                # accumulate data here as a row to be inserted in the table
                row = (agent, coh, *from_folders.block_xp_num(block))
                for func in [
                    from_folders.get_time_steps, from_folders.get_nom,
                    from_folders.get_tgt_times, from_folders.get_joystick,
                    from_folders.get_dot
                ]:
                    row += (json.dumps( func(*args) ),)
                
                rows.append(row)
        return rows

    def _insert_rows(self, rows: list):
        """Insert `rows` into the `Stimulus` database.
        """
        # insert the row into the table
        self.cur.executemany("""
            INSERT OR IGNORE INTO Main (
                agent, coh, xp, block, times,
                nominal_angle, target_times,
                joystick, mean_dot
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        self.conn.commit()

    def fill(self, choice: Literal['folders', 'h5'], n_cpus_max: int):
        """Fills the `Stimulus` database.
        
        Notes
        -----
        In order to fill from folders for an agent `abc`, the directory `hdf5/abc` must exist.

        The CPR data were downloaded from the Amazon S3 bucket at https://s3.gwdg.de/sfb1528-general.
        """
        if choice == 'folders':
            get_agents = from_folders.get_agents
            fill_single_agent = self._fill_single_folders
        
        elif choice == 'h5':
            get_agents = from_h5.get_agents
            fill_single_agent = from_h5.extract_all
        
        else:
            raise ValueError("check the value of 'choice'")

        # get the list of agents (there are 38 of them)
        agents = get_agents()

        print("nb of cpu cores available:", mp.cpu_count())
        print("nb of agents:", len(agents))
        print()

        # process each agent in parallel
        self.close()
        n_cpus = min(n_cpus_max, mp.cpu_count())
        with mp.Pool( processes=min(len(agents), n_cpus) ) as pool:
            # list_rows is a list of list of tuples
            list_rows = pool.map(fill_single_agent, agents)

        print('gathered all rows to insert')
        print()
        
        # insert agent data into the database
        self.connect()
        
        # rows is a list of tuples
        rows = []
        for el in list_rows:
            rows.extend(el)
        self._insert_rows(rows)
        self.close()
