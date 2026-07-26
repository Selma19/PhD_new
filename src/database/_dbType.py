"""Defines the Database ABC."""
import os, subprocess, sqlite3

class Database:
    """Abstract base class for a database.
    
    Notes
    -----
    SQLITE3 treats NULL values as all different!!!!
    So never use a NULL as a default value ; NULL should only be used in case a feature
    is not defined.
    """

    def __init__(self):
        self.conn = None
        self.cur = None
        self.db_path = ""
        self.db_name = ".db"

    def connect(self):
        """Creates a `sqlite3` connexion with the database."""
        if self.conn is None:
            self.conn = sqlite3.connect(
                os.path.join(self.db_path, self.db_name)
            )
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self.cur = self.conn.cursor()

    def close(self):
        try:
            self.conn.close()
            self.conn = None
            self.cur = None
        except AttributeError:
            pass

    def create(self):
        """Initializes the database skeleton (tables, foreign keys, etc.),
        i.e. creates the columns.
        """
    
    def fill(self):
        """Fills the database, i.e. appends the rows."""

    def visu(self):
        """Visualizes the database content with streamlit.
        """

    def clear(self):
        """Deletes the database."""
        self.close()
        subprocess.run(['rm', os.path.join(self.db_path, self.db_name)])
