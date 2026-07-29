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

    def connect(self,
        read_uncommitted=False,
        uri_suffix: str | None=None
    ):
        """Creates a `sqlite3` connexion with the database and a cursor."""
        filepath = os.path.join(self.db_path, self.db_name)
        if self.conn is None:
            if uri_suffix is None:
                self.conn = sqlite3.connect(filepath)
            else:
                self.conn = sqlite3.connect(
                    f"file:{filepath}{uri_suffix}",
                    uri=True
                )
            #self.conn.execute("PRAGMA foreign_keys = ON;")
            if read_uncommitted:
                self.conn.execute("PRAGMA read_uncommitted = True;")
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
