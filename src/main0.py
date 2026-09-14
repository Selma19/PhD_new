"""Creates and fills from the h5 files the stimulus database for all agents available.
"""

from database import Stimulus_db

if __name__ == "__main__":
    db = Stimulus_db()
    db.clear()
    db.create()
    db.fill(choice='folders', n_cpus_max=38)
    db.close()
