"""Creates and fills from the h5 files the stimulus database for all agents available.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

from database import Stimulus_db, load_fragments

if __name__ == "__main__":
    db = Stimulus_db()
    db.create()
    db.fill(
        choice='h5',
        n_cpus_max=os.environ['SLURM_CPUS_PER_TASK']
    )
