"""Creates and fills from the h5 files the stimulus database for all agents available.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

from database import Stimulus_db, load_fragments

def visu_filtered_data():
    """Visualizes the filtered data, in particular the time intervals that were selected,
    and computes their distribution in size.
    """
    # load the coherence values (note we want unique values)
    db = Stimulus_db()
    db.connect()
    cohs = db.cur.execute("""
        SELECT DISTINCT coh FROM Main
    """).fetchall()
    db.close()
    cohs = [coh[0] for coh in cohs]

    # load the filtered data
    agent = 'aaa'
    fragments_size = []
    for coh in cohs:
        print(coh)
        joystick_list, dot_list = load_fragments(agent, coh, min_length=0)
        fragments_size.extend([len(fragment) for fragment in dot_list])
    
    # visualize the distribution
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel("fragment length")
    ax.set_ylabel("nb of fragments")
    ax.set_title("distribution of data fragment length after filtering")
    
    median = np.median(fragments_size)
    plt.hist(fragments_size)
    ax.plot([median] * 2, [0, 740], '--', color='red', label='median length')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    db = Stimulus_db()
    db.create()
    db.fill(
        choice='h5',
        n_cpus_max=os.environ['SLURM_CPUS_PER_TASK']
    )
