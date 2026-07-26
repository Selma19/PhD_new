from database import Kernel_db, Stimulus_db, load_fragments
from database.kernel import fit_kernel
import matplotlib.pyplot as plt
import numpy as np
import sys

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

def display_fit(kernel_key: int):
    """To figure out how to extract clean kernels from noisy data, when the noise is not
    additive, but replaces the continuous signal.

    This test was triggered by ('param2', 'curve_fit') on kernel_key 120.
    """
    db = Kernel_db()
    db.connect()

    couple = ('param3', 'curve_fit')

    kernel = db._read_kernel_fct(kernel_key)
    z = np.abs(kernel)

    y = np.abs(kernel)[1:-1]
    diff_kernel = np.abs(y[1:] - y[:-1])

    noise_strength = np.max(diff_kernel) / np.max(y)
    print(f"noise_strength {noise_strength}")

    if noise_strength < 0.125:
        print("No significant noise here")
        clean_kernel = z
    else:
        hist, bin_edges = np.histogram(diff_kernel, bins=20)
        for ind in range(1, len(hist)):
            if hist[ind] >= hist[ind - 1]:
                break
        threshold = (bin_edges[ind - 1] + bin_edges[ind]) / 2

        fig, ax = plt.subplots(1, 1)
        plt.hist(diff_kernel, bins=20)
        ax.plot([threshold] * 2, [0, ax.get_ylim()[1] * 0.9], '--', label='threshold')
        ax.set_xlabel(r"$|K_t-K_{t-1}|$")
        ax.set_ylabel(r"$p(\Delta K)$")
        ax.legend()
        plt.show()

        diff_kernel = np.abs(z[1:] - z[:-1])
        indices = (diff_kernel >= threshold).nonzero()

        clean_kernel = z.copy()
        clean_kernel[indices] = 0
        clean_kernel[0] = z[0]
        clean_kernel[-1] = z[-1]

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(clean_kernel, '.')
    # ax.plot(z, 'x')
    # plt.show()

    fit_output, kernel_fct = fit_kernel(clean_kernel, *couple)

    print(fit_output)
    """for 'param1' it is:
    {'tau1': 0.11222543780886862, 'tau2': 2.9059443900266033,
    'alpha': 1.3038655320650128, 'd': 1.471004437088596e-94, 'A': 8.261459611268764}
    """

    fig, ax = plt.subplots(1, 1)
    ax.plot(z, 'x', label='kernel')
    ax.plot(clean_kernel, '.', label='clean kernel', markersize=5)
    ax.plot(kernel_fct, '.', label='fit', alpha=0.5)
    ax.legend()
    plt.show()

if __name__ == "__main__":
    # # create the stimulus database for all agents available
    # db = Stimulus_db()
    # db.clear()
    # db.create()
    # db.fill(choice='h5', n_cpus_max=38)
    # exit()

    # create the kernel database for all agents available
    # check arguments sent to the program
    try:
        args = sys.argv[1:3]
        script_num = int(args[0])
        n_scripts = int(args[1])
    except:
        script_num = 0
        n_scripts = 1
    
    db = Kernel_db(location='memory')
    db._fill_kernels(
        max_rows_per_cpu=2,
        n_cpus_max=50,
        script_num=script_num,
        n_scripts=n_scripts
    )

    #visu_filtered_data()

    # Kernel_db().visu()
    #exit()
    #kernel_key = 201 # 107 120 201 16 33 62 66 136 56
    #display_fit(kernel_key)
