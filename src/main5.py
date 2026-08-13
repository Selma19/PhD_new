"""Draws the figures using the old data stored in external storage.

The root of this storage is:
root = /Volumes/Selma_PhD/PhD/Backup/CPR_many_files/Felix_project/Data/Solo_data/CPR_psychophysics

to access the raw kernel of the agent abc and coherence coh, read the file located at:
load_path = $root/abc/Analysis/Kernels/Basic/coh/train_and_save_kernel_size_300.txt

this kernel is loaded with:
kernel = np.loadtxt(load_path).view(complex)
"""

import json, os, pickle
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt

from database.kernel.fill_db._fill_db import Fit_param3

# directory for figures
fig_dir = __file__
for _ in range(2):
    fig_dir = os.path.dirname(fig_dir)
fig_dir = os.path.join(fig_dir, 'figures')

# root for data:
rootDir = '/Volumes/Selma_PhD/PhD/Backup/CPR_many_files/Felix_project/Data/Solo_data/CPR_psychophysics/'

# coherences
cohs = [
    0.07999999821186066, 0.13199999928474426, 0.21780000627040863,
    0.3594000041484833, 0.5929999947547913, 0.9783999919891357
]

# agents
agents = []
for name in os.listdir(rootDir):
    if len(name) == 3:
        agents.append(name)
n_agents = len(agents)

def load_kernel(agent, coh):
    """Returns the modulus of the raw kernel associated to agent, coh."""
    load_path = os.path.join(
        rootDir, agent, 'Analysis',
        'Kernels', 'Basic', str( int(coh * 1000) ),
        'train_and_save_kernel_size_300.txt'
    )
    return np.abs(np.loadtxt(load_path).view(complex)) * coh

def figure2_raw(use_cache=True):
    if use_cache:
        kernels = pickle.load(
            open(os.path.join(fig_dir, 'figure2_raw', 'cache'), '+rb')
        )

    else:
        # get the kernels modulus
        kernels = [np.zeros( (n_agents, 300) ) for _ in range(len(cohs))]
        for idx_coh, coh in enumerate(cohs):
            print(f"{len(cohs) - idx_coh} cohs remaining")
            for i, agent in enumerate(agents):
                kernels[idx_coh][i, :] = load_kernel(agent, coh)

        # store in a cache
        pickle.dump(
            kernels,
            open(os.path.join(fig_dir, 'figure2_raw', 'cache'), '+wb')
        )

    # compute the average and std of kernel modulus over agents:
    # means[idx_coh] = average most probable kernel modulus over agents
    means = []
    stds = []
    for idx_coh, tab in enumerate(kernels):
        means.append(np.mean(tab, axis=0))
        stds.append(np.std(tab, axis=0))

    # plot
    # times in sec
    x = np.linspace(0, 1, 300) * 8.33 * 1e-3
    _, ax = plt.subplots(1, 1, constrained_layout=True)
    fontsize = 14
    ax.set_ylabel(r'$|k(t)|$', fontsize=fontsize)
    ax.set_xlabel(r'$t$' + ' (sec)', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)

    # there are 7 cohs, one color per coh
    colors = ['b', 'k', 'r', 'green', 'purple', 'cyan', 'pink']

    # clean the kernel values
    clean_means = []
    for y in means:
        z = Fit_param3()._clean_kernel(y)
        z[0] = 300
        z[-1] = 300
        clean_means.append(z)

    for coh, color, y, std in zip(cohs, colors, clean_means, stds):
        label = f"coh = {coh:.2f}"
        indices = (y < 250).nonzero()
        X = x[indices]
        Y = y[indices]
        Z = std[indices]
        ax.plot(X, Y, '-', color=color, label=label)
        ax.fill_between(X, Y - Z / 2, Y + Z / 2, color=color, alpha=0.1)

    ax.legend(fontsize=fontsize)

    plt.savefig(os.path.join(fig_dir, 'figure2_raw', 'kernel.png'))
    plt.show()

figure2_raw(use_cache=True)
