"""Draws the figures for the paper.

3 figures:

Figure 1 (visualize parameters individually):
- one panel per parameter
- coherence as x axis
- parameter value as y axis, organized as violins (1 point per coh per agent)
- 3 plots per panel, one for most probable param, one for the means, one for the stds

Figure 2 (visualize kernels):
- one panel
- compute the most probable kernel (mpk) for each agent, then draw,
for each coherence, the mpk averaged over agents as well as the std over agents
drawn as shades

Figure 3 (covariance btw parameters):
- one panel per coherence
- for each agent compute the covariance matrix btw params (use the nautilus built-in fcts),
then average this matrix over all agents and plot it
"""

import json, os, pickle
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt

from database import Kernel_db

# directory for figures
fig_dir = __file__
for _ in range(2):
    fig_dir = os.path.dirname(fig_dir)
fig_dir = os.path.join(fig_dir, 'figures')

k_type = 'param1'
k_meth = 'nested_sampling'
filtering_method = 'unfiltered'

db = Kernel_db()
db.connect()

def get_dic():
    # get the rows id (main_key) in the Main table
    # that meet our needs
    rows = db.cur.execute("""
        SELECT id, agent, coh
        FROM Main
        WHERE
            filtering_method = ?
        ORDER BY
            coh
    """, (filtering_method,)).fetchall()

    # organize them as a list of lists where
    # dic[idx_coh] = [id1, id2, ...] corresponding to coh = cohs[idx_coh]
    dic = []
    cohs = []
    current_coh = -1

    for row in rows:
        coh = row[2]

        if coh == current_coh:
            # append the id
            dic[-1].append(row[0])

        else:
            current_coh = coh
            dic.append([row[0]])
            cohs.append(coh)

    return cohs, dic

def get_data(restrict_cohs=True):
    """For each id, get the kernel output, then
    the kernel parameters and their uncertainties:
    means[param_name] = [
      [mean_coh_1_agent1, mean_coh_1_agent2],
      [mean_coh_2_agent1, mean_coh_2_agent2], ...
    ]
    """
    cohs, dic = get_dic()

    if restrict_cohs:
        idx_to_keep = range(0, len(cohs), 2)
        cohs = [cohs[idx] for idx in idx_to_keep]
        dic = [dic[idx] for idx in idx_to_keep]

    stds = {}
    max_vals = {}
    means = {}
    for idx_coh, ids_by_coh in enumerate(dic):
        print(f"{len(cohs) - idx_coh} steps remaining")
        for main_key in ids_by_coh:
            k_out = db.cur.execute("""
                SELECT kernel_output
                FROM Kernels
                WHERE
                    main_key = ?
                AND
                    kernel_type = ?
                AND
                    kernel_method = ?
            """, (main_key, k_type, k_meth)).fetchone()[0]

            points, log_w = json.loads(k_out)
            prob_w = softmax(log_w)

            for name, val in points.items():
                tab = np.array(val)
                mean = np.sum(prob_w * tab)
                std = np.sum(prob_w * tab ** 2) - mean ** 2
                max_val = tab[np.argmax(prob_w)]

                if name in stds:
                    stds[name][idx_coh].append(std)
                    means[name][idx_coh].append(mean)
                    max_vals[name][idx_coh].append(max_val)

                else:
                    stds[name] = [[] for _ in range(len(cohs))]
                    means[name] = [[] for _ in range(len(cohs))]
                    max_vals[name] = [[] for _ in range(len(cohs))]
                    stds[name][idx_coh].append(std)
                    means[name][idx_coh].append(mean)
                    max_vals[name][idx_coh].append(max_val)

    # store in the cache for reuse
    cache_path = os.path.join(fig_dir, 'figure1', 'cache')
    pickle.dump((cohs, means, stds, max_vals), file=open(cache_path, '+wb'))
    return cohs, means, stds, max_vals

def figure1():
    # load the cache or not
    load_cache = True
    restrict_cohs = True
    if load_cache:
        cohs, means, stds, max_vals = pickle.load(open(os.path.join(fig_dir, 'figure1', 'cache'), '+rb'))
    else:
        cohs, means, stds, max_vals = get_data(restrict_cohs=restrict_cohs)

    # plot the figure 1:
    # one panel per parameter;
    # coherence as x axis
    # parameter value as y axis, organized as violins
    # 3 plots per panel, one for max vals, one for the means, one for the stds
    params = list(means)

    fontsize = 14
    for param in params:
        fig, axs = plt.subplots(3, 1, constrained_layout=True)
        fig.suptitle(f"parameter {param}", fontsize=fontsize)

        # set the ticks and labels
        ## y axis
        axs[0].set_ylabel('most probable', fontsize=fontsize)
        axs[1].set_ylabel('mean', fontsize=fontsize)
        axs[2].set_ylabel('std', fontsize=fontsize)

        axs[0].tick_params(labelsize=13, axis='y')
        axs[1].tick_params(labelsize=13, axis='y')

        ## x axis
        axs[0].get_xaxis().set_visible(False)
        axs[1].get_xaxis().set_visible(False)

        axs[2].tick_params(labelsize=13)
        labels = [f"{coh:.2f}" for coh in cohs]
        axs[2].set_xticks(ticks=range(len(cohs)), labels=labels)

        # plot the violins
        axs[0].violinplot(
            max_vals[param],
            positions=range(len(cohs)),
            showmeans=True, showextrema=True
        )

        axs[1].violinplot(
            means[param],
            positions=range(len(cohs)),
            showmeans=True, showextrema=True
        )

        axs[2].violinplot(
            stds[param],
            positions=range(len(cohs)),
            showmeans=True, showextrema=True
        )

        plt.savefig(os.path.join(fig_dir, 'figure1', param + '.png'))
    plt.close('all')

def figure2():
    pass

def figure3():
    pass
