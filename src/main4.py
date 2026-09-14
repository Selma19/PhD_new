"""Draws the figures for the paper using the data in `data/solo`
(the databases filled from the h5py files).

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

from typing import Literal
import json, os, pickle
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt

from database import Kernel_db
from database.kernel.fill_db.utils import exponential_Fred

# directory for figures
fig_dir = __file__
for _ in range(2):
    fig_dir = os.path.dirname(fig_dir)
fig_dir = os.path.join(fig_dir, 'figures')

k_type = 'param1'
k_meth = 'nested_sampling'

db = Kernel_db()
db.connect()

def get_dic(filtering_method: str, restrict_cohs=True):
    # get the rows id (main_key) in the Main table
    # that meet our needs
    rows = db.cur.execute("""
        SELECT id, agent, coh
        FROM Main
        WHERE
            filtering_method = ?
        ORDER BY
            coh, agent
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

    if restrict_cohs:
        idx_to_keep = range(0, len(cohs), 2)
        cohs = [cohs[idx] for idx in idx_to_keep]
        dic = [dic[idx] for idx in idx_to_keep]

    return cohs, dic

def get_data(cohs, dic):
    """For each id, get the kernel output, then
    the kernel parameters and their uncertainties:
    means[param_name] = [
      [mean_coh_1_agent1, mean_coh_1_agent2],
      [mean_coh_2_agent1, mean_coh_2_agent2], ...
    ]
    """
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

    return means, stds, max_vals

def get_max_vals(cohs, dic):
    """max_vals[param_name][idx_coh] = [
        param(param_name, idx_coh, agent_k)
    ]
    """
    max_vals = {}
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
            idx_max = np.argmax(log_w)

            for name, val in points.items():
                max_val = val[idx_max]

                if name in max_vals:
                    max_vals[name][idx_coh].append(max_val)

                else:
                    max_vals[name] = [[] for _ in range(len(cohs))]
                    max_vals[name][idx_coh].append(max_val)

    return max_vals

def figure1(
        filtering_method: Literal['unfiltered', 'remove_after_tgt'],
        use_cache: bool,
        restrict_cohs: bool = True
    ):
    """Visualize parameters individually:
    - one panel per parameter
    - coherence as x axis
    - parameter value as y axis, organized as violins (1 point per coh per agent)
    - 3 plots per panel, one for most probable param, one for the means, one for the stds
    """
    # load the cache or not
    if use_cache:
        cohs, means, stds, max_vals = pickle.load(
            open(os.path.join(fig_dir, 'figure1', 'cache_' + filtering_method), '+rb')
        )

    else:
        cohs, dic = get_dic(filtering_method, restrict_cohs=restrict_cohs)
        means, stds, max_vals = get_data(cohs, dic)

        # store in the cache for reuse
        cache_path = os.path.join(fig_dir, 'figure1', 'cache_' + filtering_method)
        pickle.dump((cohs, means, stds, max_vals), file=open(cache_path, '+wb'))

    # plot the figure 1:
    # one panel per parameter;
    # coherence as x axis
    # parameter value as y axis, organized as violins
    # 3 plots per panel, one for max vals, one for the means, one for the stds
    params = list(means)

    fontsize = 14
    for param in params:
        fig, axs = plt.subplots(3, 1, constrained_layout=True)
        title = f"parameter {param}"

        if filtering_method == 'unfiltered':
            title += "; no filter"

        else:
            title += "; target influence removed"

        fig.suptitle(title, fontsize=fontsize)

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

        plt.savefig(os.path.join(
            fig_dir, 'figure1', param + '_' + filtering_method + '.png'
        ))
    plt.close('all')

def figure2():
    """popts[idx_coh][k] = {param: opt_val for param in params}
    
    where each k corresponds to a distinct agent
    """
    # get the coherences with main_keys for each agent
    cohs, dic = get_dic(restrict_cohs=True)
    n_agents = len(dic[0])

    # get the most probable parameters (according to joint law)
    max_vals = get_max_vals(cohs, dic)

    # build the kernels modulus from the parameters:
    # kernels[idx_coh] = [kernel_1, kernel_2, ...]
    kernels = [np.zeros( (n_agents, 300) ) for _ in range(len(cohs))]
    for idx_coh in range(len(cohs)):
        for i in range(n_agents):
            popt = {param: val[idx_coh][i] for param, val in max_vals.items()}
            kernels[idx_coh][i, :] = np.abs(exponential_Fred(np.linspace(0, 1, 300), **popt))

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
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fontsize = 14
    ax.set_ylabel(r'$|k(t)|$', fontsize=fontsize)
    ax.set_xlabel(r'$t$' + ' (sec)', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)

    # there are 7 cohs, one color per coh
    colors = ['b', 'k', 'r', 'green', 'purple', 'yellow', 'pink']

    # display the kernel of agent idx_agent
    idx_agent = 20

    for idx_coh, color in enumerate(colors):
        coh = cohs[idx_coh]
        y = kernels[idx_coh][idx_agent, :]
        label = f"coh = {coh:.2f}"
        ax.plot(x, y, '.', color=color, label=label)

    """for idx_coh, color in enumerate(colors):
        coh = cohs[idx_coh]
        y = means[idx_coh]
        label = f"coh = {coh:.2f}"
        ax.plot(x, y, '.', color=color, label=label)"""
    
    ax.legend(fontsize=fontsize)

    plt.savefig(os.path.join(fig_dir, 'figure2', 'kernel.png'))
    plt.show()

def get_data_raw(cohs, dic):
    """For each id, get the modulus of the raw kernel:
    kernels[idx_coh] = [kernel_1, kernel_2, ...]
    """
    n_agents = len(dic[0])

    # collect the kernels modulus:
    # kernels[idx_coh] = [kernel_1, kernel_2, ...]
    kernels = [np.zeros( (n_agents, 300) ) for _ in range(len(cohs))]
    for idx_coh in range(len(cohs)):
        for i in range(n_agents):
            k_out = db.cur.execute("""
                SELECT kernel_output
                FROM Kernels
                WHERE
                    main_key = ?
                AND
                    kernel_type = ?
                AND
                    kernel_method = ?
            """, (dic[idx_coh][i], 'raw', 'linear_reg')).fetchone()[0]
            tab_x, tab_y = json.loads(k_out)
            kernels[idx_coh][i, :] = np.sqrt(np.array(tab_x) ** 2 + np.array(tab_y) ** 2)
    return kernels

def figure2_raw():
    """See main5.py for generating the figures about raw kernels;
    this one is DEPRECIATED.
    """
    # get the coherences with main_keys for each agent
    cohs, dic = get_dic(restrict_cohs=True)

    # get the kernels modulus
    kernels = get_data_raw(cohs, dic)

    """
    # compute the average and std of kernel modulus over agents:
    # means[idx_coh] = average most probable kernel modulus over agents
    means = []
    stds = []
    for idx_coh, tab in enumerate(kernels):
        means.append(np.mean(tab, axis=0))
        stds.append(np.std(tab, axis=0))
    """

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

    # display the kernel of agent idx_agent
    idx_agent = 0

    # choose a coherence
    idx_coh = 6
    coh = cohs[idx_coh]
    color = colors[idx_coh]
    tabs = kernels[idx_coh]

    #for coh, color, tabs in zip(cohs, colors, kernels):
    y = tabs[idx_agent, :]
    label = f"coh = {coh:.2f}"
    ax.plot(x, y, '.', color=color, label=label)

    ax.legend(fontsize=fontsize)

    plt.savefig(os.path.join(fig_dir, 'figure2_raw', 'kernel.png'))
    plt.show()

figure1(filtering_method='unfiltered', use_cache=False)
figure1(filtering_method='remove_after_tgt', use_cache=False)
