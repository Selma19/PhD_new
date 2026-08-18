"""Draws the figures using the old data stored in external storage.

The root of this storage is:
root = /Volumes/Selma_PhD/PhD/Backup/CPR_many_files/Felix_project/Data/Solo_data/CPR_psychophysics

to access the raw kernel of the agent abc and coherence coh, read the file located at:
load_path = $root/abc/Analysis/Kernels/Basic/coh/train_and_save_kernel_size_300.txt

this kernel is loaded with:
kernel = np.loadtxt(load_path).view(complex)
"""

from typing import List, Literal
import json, os, pickle
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt

from database.kernel.fill_db._fill_db import Fit_param3
from database import Kernel_db

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

def get_agents(use_cache=True):
    cachePath = os.path.join(fig_dir, 'caches', 'agents')

    if use_cache:
        return pickle.load(open(cachePath, '+rb'))

    else:
        agents = []
        for name in os.listdir(rootDir):
            if len(name) == 3:
                agents.append(name)
        pickle.dump(
            agents, open(cachePath, '+wb')
        )

    return agents

def load_kernel_unfiltered(agent: str, coh: float):
    """Returns the modulus of the raw kernel associated to agent, coh,
    when no filter has been applied to the sensory data (dot complex direction).
    """
    load_path = os.path.join(
        rootDir, agent, 'Analysis',
        'Kernels', 'Basic', str( int(coh * 1000) ),
        'train_and_save_kernel_size_300.txt'
    )
    return np.abs(np.loadtxt(load_path).view(complex)) * coh

def load_kernel_no_tgt(agent: str, coh: float):
    """Returns the modulus of the raw kernel associated to agent, coh,
    when the influence of the target has been removed from the stimulus
    (dot complex direction).
    """
    db = Kernel_db()
    db.connect()

    # identify the primary key corresponding to the filtered data
    # of the agent and coh
    main_key = db.cur.execute("""
        SELECT id
        FROM Main
        WHERE
            filtering_method = ?
        AND
            coh = ?
        AND
            agent = ?
    """, (filtering_method, coh, agent)).fetchone()[0]

    # fetch the raw kernel output
    k_out = db.cur.execute("""
        SELECT kernel_output
        FROM Kernels
        WHERE
            main_key = ?
        AND
            kernel_type = ?
        AND
            kernel_method = ?
    """, (main_key, 'raw', 'linear_reg')).fetchone()[0]
    db.close()

    # return the raw kernel modulus
    tab_x, tab_y = json.loads(k_out)
    return np.sqrt(np.array(tab_x) ** 2 + np.array(tab_y) ** 2)

def get_kernels(
    agents=None,
    use_cache=True,
    filtering_method: Literal['unfiltered', 'remove_after_tgt']='unfiltered'
):
    """Returns kernels[idx_coh][idx_agent, :] = 300 kernel
    values for agent and coh.
    """
    if filtering_method == 'unfiltered':
        cachePath = os.path.join(fig_dir, 'caches', 'kernels')
        load_kernel = load_kernel_unfiltered

    elif filtering_method == 'remove_after_tgt':
        cachePath = os.path.join(fig_dir, 'caches', 'kernels_no_tgt')
        load_kernel = load_kernel_no_tgt

    if use_cache:
        return pickle.load(
            open(cachePath, '+rb')
        )

    else:
        # get the kernels modulus
        kernels = [np.zeros( (len(agents), 300) ) for _ in range(len(cohs))]
        for idx_coh, coh in enumerate(cohs):
            print(f"{len(cohs) - idx_coh} cohs remaining")
            for i, agent in enumerate(agents):
                kernels[idx_coh][i, :] = load_kernel(agent, coh)

        # store in cache
        pickle.dump(
            kernels,
            open(cachePath, '+wb')
        )
        print('done')

    return kernels

def get_popts(
    kernels=None,
    use_cache=True,
    filtering_method: Literal['unfiltered', 'remove_after_tgt']='unfiltered'
):
    """Computes the optimal parameters of Fit_param3
    for the raw kernels of each agent and coherence.

    Then, stores these parameters in a cache, as a list of numpy.ndarray:

    popts[idx_coh][idx_agent, idx_param] = optimal values of the curve_fit parameter
    of index idx_param in Fit_param3().param_names
    """
    if filtering_method == 'unfiltered':
        cachePath = os.path.join(fig_dir, 'caches', 'popts')

    elif filtering_method == 'remove_after_tgt':
        cachePath = os.path.join(fig_dir, 'caches', 'popts_no_tgt')

    # fit the kernels and eventually store the result in the cache
    # popts[idx_coh][idx_agent, idx_param] = optimal values of the curve_fit parameter
    # of index idx_param in Fit_param3().param_names

    if use_cache:
        return pickle.load(
            open(cachePath, '+rb')
        )

    else:
        n_agents = np.size(kernels[0], 0)
        fit = Fit_param3()
        popts = [
            np.zeros( (n_agents, len(fit.param_names)) )
            for _ in range(len(cohs))
        ]

        for idx_coh, tab in enumerate(kernels):
            for idx_agent in range(n_agents):
                fit.fit(tab[idx_agent], method='curve_fit')
                for idx_param, param_name in enumerate(fit.param_names):
                    popts[idx_coh][idx_agent, idx_param] = fit.out[param_name]

        pickle.dump(popts, open(cachePath, '+wb'))

    return popts

def convert_popts(popts: List[np.ndarray], with_log: bool=False):
    """convert the parameters so that they are expressed in sec
    dimensionless parameters are left unchanged.
    """
    conv_popts = [
        np.zeros_like(tab) for tab in popts
    ]
    param_names = Fit_param3().param_names
    n_params = np.size(conv_popts[0], 1)
    for idx_coh in range(len(cohs)):
        for idx_param in range(n_params):
            conv_popts[idx_coh][:, idx_param] = convert_param(
                param_names[idx_param], popts[idx_coh][:, idx_param],
                with_log=with_log
            )
    return conv_popts

def check_popts(agents, kernels, popts):
    """Checks the parameters returned by get_popts recover the correct kernel.
    """
    fit = Fit_param3()
    fit.method = 'curve_fit'
    for idx_agent in range(10):
        for idx_coh in range(len(cohs)):
            fit.out = {name: val for name, val in zip(fit.param_names, popts[idx_coh][idx_agent])}
            fit_ker = fit.read_kernel_from_out()
            ker = kernels[idx_coh][idx_agent, :]
            _, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(ker, '.')
            ax.plot(fit_ker, '-')
            fontsize = 14
            title = f"coh = {cohs[idx_coh]:.2f}, agent {agents[idx_agent]}"
            idx_param = 4
            title += f"\n {fit.param_names[idx_param]} = {popts[idx_coh][idx_agent, idx_param]:.2f}"
            ax.set_title(title, fontsize=fontsize)
            plt.show()

def get_corrmat(conv_popts):
    """Returns the correlation matrix
    btw the parameters of Fit_param3() (stored in popts),
    sampled over the agents for each coherence.
    """
    fit = Fit_param3()
    n_params = len(fit.param_names)
    corrmat = [
        np.zeros( (n_params, n_params) )
        for _ in cohs
    ]
    for idx_coh in range(len(cohs)):
        corrmat[idx_coh] = np.corrcoef(conv_popts[idx_coh], rowvar=False)
    return corrmat

def title_filter(title, filtering_method):
    """Appends the suffix to title that corresponds to filtering_method.
    """
    if filtering_method == 'unfiltered':
        title += "; no filter"

    else:
        title += "; target influence removed"
    return title

def convert_param(name: str, values: List[float], with_log: bool=False):
    """Converts parameter values in order to match
    seconds as parameter units.

    To perform this conversion, consider that:
    - consecutive frames are separated by 8.33 ms
    - the kernel function is defined on [0, 1] and 300 frames
    are sampled
    """
    # d is btw 0 and 1; d = 1 corresponding to 300 frames,
    # and 2 frames being separated by 8.33 ms
    x = 299 * 8.33 * 1e-3
    if name == 'd':
        return values * x

    # note we must invert omega to get a time
    elif name == 'omega1' or name == 'omega2':
        if with_log:
            return np.log10( x / (values + 1e-10) )
        return x / (values + 1e-10)

    # dimensionless parameters are left unchanged
    else:
        return values

def convert_name(
    name: str,
    with_units: bool = True,
    with_log: bool = False
):
    """Replaces the original name of a parameter
    with the name and units of its sec-homogeneous counterpart.
    """
    prefix = name
    suffix = ''
    if name == 'omega1' or name == 'omega2':
        prefix = f'tau{name[-1]}'
        if with_units:
            suffix = ' (sec)'
            if with_log:
                suffix = ' (sec, log scale)'
        elif with_log:
            suffix = ', log10'
    elif name == 'd':
        if with_units:
            suffix = ' (sec)'
    return prefix + suffix

def figure1_raw(
    conv_popts,
    filtering_method: Literal['unfiltered', 'remove_after_tgt']='unfiltered',
    with_log: bool = True
):
    """Figure 1 (visualize parameters individually):
    - one panel per parameter
    - coherence as x axis
    - probability density as y axis, 1 point per coh per agent
    - histogram of parameters returned by curve_fit run on the raw kernels
    """
    # plot
    # popts[idx_coh][idx_agent, idx_param] = optimal values of the curve_fit parameter
    # of index idx_param in Fit_param3().param_names
    fit = Fit_param3()
    fontsize = 18
    for idx_param, param_name in enumerate(fit.param_names):
        if param_name in ['omega1', 'omega2']:
            xlabel = 'value'
            if with_log:
                xlabel = 'log10(value)'
        
        else:
            xlabel = 'value'

        fig, axs = plt.subplots(
            1, len(cohs), constrained_layout=True,
            figsize=(16, 8)
        )
        title = title_filter(
            f"parameter {convert_name(param_name, with_log=with_log)}",
            filtering_method
        )
        fig.suptitle(title, fontsize=fontsize)

        # set the ticks and axes titles
        ## y axis
        axs[0].set_ylabel('probability density', fontsize=fontsize)
        axs[0].tick_params(labelsize=fontsize, axis='y')

        ### hide the y axes on the other subplots (since it is shared)
        for ax in axs[1:]:
            ax.get_yaxis().set_visible(False)

        ## x axis
        for ax in axs:
            ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.tick_params(labelsize=13, axis='x')

        ## titles
        for ax, coh in zip(axs, cohs):
            ax.set_title(f"coh: {coh:.2f}", fontsize=fontsize)

        # compute and plot the distributions
        x_lims = [np.inf, -np.inf]
        y_lims = [np.inf, -np.inf]
        for tab, ax in zip(conv_popts, axs):
            ax.hist(
                tab[:, idx_param], density=True
            )

            x_lim = ax.get_xlim()
            x_lims[0] = min(x_lims[0], x_lim[0])
            x_lims[1] = max(x_lims[1], x_lim[1])

            y_lim = ax.get_ylim()
            y_lims[0] = min(y_lims[0], y_lim[0])
            y_lims[1] = max(y_lims[1], y_lim[1])

        # impose the same scale on the x and y axes for each subplot (i.e. each coherence)
        for ax in axs:
            ax.set_xlim(*x_lims)
            ax.set_ylim(*y_lims)

        plt.savefig(
            os.path.join(fig_dir, 'figure1_raw', filtering_method, param_name + '_distr.png')
        )

    plt.close('all')

def figure2_raw(kernels, filtering_method='unfiltered'):
    # compute the average and std of kernel modulus over agents:
    # means[idx_coh] = average most probable kernel modulus over agents
    means = []
    stds = []
    for tab in kernels:
        means.append(np.mean(tab, axis=0))
        stds.append(np.std(tab, axis=0))

    # plot
    # times in sec
    x = np.linspace(0, 1, 300) * 299 * 8.33 * 1e-3
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

    plt.savefig(os.path.join(fig_dir, 'figure2_raw', filtering_method, 'kernel.png'))

def figure3_raw(
    conv_popts,
    filtering_method='unfiltered',
    with_log: bool = True
):
    """Figure 3 (visualize parameters with respect to each other):
    - one panel per coherence
    - one subplot per parameter couple
    - value of parameter 1 as x axis
    - value of parameter 2 as y axis, 1 point per coh per agent 
    - each coherence is represented by a color
    """
    # exclude V0, V1 from the parameters
    param_names = Fit_param3().param_names[:-2]
    n_params = len(param_names)
    fontsize = 22

    for idx_coh, coh in enumerate(cohs):
        fig, axs = plt.subplots(
            n_params, n_params, figsize=(16, 16),
            constrained_layout=True
        )
        title = title_filter(f"coh = {coh:.2f}", filtering_method=filtering_method)
        fig.suptitle(title, fontsize=fontsize)

        # set the labels and tick params
        for i in range(n_params):
            param_name = convert_name(param_names[i], with_log=with_log)
            axs[i, 0].set_ylabel(param_name, fontsize=fontsize)
            axs[-1, i].set_xlabel(param_name, fontsize=fontsize)

        for i in range(n_params):
            for j in range(i):
                axs[i, j].tick_params(labelsize=fontsize)

            # clear the unused axes
            for j in range(i, n_params):
                axs[i, j].axis('off')

        # plot
        for i in range(n_params):
            for j in range(i):
                axs[i, j].plot(
                    conv_popts[idx_coh][:, j],
                    conv_popts[idx_coh][:, i],
                    '.', color='blue'
                )

        plt.savefig(
            os.path.join(fig_dir, 'figure3_raw', filtering_method, 'coh_' + f'{coh:.2f}.png'[2:])
        )
    plt.close('all')

def figure4_raw(
    corrmat,
    filtering_method='unfiltered',
    with_log: bool = True
):
    """Figure 4 (visualize parameters with respect to each other):
    - one panel per coherence
    - correlation matrix btw parameters, sampled over agents
    """
    param_names = Fit_param3().param_names
    conv_names = [convert_name(name, with_units=False, with_log=with_log) for name in param_names]
    n_params = len(param_names)
    fontsize = 14

    for idx_coh, coh in enumerate(cohs):
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        title = title_filter(f"coh = {coh:.2f}", filtering_method)

        ax.set_title(title, fontsize=fontsize)
        ax.set_xticks(ticks=range(n_params), labels=conv_names, rotation=45)
        ax.set_yticks(ticks=range(n_params), labels=conv_names)
        ax.tick_params(labelsize=fontsize)

        img = ax.imshow(corrmat[idx_coh], cmap='gnuplot2')
        fig.colorbar(img, ax=ax, fraction=0.05)

        # set the fontsize for the ticks of the colorbar
        fig.axes[1].tick_params(labelsize=fontsize)
        plt.savefig(
            os.path.join(fig_dir, 'figure4_raw', filtering_method, 'coh_' + f'{coh:.2f}.png'[2:])
        )
    plt.close('all')

def figure5_raw(
    conv_popts,
    filtering_method='unfiltered',
    with_log: bool = True
):
    """Computes the mean values of the fitted parameters
    converted to seconds and log scales.

    The average is perfomed over the agents, and the result is plotted
    as a function of the coherence.

    One figure per parameter is drawn.
    """
    # exclude V0, V1 from the plot
    param_names = Fit_param3().param_names[:-2]

    fontsize = 18
    for idx_param, param_name in enumerate(param_names):
        fig, ax = plt.subplots(
            1, 1, constrained_layout=True
        )
        title = title_filter(
            f"{convert_name(param_name, with_log=with_log)}",
            filtering_method
        )
        fig.suptitle(title, fontsize=fontsize)

        # set the ticks and axes titles
        ax.set_xlabel('coherence', fontsize=fontsize)
        ax.set_ylabel('mean value', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)

        # compute and plot
        y = np.zeros(len(cohs))
        z = np.zeros(len(cohs))
        for i, tab in enumerate(conv_popts):
            y[i] = np.mean(tab[:, idx_param])
            z[i] = np.std(tab[:, idx_param])
        
        ax.plot(cohs, y, '--')
        ax.fill_between(cohs, y - z / 2, y + z / 2, alpha=0.3)

        plt.savefig(
            os.path.join(fig_dir, 'figure5_raw', filtering_method, param_name + '_mean.png')
        )

    plt.close('all')


filtering_method = 'remove_after_tgt'
with_log = True
use_cache = True
agents = get_agents()
kernels = get_kernels(agents, use_cache=use_cache, filtering_method=filtering_method)
popts = get_popts(kernels, use_cache=use_cache, filtering_method=filtering_method)
conv_popts = convert_popts(popts, with_log=with_log)
corrmat = get_corrmat(conv_popts)

# figure1_raw(conv_popts, filtering_method=filtering_method, with_log=with_log)
# figure2_raw(kernels, filtering_method=filtering_method)
# figure3_raw(conv_popts, filtering_method=filtering_method, with_log=with_log)
# figure4_raw(corrmat, filtering_method=filtering_method, with_log=with_log)
# figure5_raw(conv_popts, filtering_method=filtering_method, with_log=with_log)
