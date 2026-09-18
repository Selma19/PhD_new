"""Draws the figures using the old data stored in the external storage.
Also compares the raw kernels computed from the h5py data with the
ones computed from the external storage data.

The root of this storage is:
root = /Volumes/Selma_PhD/PhD/Backup/CPR_many_files/Felix_project/Data/Solo_data/CPR_psychophysics

to access the raw kernel of the agent abc and coherence coh, read the file located at:
load_path = $root/abc/Analysis/Kernels/Basic/coh/train_and_save_kernel_size_300.txt

this kernel is loaded with:
kernel = np.loadtxt(load_path).view(complex)
"""

from typing import List, Literal
from sqlite3 import ProgrammingError
import json, os, pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from blume.table import table as tb

from database.kernel.fill_db._fill_db import Fit_param3
from database import Kernel_db
from database.stimulus.fill_db.from_folders import get_agents as get_agents_in_db

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

def build_dirs():
    """Creates the directories which will store
    the figures.

    Assumes the directory `figures` has already been created.
    """
    methods = ['unfiltered', 'remove_after_tgt']
    from_dbs = [True, False]
    n_dirs = 6
    base_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'figures'
    )
    
    for i in range(1, n_dirs + 1):
        for method in methods:
            for from_db in from_dbs:
                path = os.path.join(
                    base_path, f'figure{i}_raw',
                    f'{method}{from_db}'
                )
                try:
                    os.mkdir(path)
                except FileExistsError:
                    pass

def get_agents(use_cache=True, **kwargs):
    cachePath = os.path.join(fig_dir, 'caches', 'agents')

    if use_cache:
        res = pickle.load(open(cachePath, '+rb'))
        tab = [
            'roh', 'ang', 'yam', 'nie', 'sol', 'swp', 'tak', 'mls',
           'anb', 'gac', 'rit', 'seo', 'rab', 'mod'
        ]
        return [agent for agent in res if agent not in tab]

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
    return np.loadtxt(load_path).view(complex) * coh

def load_kernel_from_db(
        agent: str,
        coh: float,
        filtering_method: Literal["unfiltered","remove_after_tgt"]
):
    db = Kernel_db()
    db.connect()

    try:
        main_key = db.cur.execute("""
            SELECT id
            FROM Main
            WHERE
                agent = ?
                AND coh = ?
                AND filtering_method = ?
        """, (agent, coh, filtering_method)).fetchone()

        k_out = db.cur.execute("""
            SELECT kernel_output
            FROM Kernels
            WHERE
                main_key = ?
                AND kernel_type = 'raw'
                AND kernel_method = 'linear_reg'
        """, main_key).fetchone()[0]

    except ProgrammingError:
        db.close()
        return -np.ones(300, dtype=np.complex128)

    db.close()

    res = json.loads(k_out)
    return np.array(res[0]) + 1j * np.array(res[1])

def suffix_cache(filtering_method, from_db):
    suffixCache = ''
    if filtering_method == 'remove_after_tgt':
        suffixCache = '_no_tgt'
    if from_db:
        suffixCache += '_from_db'
    return suffixCache

def get_kernels(
    agents=None,
    use_cache=True,
    filtering_method: Literal['unfiltered', 'remove_after_tgt']='unfiltered',
    from_db: bool = False,
    **kwargs
):
    """Returns kernels[idx_coh][idx_agent, :] = 300 kernel
    values for agent and coh.
    """
    if from_db:
        load_kernel = lambda agent, coh:\
        load_kernel_from_db(agent, coh, filtering_method)

    elif filtering_method == 'unfiltered':
        load_kernel = load_kernel_unfiltered

    elif filtering_method == 'remove_after_tgt':
        raise ValueError(
            "`from_db` must be `True`" \
            "for removing the target"
        )

    cachePath = os.path.join(
        fig_dir, 'caches',
        'kernels' + suffix_cache(filtering_method, from_db)
    )

    if use_cache:
        return pickle.load(
            open(cachePath, '+rb')
        )

    else:
        # get the kernels modulus
        kernels = [
            np.zeros( (len(agents), 300), dtype=np.complex128 )
            for _ in range(len(cohs))
        ]
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
    filtering_method: Literal['unfiltered', 'remove_after_tgt']='unfiltered',
    from_db: bool = False,
    **kwargs
):
    """Computes the optimal parameters of Fit_param3
    for the raw kernels of each agent and coherence.

    Then, stores these parameters in a cache, as a list of numpy.ndarray:

    popts[idx_coh][idx_agent, idx_param] = optimal values of the curve_fit parameter
    of index idx_param in Fit_param3().param_names

    `filtering_method` and `from_db` are only used to compute the cache path.
    """
    cachePath = os.path.join(
        fig_dir, 'caches', 'popts' + suffix_cache(filtering_method, from_db)
    )

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

def convert_popts(popts: List[np.ndarray], with_log: bool=False, **kwargs):
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

def params_as_title(p_names, p_vals):
    title = ""
    count = 0
    for p_name, p_val in zip(p_names, p_vals):
        if p_val >= 0.1 and p_val <= 10:
            s = f"{p_val:.2f}"
        else:
            s = f"{p_val:.2e}"
        if len(title) - count < 36:
            title += f"{p_name} = {s}, "
        else:
            title = title[:-2] + f"\n{p_name} = {s}, "
            count = len(title)
    return title

def check_popts(agents, kernels, popts, conv_popts, **kwargs):
    """Checks the parameters returned by get_popts recover the correct kernel.

    For each agent and coherence, plots on a separate figure
    the raw kernel together with its fit (Param3 model tuned with curve_fit),
    as well as the values of the fit parameters.
    """
    fit = Fit_param3()
    fit.method = 'curve_fit'
    conv_names = [convert_name(name, with_units=False, with_log=False) for name in fit.param_names[:-2]]

    for idx_agent, agent in enumerate(agents):
        for idx_coh, coh in enumerate(cohs):
            fit.out = {name: val for name, val in zip(fit.param_names, popts[idx_coh][idx_agent])}
            fit_ker = fit.read_kernel_from_out()
            ker = np.abs(kernels[idx_coh][idx_agent, :])
            _, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(ker, '.')
            ax.plot(fit_ker, '-')
            fontsize = 14
            title = f"coh = {coh:.2f}, agent {agent}\n"
            title += params_as_title(conv_names, conv_popts[idx_coh][idx_agent, :-2])
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

def title_filter(title, filtering_method, **kwargs):
    """Appends the suffix to title that corresponds to filtering_method.
    """
    if filtering_method == 'unfiltered':
        title += "; no filter"

    else:
        title += "; target influence removed"
    return title

def convert_param(name: str, values: List[float], with_log: bool=False, **kwargs):
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
    with_log: bool = False, **kwargs
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
    filtering_method: Literal['unfiltered', 'remove_after_tgt'],
    from_db: bool,
    with_log: bool = True,
    smooth_hist: bool = False,
    **kwargs
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

            if smooth_hist:
                x = tab[:, idx_param]
                density = stats.gaussian_kde(x)
                x = np.linspace(np.min(x), np.max(x), 20)
                # ax.fill_between(x, density(x), color='blue', alpha=0.3)
                ax.plot(x, density(x), color='blue')

            else:
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

        if smooth_hist:
            savePath = os.path.join(
                fig_dir, 'figure1_raw', filtering_method + str(from_db),
                param_name + '_distr_smooth.png'
            )
        else:
            savePath = os.path.join(
                fig_dir, 'figure1_raw', filtering_method + str(from_db),
                param_name + '_distr.png'
            )
        plt.savefig(savePath)

    plt.close('all')

def figure2_raw(
    kernels,
    filtering_method: Literal['unfiltered', 'remove_after_tgt'],
    from_db: bool,
    **kwargs):
    """Plots the modulus of the raw kernels, averaged
    over all agents, for each coherence on the same figure.
    The standard deviation of the modulus, computed over
    agents, is also shown.
    """
    # compute the average and std of kernel modulus over agents:
    # means[idx_coh] = average most probable kernel modulus over agents
    means = []
    stds = []
    for tab in kernels:
        means.append(np.abs(np.mean(tab, axis=0)))
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
    savepath = os.path.join(
        fig_dir, 'figure2_raw',
        filtering_method + str(from_db), 'kernel.png'
    )
    plt.savefig(savepath)

def figure3_raw(
    conv_popts,
    filtering_method: Literal['unfiltered', 'remove_after_tgt'],
    from_db: bool,
    with_log: bool = True,
    **kwargs
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
            os.path.join(
                fig_dir, 'figure3_raw', filtering_method + str(from_db),
                'coh_' + f'{coh:.2f}.png'[2:])
        )
    plt.close('all')

def figure4_raw(
    corrmat,
    filtering_method: Literal['unfiltered', 'remove_after_tgt'],
    from_db: bool,
    with_log: bool = True,
    **kwargs
):
    """Figure 4 (visualize parameters with respect to each other):
    - one panel per coherence
    - correlation matrix btw parameters, sampled over agents
    """
    param_names = Fit_param3().param_names
    conv_names = [
        convert_name(name, with_units=False, with_log=with_log)
        for name in param_names
    ]
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
            os.path.join(
                fig_dir, 'figure4_raw', filtering_method + str(from_db),
                'coh_' + f'{coh:.2f}.png'[2:]
            )
        )
    plt.close('all')

def figure5_raw(
    conv_popts,
    filtering_method: Literal['unfiltered', 'remove_after_tgt'],
    from_db: bool,
    with_log: bool = True, **kwargs
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
            os.path.join(
                fig_dir, 'figure5_raw',
                filtering_method + str(from_db), param_name + '_mean.png'
            )
        )

    plt.close('all')

def figure6_raw(
    agents: List[str],
    kernels: List[np.ndarray],
    popts: List[np.ndarray],
    conv_popts: List[np.ndarray],
    filtering_method: Literal['unfiltered', 'remove_after_tgt'],
    from_db: bool,
    **kwargs
):
    """Plots the raw kernels together with their fits.

    Combination of check_popts and figure2_raw:
    - one panel per agent
    - one figure per panel, all coherences colored as in figure2_raw

    Also creates a table per agent, gathering the parameter values
    for each coherence.
    """
    # there are 6 cohs, one color per coh
    colors = ['b', 'k', 'r', 'green', 'purple', 'cyan', 'pink']

    fit = Fit_param3()
    fit.method = 'curve_fit'

    fontsize = 14

    X = np.linspace(0, 1, 300) * 299 * 8.33 * 1e-3
    xlabel = r'$t$' + ' (sec)'
    ylabel = r'$|k(t)|$'
    
    for idx_agent, agent in enumerate(agents):
        # first create the table
        # one row per coherence, and each column for a distinct parameter
        fig, ax = plt.subplots(1, 1)
        ax.set_axis_off()
        colLabels = [
            'coherence', 'tau1 (sec)', 'tau2 (sec)', 'alpha', 'd (sec)', 'A'
        ]
        cellText = []

        for idx_coh, coh in enumerate(cohs):
            row = [f"{coh:.2f}"]
            for idx_param in range(5):
                val = conv_popts[idx_coh][idx_agent, idx_param]
                if val >= 1 and val < 10:
                    row.append(f"{val:.2f}")
                else:
                    row.append(f"{val:.2e}")
            cellText.append(row)

        tb(ax, cellText=cellText, cellLoc='center', colLabels=colLabels, loc='center')
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        plt.savefig(
            os.path.join(fig_dir,
                'figure6_raw',
                filtering_method + str(from_db),
                agent + '_params.png')
        )
        plt.close()

        # second create the plot
        _, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(8, 8))

        # set the title and axes labels
        title = f"agent {agent}"
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_ylim(0, 0.145)
        
        for idx_coh, (coh, color) in enumerate(zip(cohs, colors)):
            label = f"coh = {coh:.2f}"

            fit.out = {name: val for name, val in zip(fit.param_names, popts[idx_coh][idx_agent])}
            fit_ker = fit.read_kernel_from_out()
            ker = np.abs(kernels[idx_coh][idx_agent, :])

            ax.plot(X[1:-1], ker[1:-1], '.', label=label, color=color)
            ax.plot(X[1:-1], fit_ker[1:-1], '-', color=color)
        ax.legend(fontsize=fontsize)

        plt.savefig(
            os.path.join(
                fig_dir,
                'figure6_raw',
                filtering_method + str(from_db),
                agent + '.png'
            )
        )
        plt.close()

def get_figures():
    """Generates and saves all figures.
    """
    agents = get_agents()
    for filtering_method in ['unfiltered', 'remove_after_tgt']:
        kwargs = {
            'use_cache': False,
            'filtering_method': filtering_method,
            'with_log': False,
            'from_db': True
        }
        kernels = get_kernels(agents, **kwargs)
        popts = get_popts(kernels, **kwargs)
        conv_popts = convert_popts(popts, **kwargs)
        corrmat = get_corrmat(conv_popts)

        figure1_raw(conv_popts, smooth_hist=True, **kwargs)
        figure2_raw(kernels, **kwargs)
        figure3_raw(conv_popts, **kwargs)
        figure4_raw(corrmat, **kwargs)
        figure5_raw(conv_popts, **kwargs)
        figure6_raw(agents, kernels, popts, conv_popts, **kwargs)

    # check_popts(agents, kernels, popts, conv_popts)

def compare_db_with_former_raw_kernels():
    """On a separate figure for each agent and coherence,
    plots the raw kernel from the db with the raw kernel
    from the former storage on the same figure.

    To ease comparison, both kernels are normalized.
    """
    agents = get_agents()

    # load the db kernels
    db_kernels = get_kernels(
        agents, use_cache=False, from_db=True,
        filtering_method='unfiltered'
    )

    # load the former kernels
    former_kernels = get_kernels(
        agents, use_cache=True, from_db=False
    )

    # plot
    # times in sec
    x = np.linspace(0, 1, 300) * 299 * 8.33 * 1e-3
    fontsize = 14
    ylabel = r'$|k(t)|$'
    xlabel = r'$t$' + ' (sec)'

    for idx_agent, agent in enumerate(agents):
        for idx_coh, coh in enumerate(cohs):
            _, ax = plt.subplots(1, 1, constrained_layout=True)
            title = f"agent {agent}, coh = {coh:.2f}"
            ax.set_title(title, fontsize=fontsize)
            ax.set_ylabel(ylabel, fontsize=fontsize)
            ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.tick_params(labelsize=fontsize - 1)

            y = np.abs(db_kernels[idx_coh][idx_agent, :])
            y /= np.sqrt(np.sum(y ** 2))
            ax.plot(x, y, '.', label='db kernel')

            y = np.abs(former_kernels[idx_coh][idx_agent, :])
            y /= np.sqrt(np.sum(y ** 2))
            ax.plot(x, y, '.', label='former kernel')

            ax.legend(fontsize=fontsize)
            plt.show()

def solve_shift_problem():
    """Same as `figure2_raw`
    but instead of plotting the average of the modulus,
    we plot the modulus of the average kernels.
    """
    # redefine the functions so that we consider
    # the complex kernels instead of their modulus
    def load_kernel_unfiltered(agent: str, coh: float):
        """Returns the raw kernel associated to agent, coh,
        when no filter has been applied to the sensory data
        (dot complex direction).
        """
        load_path = os.path.join(
            rootDir, agent, 'Analysis',
            'Kernels', 'Basic', str( int(coh * 1000) ),
            'train_and_save_kernel_size_300.txt'
        )
        return np.loadtxt(load_path).view(complex) * coh

    def get_kernels(agents):
        """Returns kernels[idx_coh][idx_agent, :] = 300 kernel
        complex values for agent and coh.
        """
        # get the complex raw kernels
        kernels = [
            np.zeros( (len(agents), 300), dtype=np.complex128 )
            for _ in range(len(cohs))
        ]

        for idx_coh, coh in enumerate(cohs):
            print(f"{len(cohs) - idx_coh} cohs remaining")
            for i, agent in enumerate(agents):
                kernels[idx_coh][i, :] = load_kernel_unfiltered(agent, coh)

        return kernels

    def figure2_raw(kernels):
        """Plots the modulus of the average raw kernel
        over all agents, for each coherence on the same figure.
        The standard deviation of the kernels, computed over
        agents, is also shown.
        """
        # compute the average and std of kernels over agents,
        # and after take the modulus:
        # means[idx_coh] = modulus of the average
        # raw kernel over agents
        means = []
        #stds = []
        for tab in kernels:
            means.append(np.abs(np.mean(tab, axis=0)))
            #stds.append(np.std(tab, axis=0))

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

        # for coh, color, y, std in zip(cohs, colors, clean_means, stds):
        for coh, color, y in zip(cohs, colors, clean_means):
            label = f"coh = {coh:.2f}"
            indices = (y < 250).nonzero()
            X = x[indices]
            Y = y[indices]
            #Z = std[indices]
            ax.plot(X, Y, '-', color=color, label=label)
            #ax.fill_between(X, Y - Z / 2, Y + Z / 2, color=color, alpha=0.1)

        ax.legend(fontsize=fontsize)
        plt.show()
        #plt.savefig(
        # os.path.join(
        #   fig_dir, 'figure2_raw', filtering_method, 'kernel.png'
        # ))

    # plot the new figure
    agents = get_agents()
    kernels = get_kernels(agents)
    figure2_raw(kernels)

def compare_db_raw_with_truncated_kernels(
    normalize: bool
):
    """On a separate figure for each agent and coherence,
    plots the raw kernel from the db with the truncated 
    kernel from the db.

    To ease comparison, both kernels can be normalized.
    """
    if normalize:
        func = lambda y: np.abs(y) / np.sqrt(np.sum(y ** 2))
    else:
        func = np.abs

    agents = get_agents()

    # load the db kernels
    db_kernels = get_kernels(
        agents, use_cache=False, from_db=True,
        filtering_method='unfiltered'
    )

    # load the db trunckated kernels
    db_trunc_kernels = get_kernels(
        agents, use_cache=False, from_db=True,
        filtering_method='remove_after_tgt'
    )

    # plot
    # times in sec
    x = np.linspace(0, 1, 300) * 299 * 8.33 * 1e-3
    fontsize = 14
    ylabel = r'$|k(t)|$'
    xlabel = r'$t$' + ' (sec)'

    for idx_agent, agent in enumerate(agents):
        for idx_coh, coh in enumerate(cohs):
            _, ax = plt.subplots(1, 1, constrained_layout=True)
            title = f"agent {agent}, coh = {coh:.2f}"
            ax.set_title(title, fontsize=fontsize)
            ax.set_ylabel(ylabel, fontsize=fontsize)
            ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.tick_params(labelsize=fontsize - 1)

            y = db_kernels[idx_coh][idx_agent, :]
            ax.plot(x, func(y), '.', label='raw kernel')

            y = db_trunc_kernels[idx_coh][idx_agent, :]
            ax.plot(x, func(y), '.', label='truncated kernel')

            ax.legend(fontsize=fontsize)
            plt.show()

exit()

agents = get_agents()
filtering_method = 'remove_after_tgt'
from_db = True
db_kernels = get_kernels(
    agents, use_cache=True,
    filtering_method=filtering_method,
    from_db=from_db
)
popts = get_popts(
    db_kernels,
    use_cache=True,
    filtering_method=filtering_method,
    from_db=from_db
)
conv_popts = convert_popts(
    popts, with_log=False
)

figure6_raw(
    agents, kernels=db_kernels,
    popts=popts, conv_popts=conv_popts,
    filtering_method=filtering_method,
    from_db=from_db
)
