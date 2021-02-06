"""Running decomposition analysis from NWB files.

These files are supposed to be copied and modified.
"""

import numpy as np
import matplotlib.pyplot as plt

from .dpca import dPCA
from .jpca import jPCA
from .jpca_utils import plot_projections
from analysis.preprocess import trial_avg_rate
from .tda import TDA
import analysis.decomposition.tda_plots as tda_plots


def run_dpca(file, cond, align='start_time'):
    """Run and plot dPCA.

    Args:
        file: NWB file handle
        cond: str, the task condition to demixing
        align: str, event to align trials to
    """
    trials = file.trials
    trial_conds = trials[cond].data[:]

    cond_vals = np.unique(trial_conds)
    n_neuron = len(file.units.spike_times_index)

    # Loop through all condition values of the task condition
    trial_inds_list = [np.where(trial_conds == c)[0] for c in cond_vals]
    R, times = trial_avg_rate(file, trial_inds_list, align=align)

    # reshape and center data
    R = np.moveaxis(R, -1, 0)  # reshape to (n_neuron, n_cond, n_time)
    R -= np.mean(R.reshape((n_neuron, -1)), 1)[:, None, None]
    # R = np.random.randn(n_neuron, n_cond, n_time)

    dpca = dPCA(labels='st')
    dpca.protect = ['t']

    Z = dpca.fit_transform(R)
    return Z, times


def run_plot_dpca(file, cond, align='start_time'):
    """Run and plot dPCA.

    This code is meant to be adapted and modified by the user. Now it only
    supports dPCA using the time axis and an additional task condition.

    Args:
        file: NWB file handle
        cond: str, the task condition to demixing
        align: str, event to align trials to
    """

    Z, times = run_dpca(file, cond, align)

    trials = file.trials
    trial_conds = trials[cond].data[:]
    cond_vals = np.unique(trial_conds)
    n_cond = len(cond_vals)

    # Plotting results
    fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

    ax = axes[0]
    for s in range(n_cond):
        ax.plot(times, Z['t'][0, s])
    ax.set_title('1st time component')

    ax = axes[1]
    for s in range(n_cond):
        ax.plot(times, Z['s'][0, s], label=str(cond_vals[s]))
    ax.legend(title=cond, bbox_to_anchor=(1, 1))
    ax.set_title('1st {:s} component'.format(cond))

    ax = axes[2]
    for s in range(n_cond):
        ax.plot(times, Z['st'][0, s], label=str(cond_vals[s]))
    ax.legend(title=cond, bbox_to_anchor=(1, 1))
    ax.set_title('1st mixing component')
    ax.set_xlabel('Time from {:s} (s)'.format(align))

    plt.tight_layout()

    return plt


def run_jpca(file, cond, align='start_time'):
    """Run jPCA.

    Args:
        file: NWB file handle
        cond: str, the task condition to demixing
        align: str, event to align trials to
    """
    trials = file.trials
    trial_conds = trials[cond].data[:]
    cond_vals = np.unique(trial_conds)

    # Loop through all condition values of the task condition
    trial_inds_list = [np.where(trial_conds == c)[0] for c in cond_vals]
    R, times = trial_avg_rate(file, trial_inds_list, align=align)

    datas = [x for x in R]

    jpca = jPCA(num_jpcs=2)
    # Fit the jPCA object to data
    results = jpca.fit(datas, times=list(times), tstart=times[
        0], tend=times[-1])
    projected, full_data_var, pca_var_capt, jpca_var_capt = results
    return projected, full_data_var, pca_var_capt, jpca_var_capt


def run_plot_jpca(file, cond, align='start_time'):
    """Run and plot jPCA.

    Args:
        file: NWB file handle
        cond: str, the task condition to demixing
        align: str, event to align trials to
    """
    results = run_jpca(file, cond, align=align)
    projected = results[0]

    # Plot the projected data
    s = np.max(projected)*0.03
    plot_projections(projected, circle_size=s, arrow_size=s)


def run_plot_tda(file, align='start_time'):
    """Run and plot Tensor-decomposition-analysis."""
    trials = file.trials
    n_trials = len(trials)

    # Loop through all condition values of the task condition
    trial_inds_list = [[i] for i in range(n_trials)]
    data, times = trial_avg_rate(file, trial_inds_list, align=align)

    # Fit an tda of models, 4 random replicates / optimization runs per model rank
    tda = TDA(fit_method="ncp_hals")
    tda.fit(data, ranks=range(1, 9), replicates=4)

    fig, axes = plt.subplots(1, 2)
    # plot reconstruction error as a function of num components.
    tda_plots.plot_objective(tda, ax=axes[0])
    # plot model similarity as a function of num components.
    tda_plots.plot_similarity(tda, ax=axes[1])
    fig.tight_layout()

    # Plot the low-d factors for an example model, e.g. rank-2, first optimization run / replicate.
    num_components = 2
    replicate = 0
    tda_plots.plot_factors(
        tda.factors(num_components)[replicate])  # plot the low-d factors

    plt.show()
