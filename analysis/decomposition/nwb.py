"""Running decomposition analysis from NWB files.

These files are supposed to be copied and modified.
"""

import numpy as np
import matplotlib.pyplot as plt

from .dpca import dPCA
from .jpca import jPCA
from .jpca_utils import plot_projections
from analysis.preprocess import get_trial_spikes_bytrials, get_rate, trial_avg_rate



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


def run_plot_jpca(file, cond, align='start_time'):
    """Run and plot jPCA.

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

    # Plot the projected data
    s = np.max(projected)*0.03
    plot_projections(projected, circle_size=s, arrow_size=s)

    # return plt
