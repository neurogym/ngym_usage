"""Running decomposition analysis from NWB files."""

import numpy as np
import matplotlib.pyplot as plt

from .dpca import dPCA
from analysis.preprocess import get_trial_spikes_bytrials, get_rate


def run_plot_dpca(file, cond, align='start_time'):
    """Run and plot dPCA.

    This code is meant to be adapted and modified by the user. Now it only
    supports dPCA using the time axis and an additional task condition.

    Args:
        file: NWB file handle
        cond: str, the task condition to demixing
        align: str, event to align trials to
    """
    trials = file.trials
    trial_conds = trials[cond].data[:]

    cond_vals = np.unique(trial_conds)
    t_start, t_stop = -1., 1.
    sampling_period = 0.01

    n_neuron = len(file.units.spike_times_index)
    n_cond = len(cond_vals)
    n_time = int((t_stop - t_start) / sampling_period)
    R = np.zeros((n_neuron, n_cond, n_time))

    # Loop through all condition values of the task condition
    for j, cond_val in enumerate(cond_vals):
        trial_inds = np.where(trial_conds == cond_val)[0]
        trial_spikes = get_trial_spikes_bytrials(
            file, neurons=range(n_neuron), trial_inds=trial_inds, align=align)

        n_trials = np.zeros(n_neuron)
        for i_neuron in range(n_neuron):
            n_trials[i_neuron] = len(trial_spikes[i_neuron])  # num of trials
            trial_spikes[i_neuron] = np.concatenate(trial_spikes[i_neuron])

        # Get rate for all neurons (time, units)
        rate, times = get_rate(trial_spikes, t_start=t_start,
                               t_stop=t_stop,
                               sampling_period=sampling_period)
        rate = rate / n_trials
        R[:, j, :] = rate.T

    # center data
    R -= np.mean(R.reshape((n_neuron, -1)), 1)[:, None, None]

    # R = np.random.randn(n_neuron, n_cond, n_time)

    dpca = dPCA(labels='st')
    dpca.protect = ['t']

    Z = dpca.fit_transform(R)

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
