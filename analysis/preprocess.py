"""Preprocessing for analysis."""

import numpy as np
from sklearn.neighbors import KernelDensity


def get_rate(spikes, times=None, bandwidth=0.05, kernel='gaussian'):
    """Get rate for a list of spikes across trials.

    Args:
        spikes: a list of spike times, or a list of lists of spike times
        times: a list of time points, must be uniformly spaced

    Return:
        rate: a list of firing rate
        times: the time points of the estimated rates
    """
    try:
        all_spikes = np.concatenate(spikes)
        n_spike_list = len(spikes)
    except ValueError:
        all_spikes = spikes
        n_spike_list = 1

    if times is None:
        dt = 0.02
        start = (np.min(all_spikes) + 0.3) // dt
        stop = (np.max(all_spikes) - 0.3) // dt
        times = np.arange(start, stop) * dt
    else:
        dt = times[1] - times[0]

    times = np.array(times)
    bins = np.array(times) - dt / 2
    bins = np.append(bins, bins[-1] + dt)

    if len(all_spikes) == 0:
        rate = np.zeros_like(times)
    else:
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        kde = kde.fit(all_spikes[:, np.newaxis])
        rate = np.exp(kde.score_samples(times[:, np.newaxis])) * len(all_spikes)

    # For debugging, plain histogram
    # rate, bins = np.histogram(all_spikes, bins=bins)
    # rate = rate/dt

    times = (bins[:-1] + bins[1:]) / 2
    rate = rate / n_spike_list
    return rate, times


def get_eventnames(file):
    """Return a list of sorted temporal event names in trials."""
    trials = file.intervals['trials']
    event_names = []
    for name in trials.colnames:
        t = trials[name].data[-1]
        try:
            if trials['start_time'][-1] <= t <= trials['stop_time'][-1]:
                event_names.append(name)
        except:
            pass
    ts = [trials[name].data[-1] for name in event_names]
    event_names = [name for _, name in sorted(zip(ts, event_names))]
    return event_names


def get_conditions(file, max_unique=5):
    """Get all conditions that have a small number of unique values."""
    conditions = list()
    for cond in file.trials.colnames:
        n_unique = len(np.unique(file.trials[cond].data[:]))
        if 1 < n_unique < max_unique:
            conditions.append(cond)
    return conditions


def argsort_trials(file, sort_cond=None, trial_inds=None, align='start_time'):
    n_trials = file.trials['start_time'].data.shape[0]
    if trial_inds is None:
        trial_inds = np.arange(n_trials)

    if sort_cond is None:
        return trial_inds

    trial_conds = file.trials[sort_cond].data[trial_inds]

    # If condition is the time of an event, need to align
    if sort_cond in get_eventnames(file):
        trial_conds = trial_conds - file.trials[align].data[trial_inds]

    ind_sort = np.argsort(trial_conds, kind='stable')
    trial_inds = trial_inds[ind_sort]
    return trial_inds


def get_trial_spikes_bytrials(file, i_neuron, trial_inds, align='start_time',
                              t_offset=None):
    """Get a list of spike timing for a neuron across trials.

    Args:
        file: nwb file handle
        i_neuron: int, unit index
        trial_inds: list of ints, trial indices
        align: str, name of event to align to
        t_offset: None or tuple of time, offset to get time
    """

    trials = file.trials
    trial_spikes = []  # spike times of different trials
    neuron_spikes = file.units.spike_times_index[i_neuron]
    align_times = trials[align].data[:]
    for i_trial in trial_inds:
        align_time = align_times[i_trial]

        if t_offset is None:
            start_time = align_time - 1
            stop_time = align_time + 3
        else:
            start_time = align_time + t_offset[0]
            stop_time = align_time + t_offset[1]

        i_start = np.searchsorted(neuron_spikes, start_time - 1.0)
        i_stop = np.searchsorted(neuron_spikes, stop_time + 1.0)
        trial_spikes.append(neuron_spikes[i_start:i_stop] - align_time)
    return trial_spikes
