"""Preprocessing for analysis."""

import numpy as np
from neo.core import SpikeTrain
from elephant.kernels import GaussianKernel
import quantities as qt

from analysis.statistics import myrate


def get_rate(spikes, t_start=0., t_stop=1., sampling_period=0.01):
    """Get rate for a list of spikes across trials.

    Args:
        spikes: a list of spike times, or a list of lists of spike times
        t_start: float, start time
        t_stop: float, stop time
        sampling_period: float, sampling period

    Return:
        rate: a list of firing rate
        times: the time points of the estimated rates
    """

    t_cut = 0.5
    t_start = t_start - t_cut
    t_stop = t_stop + t_cut
    rate, times = myrate(spikes, sampling_period=sampling_period,
                  kernel=GaussianKernel(0.1 * qt.s),
                  t_start=t_start, t_stop=t_stop)
    i_cut = int(t_cut / sampling_period)
    return rate[i_cut:-i_cut], times[i_cut:-i_cut]


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


def get_trial_spikes_bytrials(file, neurons=None, trial_inds=None,
                              align='start_time', t_offset=None):
    """Get a list of spike timing for a neuron across trials.

    Args:
        file: nwb file handle
        neurons: None or int or list of ints, optional unit indices
        trial_inds: None or list of ints, trial indices
        align: str, name of event to align to
        t_offset: None or tuple of time, offset to get time

    Returns:
        trial_spikes: a list of lists, each inner-list contains spikes in
            one trial, or a list of these for the neurons
    """

    trials = file.trials
    if trial_inds is not None:
        align_times = trials[align].data[trial_inds]
    else:
        align_times = trials[align].data[:]
    if t_offset is None:
        start_times = align_times - 1
        stop_times = align_times + 3
    else:
        start_times = align_times + t_offset[0]
        stop_times = align_times + t_offset[1]

    single_neuron = False
    if neurons is None:
        neurons = range(len(file.units.spike_times_index))
    if isinstance(neurons, int):
        neurons = [neurons]
        single_neuron = True

    all_spikes = []
    for i_neuron in neurons:
        neuron_spikes = file.units.spike_times_index[i_neuron]
        i_starts = np.searchsorted(neuron_spikes, start_times - 1.0)
        i_stops = np.searchsorted(neuron_spikes, stop_times + 1.0)
        trial_spikes = [neuron_spikes[i_starts[j]:i_stops[j]] - align_times[j] for j in range(len(align_times))]
        if single_neuron:
            return trial_spikes
        else:
            all_spikes.append(trial_spikes)
    return all_spikes


def trial_avg_rate(file, trial_inds_list, align='start_time'):
    """Get trial-averaged rate.

    Args:
        file: NWB file handle
        trial_inds_list: a list of lists. Each inner list contains trial inds
            to be averaged over
        align: str, even to align trials to

    Return:
        rate: np array (n_cond, n_time, n_neuron).
            Here n_cond = len(trial_inds_list)
    """
    t_start, t_stop = -1., 1.
    sampling_period = 0.01

    n_cond = len(trial_inds_list)
    n_time = int((t_stop - t_start) / sampling_period)
    n_neuron = len(file.units.spike_times_index)

    R = np.zeros((n_cond, n_time, n_neuron))

    # Loop through all condition values of the task condition
    for j in range(n_cond):
        trial_inds = trial_inds_list[j]
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
        R[j] = rate / n_trials

    return R, times