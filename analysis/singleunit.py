import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .preprocess import (get_eventnames, argsort_trials,
                         get_trial_spikes_bytrials, get_conditions, get_rate)


def example_trial_raster(file, i_trial=0, n_neurons=50):
    """Spike raster for an example trial across neurons."""
    # Plot raster
    trials = file.trials
    start_time = trials['start_time'].data[i_trial]
    stop_time = trials['stop_time'].data[i_trial]
    event_names = get_eventnames(file)
    n_neurons_all = len(file.units.spike_times_index)

    trial_spikes = []  # spike times of different units
    for i_neuron in range(n_neurons_all):
        ind = file.units.spike_times_index[i_neuron]
        i_start = np.searchsorted(ind, start_time-1.0)
        i_stop = np.searchsorted(ind, stop_time+1.0)
        trial_spikes.append(ind[i_start:i_stop] - start_time)

    n_neurons = min(n_neurons, n_neurons_all)

    cmap = mpl.cm.get_cmap('Dark2')
    plt.figure(figsize=(5, 5))
    _ = plt.eventplot(trial_spikes[:n_neurons])
    for j, event in enumerate(event_names):
        t = file.intervals['trials'][event].data[i_trial] - start_time
        plt.plot([t, t], [0.5, n_neurons+0.5], color=cmap(j))
        plt.text(t, n_neurons*(1+j*0.05), event, ha='center', color=cmap(j))
    plt.xlabel('Time (s)')
    plt.ylabel('Neurons')
    return plt


def example_neuron_raster(file, i_neuron=0, n_trials=100, align='start_time',
                          sort_cond=None):
    """Spike raster for an example neuron across trials."""
    # Plot raster
    trials = file.trials

    n_trials_all = len(trials['start_time'].data)
    n_trials = min(n_trials, n_trials_all)
    trial_inds = np.arange(n_trials)
    trial_inds = argsort_trials(
        file, sort_cond, trial_inds=trial_inds, align=align)

    trial_spikes = get_trial_spikes_bytrials(
        file, neurons=i_neuron, trial_inds=trial_inds, align=align)

    cmap = mpl.cm.get_cmap('Dark2')
    plt.figure(figsize=(5, 5))
    _ = plt.eventplot(trial_spikes)

    event_names = get_eventnames(file)
    for j, event in enumerate(event_names):
        eventtimes = [[trials[event].data[i] - trials[align].data[i]] for i in
                      trial_inds]
        plt.eventplot(eventtimes, color=cmap(j))
        plt.text(eventtimes[-1][0], n_trials * (1 + j * 0.05), event,
                 ha='center', color=cmap(j))
    plt.xlabel('Time (s)')
    plt.ylabel('Trials')
    return plt


def example_neuron_rate(file, i_neuron=0, n_trials=100, align='start_time',
                        sort_cond=None):
    trials = file.trials

    n_trials_all = len(trials['start_time'].data)
    n_trials = min(n_trials, n_trials_all)
    trial_inds = np.arange(n_trials)
    trial_inds = argsort_trials(file, sort_cond, trial_inds=trial_inds, align=align)

    trial_spikes = get_trial_spikes_bytrials(
        file, neurons=i_neuron, trial_inds=trial_inds, align=align)

    rate, times = get_rate(trial_spikes)

    plt.figure()
    plt.plot(times, rate)
    plt.plot([0, 0], [0, np.max(rate ) *1.1])
    plt.text(0, np.max(rate ) *1.1, align)
    return plt


# TODO: Should set times to be the same for all trials
def example_neuron_rate_bycondition(
        file, cond=None, i_neuron=0, align='start_time'):
    if cond is None:
        conds = get_conditions(file)
        for cond in conds:
            if cond is not None:
                example_neuron_rate_bycondition(
                    file, cond, i_neuron=i_neuron, align=align)
        return

    trials = file.trials

    n_trials_all = len(trials['start_time'].data)
    trial_inds = np.arange(n_trials_all)

    cond_vals = np.unique(file.trials[cond].data[:])

    rate_list = list()
    times_list = list()
    for cond_val in cond_vals:
        trial_inds = np.where(trials[cond].data[:] == cond_val)[0]

        trial_spikes = get_trial_spikes_bytrials(
            file, neurons=i_neuron, trial_inds=trial_inds, align=align)

        rate, times = get_rate(trial_spikes)
        rate_list.append(rate)
        times_list.append(times)

    plt.figure()
    for i in range(len(cond_vals)):
        plt.plot(times_list[i], rate_list[i], label=cond_vals[i])
    plt.legend(title=cond)
    plt.ylabel('Rate (sp/s)')
    plt.xlabel('Time from {:s} (s)'.format(align))
    return plt