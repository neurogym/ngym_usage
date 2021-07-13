
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 21:25:18 2020

@author: gryang
"""
from pathlib import Path
from pynwb import NWBHDF5IO
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
import seaborn as sns
import pandas as pd

import sys
sys.path.append('..')  # TODO: Temporary hack

sns.set(font_scale=0.6)
sns.set_palette("deep")


from analysis.preprocess import get_conditions
import analysis.singleunit as su
from analysis.decomposition.dpca import dPCA


if __name__ == '__main__':
    dataset = 'allenbrain'  # Doesn't work for some reason
    # dataset = 'chandravadia19'
    dataset = 'steinmetz19'
    # dataset = 'yu19'
    
    # rootpath = Path(dataset)
    rootpath = Path('files') / 'Steinmetz19-v0'
    nwbfiles = [d for d in rootpath.iterdir() if d.suffix == '.nwb']
    filepath = nwbfiles[0]
    figpath = Path('figures')
    
    print('Loading NWB file from ', str(filepath)) 
    io = NWBHDF5IO(str(filepath), 'r')
    file = io.read()

    conditions = get_conditions(file)
    # explain_nwb_file(file)
    # example_trial_raster(file, i_trial=28, n_neurons=500)
    # example_neuron_raster(file, i_neuron=1, n_trials=100, sort_cond='visual_stimulus_time')
    # example_neuron_raster(file, i_neuron=1, align='visual_stimulus_time', sort_cond='go_cue')
    # example_neuron_raster(file, i_neuron=0, n_trials=100, align='visual_stimulus_time')
    # example_neuron_rate(file, i_neuron=0, n_trials=1000)
    # example_neuron_rate_bycondition(file, i_neuron=10, align='start_time')
    # su.example_neuron_rate_bycondition(file, i_neuron=10, align='start_time')
    # example_neuron_rate_bycondition(file, i_neuron=1, align='end_stimulus')
    
    # io.close()
    
    from analysis.decomposition.nwb import run_plot_dpca, run_plot_jpca, run_plot_tda, run_dpca
    from analysis.preprocess import trial_avg_rate
    cond, align, cond_vals = 'choice', 'start_decision', None
    # cond, align, cond_vals = 'response_choice', 'response_time', [1, -1]
    # run_plot_dpca(file, cond=cond, align=align, cond_vals=cond_vals)
    # run_plot_jpca(file, cond=cond, align=align)
    # run_plot_tda(file, align=align)
    
    Z, times = run_dpca(file, cond, align, cond_vals=cond_vals)

    trials = file.trials
    trial_conds = trials[cond].data[:]
    if cond_vals is None:
        cond_vals = np.unique(trial_conds)
    n_cond = len(cond_vals)

    # Plotting results
    fig, axes = plt.subplots(3, 1, figsize=(2, 3), sharex=True)

    ax = axes[0]
    for s in range(n_cond):
        ax.plot(times, Z['t'][0, s])
    ax.set_title('1st time component')

    ax = axes[1]
    for s in range(n_cond):
        ax.plot(times, Z['s'][0, s], label=str(cond_vals[s]))
    # ax.legend(title=cond, bbox_to_anchor=(1, 1))
    ax.set_title('1st {:s} component'.format(cond))

    ax = axes[2]
    for s in range(n_cond):
        ax.plot(times, Z['st'][0, s], label=str(cond_vals[s]))
    # ax.legend(title=cond, bbox_to_anchor=(1, 1))
    ax.set_title('1st mixing component')
    ax.set_xlabel('Time from {:s} (s)'.format(align))

    plt.tight_layout()
    plt.savefig(figpath / 'dpca')
    