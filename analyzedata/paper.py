#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for generating nice figures for paper
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

from analysis.decomposition.dpca import dPCA
from analysis.decomposition.nwb import run_plot_dpca, run_plot_jpca, run_plot_tda, run_dpca

if __name__ == '__main__':
    figpath = Path('figures')    
    for source in ['data', 'model']:
        if source == 'data':
            cond, align, cond_vals = 'response_choice', 'response_time', [1, -1]
            rootpath = Path('steinmetz19')
            
        elif source == 'model':
            cond, align, cond_vals = 'choice', 'start_decision', [2, 1]
            rootpath = Path('files') / 'Steinmetz19-v0'
        else:
            raise ValueError
    
        nwbfiles = [d for d in rootpath.iterdir() if d.suffix == '.nwb']
        filepath = nwbfiles[0]
        print('Loading NWB file from ', str(filepath)) 
        io = NWBHDF5IO(str(filepath), 'r')
        file = io.read()
    
        Z, times = run_dpca(file, cond, align, cond_vals=cond_vals)
        
        trials = file.trials
        trial_conds = trials[cond].data[:]
        if cond_vals is None:
            cond_vals = np.unique(trial_conds)
        n_cond = len(cond_vals)
    
        io.close()
    
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
        plt.savefig(figpath / (source + '_dpca.pdf'))