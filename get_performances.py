#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:32:02 2020

@author: molano
"""
import os
import sys
import glob
import ntpath
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(os.path.expanduser("~/neurogym"))
from neurogym.utils import plotting
# import matplotlib
# matplotlib.use('Agg')


def get_alg_task(file):
    spl = file.split('_')
    if spl[-1].find('.') == -1:
        alg = spl[0]
        task = spl[1]
    else:
        alg = None
        task = None
    return alg, task


def add_to_list(item, l):
    add = False
    if item not in l:
        l.append(item)
        add = True
    return l, add


def build_pair(alg, task):
    return alg + ' / ' + task


def inventory(folder, inv=None):
    # inventory
    files = glob.glob(folder + '/*')
    if inv is None:
        inv = {'algs': [], 'tasks': [], 'runs': {}}
    for ind_f, file in enumerate(files):
        alg, task = get_alg_task(ntpath.basename(file))
        if alg is not None:
            inv['algs'], _ = add_to_list(alg, inv['algs'])
            inv['tasks'], _ = add_to_list(task, inv['tasks'])
            pair = build_pair(alg, task)
            _, add_flag = add_to_list(pair, list(inv['runs'].keys()))
            if add_flag:
                inv['runs'][pair] = [file]
            else:
                inv['runs'][pair].append(file)
    return inv


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 16})
    if len(sys.argv) > 2:
        raise ValueError("usage: get_performances.py [folder]")
    main_folder = sys.argv[1]
    folders = glob.glob(main_folder + '/*')
    print(folders)
    inv = None
    for f in folders:
        inv = inventory(folder=f, inv=inv)
    colors = sns.color_palette()
    tasks = inv['tasks']
    algs = inv['algs']
    runs = inv['runs']
    rows = 2
    cols = 2
    f, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 20))
    ax = ax.flatten()
    ax_count = -1
    fig_count = 0
    for indt, t in enumerate(tasks):
        print('xxxxxxxx')
        print(t)
        ax_count += 1
        if ax_count == rows*cols:
            ax_count = 0
            f.savefig(main_folder +
                      '/mean_reward_across_training_'+str(fig_count)+'.png')
            f, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 20))
            ax = ax.flatten()
            fig_count += 1
        for indalg, alg in enumerate(algs):
            pair = build_pair(alg, t)
            for ind_inst in range(len(runs[pair])):
                path = runs[pair][ind_inst] + '/'
                print(path)
                plotting.plot_rew_across_training(path, window=0.05,
                                                  ax=ax[ax_count], ytitle=t,
                                                  legend=(ind_inst == 0),
                                                  zline=True,
                                                  fkwargs={'c': colors[indalg],
                                                           'ls': '--',
                                                           'alpha': 0.5,
                                                           'label': alg})
