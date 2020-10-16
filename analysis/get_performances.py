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


def inventory(folder):
    # inventory
    files = glob.glob(folder + '/*')
    inv = {'algs': [], 'tasks': [], 'num_instances': {}}
    for ind_f, file in enumerate(files):
        alg, task = get_alg_task(ntpath.basename(file))
        if alg is not None:
            inv['algs'], _ = add_to_list(alg, inv['algs'])
            inv['tasks'], _ = add_to_list(task, inv['tasks'])
            pair = build_pair(alg, task)
            _, add_flag = add_to_list(pair,
                                      list(inv['num_instances'].keys()))
            if add_flag:
                inv['num_instances'][pair] = 1
            else:
                inv['num_instances'][pair] += 1
    for key in sorted(inv['num_instances'].keys()):
        print(key + ' (' + str(inv['num_instances'][key]) + ' instances)')

    return inv


if __name__ == '__main__':

    if len(sys.argv) > 2:
        raise ValueError("usage: get_performances.py [folder]")
    folder = sys.argv[1]

    inv = inventory(folder)
    colors = sns.color_palette()
    tasks = inv['tasks']
    algs = inv['algs']
    runs = inv['num_instances']
    rows = 6
    cols = int(np.ceil(len(tasks)/rows))
    f, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 20))
    ax = ax.flatten()
    for indt, t in enumerate(tasks):
        for indalg, alg in enumerate(algs):
            pair = build_pair(alg, t)
            for ind_inst in range(runs[pair]):
                path = folder + '/{}_{}_{}/'.format(alg, t, ind_inst)
                plotting.plot_rew_across_training(path, window=0.05,
                                                  ax=ax[indt], ytitle=t,
                                                  legend=True, zline=True,
                                                  fkwargs={'c': colors[indalg],
                                                           'ls': '--',
                                                           'alpha': 0.5,
                                                           'label': alg})
            # plt.legend()
    f.savefig(folder + '/mean_reward_across_training.png')
