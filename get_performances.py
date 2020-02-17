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
from neurogym.utils import plotting as pl
#    selected_tasks = ['ContextDecisionMaking-v0', 'GoNogo-v0',
#                      'ReadySetGo-v0', 'DawTwoStep-v0', 'MatchingPenny-v0',
#                      'PerceptualDecisionMakingDelayResponse-v0',
#                      'IntervalDiscrimination-v0', 'Detection-v0',
#                      'ReachingDelayResponse-v0', 'ChangingEnvironment-v0']
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


def plot_SL_rew_across_train(folder, ax, ytitle='', legend=False,
                             zline=False, fkwargs={'c': 'tab:blue'},
                             metric_name='reward'):
    files = glob.glob(folder + '/*_bhvr_data*npz')
    if len(files) > 0:
        files = order_by_sufix(files)
        metric_mat = []
        counts = []
        trials_count = 0
        for ind_f in range(len(files)):
            file_data = np.load(files[ind_f], allow_pickle=True)
            metric = file_data[metric_name][10:]
            metric_mat.append(np.mean(metric))
            counts.append(trials_count+metric.shape[0]/2)
            trials_count += metric.shape[0]
        ax.plot(counts, metric_mat, **fkwargs)
        ax.set_xlabel('trials')
        if not ytitle:
            ax.set_ylabel('mean ' + metric_name +
                          '({:d} trials)'.format(metric.shape[0]))
        else:
            ax.set_ylabel(ytitle)
        if legend:
            ax.legend()
        if zline:
            ax.axhline(0, c='k', ls=':')
    else:
        print('No data in: ', folder)


def order_by_sufix(file_list):
    temp = [x[:x.rfind('_')] for x in file_list]
    sfx = [int(x[x.rfind('_')+1:]) for x in temp]
    sorted_list = [x for _, x in sorted(zip(sfx, file_list))]
    return sorted_list


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
    print(inv)
    colors = sns.color_palette()
    tasks = inv['tasks']
    # tasks = [x for x in tasks if x in selected_tasks]
    algs = inv['algs']
    runs = inv['runs']
    rows = 2
    cols = 2
    for metric_name in ['reward', 'performance']:
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
                          '/mean_' + metric_name + '_across_training_' +
                          str(fig_count)+'.png')
                f, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 20))
                ax = ax.flatten()
                fig_count += 1
            for indalg, alg in enumerate(algs):
                pair = build_pair(alg, t)
                if pair in runs.keys():
                    for ind_inst in range(len(runs[pair])):
                        path = runs[pair][ind_inst] + '/'
                        print(path)
                        c = colors[indalg]
                        lbl = alg if ind_inst == 0 else ''
                        if alg != 'SL':
                            pl.plot_rew_across_training(path, window=0.05,
                                                        ax=ax[ax_count],
                                                        ytitle=t,
                                                        legend=False,
                                                        zline=True,
                                                        metric=metric_name,
                                                        fkwargs={'c': c,
                                                                 'ls': '--',
                                                                 'alpha': 0.5,
                                                                 'label': lbl})
                        else:
                            plot_SL_rew_across_train(folder=path,
                                                     ax=ax[ax_count],
                                                     ytitle=t,
                                                     legend=False,
                                                     zline=True,
                                                     metric=metric_name,
                                                     fkwargs={'c': c,
                                                              'ls': '--',
                                                              'alpha': 0.5,
                                                              'label': lbl,
                                                              'marker': '+'})
            ax[ax_count].legend()
        f.savefig(main_folder + '/mean_' + metric_name + '_across_training_' +
                  str(fig_count)+'.png')
