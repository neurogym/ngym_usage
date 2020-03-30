#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:32:02 2020

@author: molano
"""
import sys
import glob
import ntpath
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 7

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


def plot_RL_rew_acr_train(folder, window=500, ax=None,
                          fkwargs={'c': 'tab:blue'}, ylabel='',
                          legend=False, zline=False, metric_name='reward'):
    data = put_together_files(folder)
    mean_metric = []
    if data:
        sv_fig = False
        if ax is None:
            sv_fig = True
            f, ax = plt.subplots(figsize=(8, 8))
        metric = data[metric_name]
        if isinstance(window, float):
            if window < 1.0:
                window = int(metric.size * window)
        mean_metric = np.convolve(metric, np.ones((window,))/window,
                                  mode='valid')
        ax.plot(np.arange(len(mean_metric))/1000, mean_metric, **fkwargs)
        ax.set_xlabel('Trials (x1000)')
        if not ylabel:
            string = 'mean ' + metric_name + ' (running window' +\
                ' of {:d} trials)'.format(window)
            ax.set_ylabel(string.capitalize())
        else:
            ax.set_ylabel(ylabel.capitalize())
        if legend:
            ax.legend()
        if zline:
            ax.axhline(0, c='k', ls=':')
        if sv_fig:
            f.savefig(folder + '/mean_' + metric_name + '_across_training.png')
    else:
        print('No data in: ', folder)
    return mean_metric


def plot_SL_rew_acr_train(folder, ax, ylabel='', legend=False,
                          zline=False, fkwargs={'c': 'tab:blue'},
                          metric_name='reward'):
    files = glob.glob(folder + '/*_bhvr_data*npz')
    metric_mat = []
    counts = []
    if len(files) > 0:
        files = order_by_sufix_SL(files)
        trials_count = 0
        for ind_f in range(len(files)):
            file_data = np.load(files[ind_f], allow_pickle=True)
            metric = file_data[metric_name][10:]
            metric_mat.append(np.mean(metric))
            counts.append((trials_count+metric.shape[0]/2)/1000)
            trials_count += metric.shape[0]
        ax.plot(counts, metric_mat, **fkwargs)
        ax.set_xlabel('Trials (x1000)')
        if not ylabel:
            string = 'mean ' + metric_name +\
                '({:d} trials)'.format(metric.shape[0])
            ax.set_ylabel(string.capitalize())
        else:
            ax.set_ylabel(ylabel.capitalize())
        if legend:
            ax.legend()
        if zline:
            ax.axhline(0, c='k', ls=':')
    else:
        print('No data in: ', folder)
    return metric_mat, counts


def put_together_files(folder):
    files = glob.glob(folder + '/*_bhvr_data*npz')
    data = {}
    if len(files) > 0:
        files = order_by_sufix_RL(files)
        file_data = np.load(files[0], allow_pickle=True)
        for key in file_data.keys():
            data[key] = file_data[key]

        for ind_f in range(1, len(files)):
            file_data = np.load(files[ind_f], allow_pickle=True)
            for key in file_data.keys():
                data[key] = np.concatenate((data[key], file_data[key]))
        np.savez(folder + '/bhvr_data_all.npz', **data)
    return data


def order_by_sufix_RL(file_list):
    sfx = [int(x[x.rfind('_')+1:x.rfind('.')]) for x in file_list]
    sorted_list = [x for _, x in sorted(zip(sfx, file_list))]
    return sorted_list


def order_by_sufix_SL(file_list):
    temp = [x[:x.rfind('_')] for x in file_list]
    sfx = [int(x[x.rfind('_')+1:]) for x in temp]
    sorted_list = [x for _, x in sorted(zip(sfx, file_list))]
    return sorted_list


if __name__ == '__main__':
    selected_exps = ['200214']
    # plt.rcParams.update({'font.size': 16})
    if len(sys.argv) > 2:
        raise ValueError("usage: get_performances.py [folder]")
    main_folder = sys.argv[1]
    folders = glob.glob(main_folder + '/*')
    inv = None
    for f in folders:
        if ntpath.basename(f) in selected_exps:
            inv = inventory(folder=f, inv=inv)
    colors = sns.color_palette()
    tasks = inv['tasks']
    # tasks = [x for x in tasks if x in selected_tasks]
    algs = inv['algs']
    runs = inv['runs']
    rows = 1
    cols = 2
    ax_count = -1
    fig_count = 0
    for indt, t in enumerate(tasks):
        print('xxxxxxxx')
        print(t)
        f, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(8, 4))
        ax = ax.flatten()
        for indalg, alg in enumerate(sorted(algs)):
            for ind_met, met in enumerate(['reward', 'performance']):
                ylabel = 'Average ' + met
                metr_mat = []
                pair = build_pair(alg, t)
                if pair in runs.keys():
                    for ind_inst in range(len(runs[pair])):
                        path = runs[pair][ind_inst] + '/'
                        print(path)
                        c = colors[indalg]
                        lbl = alg if ind_inst == 0 else ''
                        if alg != 'SL':
                            metr = plot_RL_rew_acr_train(path, window=0.05,
                                                         ax=ax[ind_met],
                                                         ylabel=ylabel,
                                                         legend=False,
                                                         zline=True,
                                                         metric_name=met,
                                                         fkwargs={'c': c,
                                                                  'ls': '-',
                                                                  'alpha': 0.2,
                                                                  'label': '',
                                                                  'lw': 0.5})
                        else:
                            metr, counts =\
                                plot_SL_rew_acr_train(folder=path,
                                                      ax=ax[ind_met],
                                                      ylabel=ylabel,
                                                      legend=False, zline=True,
                                                      metric_name=met,
                                                      fkwargs={'c': c,
                                                               'ls': '--',
                                                               'alpha': 0.2,
                                                               'label': '',
                                                               'lw': 0.5})
                        if len(metr) > 0:
                            metr_mat.append(metr)
                if len(metr_mat) > 0:
                    min_dur = np.min([len(x) for x in metr_mat])
                    metr_mat = [x[:min_dur] for x in metr_mat]
                    metr_mat = np.array(metr_mat)
                    c = colors[indalg]
                    sh = metr_mat.shape[1]
                    if alg == 'SL':
                        xs = counts[:metr_mat.shape[1]]
                    else:
                        xs = np.arange(sh)/1000
                    ax[ind_met].plot(xs, np.nanmean(metr_mat, axis=0), color=c,
                                     lw=1, label=alg)
        ax[0].legend()
        ax[1].set_ylim([0, 1])
        for x in ax:
            xlim = x.get_xlim()
            x.set_xlim([0, xlim[1]])
            # add xK
            # xticks = x.get_xticks().tolist()
            # xticks.append(xticks[-1]+(xticks[1]-xticks[0])/1.5)
            # x.set_xticks(xticks)
            # xticks = [str(x) for x in xticks]
            # xticks[-1] = 'x1000'
            # x.set_xticklabels(xticks)
            x.spines['right'].set_visible(False)
            x.spines['top'].set_visible(False)
        plt.suptitle(t)
        f.savefig(main_folder + '/means_across_training_' + t + '.png', dpi=400)
