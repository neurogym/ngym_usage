"""Analyze."""

import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import neurogym as ngym

from train import get_logname

tasks = ngym.get_collection('yang19')

def load_perf_tensor():
    task = pretrain_task = tasks[0]
    log = np.load(os.path.join('files', get_logname(task, pretrain_task)))
    len_perf = len(log['perf'])
    n_task = len(tasks)
    
    perf_tensor = np.zeros((n_task, n_task, len_perf))
    
    for i in range(n_task):
        for j in range(n_task):
            task = tasks[i]
            pretrain_task = tasks[j]
            log = np.load(os.path.join('files', get_logname(task, pretrain_task)))
            perf_tensor[i, j, :] = log['perf']
    return perf_tensor


def plot_perf_matrix(perf_tensor, normalize=True):
    perf_show = perf_tensor[:, :, -1]
    perf_show = perf_tensor.mean(axis=-1)
    perf_show = perf_tensor[:, :, 1]
    
    if normalize:
        perf_show = (perf_show.T / np.max(perf_show, axis=1)).T
    
    n = perf_show.shape[0]
    
    # fig = plt.figure(figsize=(3, 3))
    # ax = fig.add_axes([0.4, 0.4, 0.4, 0.4])
    plt.figure()
    im = plt.imshow(perf_show, extent=(-0.5, n-0.5, -0.5, n-0.5))
    tick_names = [task[len('yang19.'):-len('-v0')] for task in tasks]
    fs = 7
    plt.yticks(range(len(tick_names)), tick_names[::-1],
               rotation=0, va='center', fontsize=fs)
    plt.xticks(range(len(tick_names)), tick_names,
               rotation=90, va='top', fontsize=fs)
    plt.xlabel('Pre-trained task')
    plt.ylabel('Task')
    
    plt.plot([-0.5, n-0.5], [n-0.5, -0.5], color='white')
    plt.tick_params(length=0)
    
    cb = plt.colorbar(im)
    cb.outline.set_linewidth(0.5)
    clabel = 'Normalized Accuracy' if normalize else 'Accuracy'
    cb.set_label(clabel, fontsize=7, labelpad=0)
    

perf_tensor = load_perf_tensor()
plot_perf_matrix(perf_tensor, normalize=True)
plot_perf_matrix(perf_tensor, normalize=False)