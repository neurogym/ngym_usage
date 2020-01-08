import matplotlib
import numpy as np
import glob
import os
from priors.codes.ops import utils as ut
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
N_CONV = 50


def inventory(folder):
    files = glob.glob(folder + '*')
    envs = []
    algs = []
    exps = []
    for path in files:
        file = os.path.basename(path)
        env = file[:file.find('_')]
        if env not in envs:
            envs.append(env)
        file = file[file.find('_')+1:]
        alg = file[:file.find('_')]
        if alg not in algs:
            algs.append(alg)
        exps.append([env, alg])
    
    return envs, algs, exps, files


def plot_perf_exp(folder, fig=True):
    files = glob.glob(folder + '*bhvr_data*')
    files = ut.order_by_sufix(files)
    colors = 'rgbkm'
    if fig:
        plt.figure()
    tr_counter = 0
    for file in files:
        data = np.load(file)
        if file.find('CurriculumLearning') != -1:
            for ind_ph in range(5):
                inds_ph = data['curr_ph'] == ind_ph
                if np.sum(inds_ph):
                    nc = min(N_CONV, np.sum(inds_ph))
                    plt.plot(np.arange(np.sum(inds_ph))+tr_counter,
                             np.convolve(data['first_rew'][inds_ph],
                                         np.ones((nc,))/nc, mode='same'),
                             color=colors[ind_ph])
                    tr_counter += np.sum(inds_ph)
        else:
            plt.plot(np.arange(data['reward'].shape[0])+tr_counter,
                     np.convolve(data['reward'], np.ones((N_CONV,))/N_CONV,
                                 mode='same'), color='k')
            tr_counter += data['reward'].shape[0]


def plot_perf_all(folder, rows=3, cols=4, values=[.25, .5, .75]):
    envs, algs, exps, files = inventory(folder)
    envs.sort()
    figs = []
    for ind_f in range(len(algs)):
        figs.append(plt.figure())
    for ind_f, f in enumerate(files):
        exp = exps[ind_f]
        plt.figure(figs[algs.index(exp[1])].number)
        plt.subplot(rows, cols, envs.index(exp[0])+1)
        plot_perf_exp(f+'/', fig=False)
    for ind_a, alg in enumerate(algs):
        for ind_e, env in enumerate(envs):
            plt.figure(figs[algs.index(alg)].number)
            plt.subplot(rows, cols, envs.index(env)+1)
            plt.title(env + '  ' + alg)
            ax = plt.gca()
            ax.set_ylim([0, 1])
            xlims = ax.get_xlim()
            for ind_v in values:
                plt.plot(xlims, [ind_v]*2, '--y')
            


if __name__ == '__main__':
    plt.close('all')
    main_folder = '/home/molano/CV_learning/'
    plot_perf_all(main_folder)
