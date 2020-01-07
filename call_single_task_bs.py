#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:05:09 2019

@author: manuel
"""
import matplotlib
import sys
import os
import glob
import numpy as np
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, ACER, ACKTR, PPO2
import gym
import neurogym
import matplotlib.pyplot as plt
from neurogym.wrappers import trial_hist_nalt as thn
from neurogym.wrappers import cv_learning as cv_l
from neurogym.wrappers import manage_data as md
from priors.codes.ops import utils as ut
matplotlib.use('Qt5Agg')
plt.close('all')
num_tr = 10
n_stps_tr = 1000000
n_tr_sv = 100000
# trial-history tasks params
n_ch = 2
tr_prob = 0.8
block_dur = 200
trans = 'RepAlt'
# dual-task params
delay = 800
mix = [.3, .3, .4]
share_action_space = True
defaults = [0, 0]
# cv task parameters
th = 0.8
perf_w = 100
init_ph = 0
main_folder = '/home/molano/CV_learning/'
tasks = ['DUAL-TASK', 'DELAY-RESPONSE-cv', 'RDM-hist', 'ROMO-hist',
         'DELAY-RESPONSE-hist', 'DPA', 'GNG', 'RDM', 'ROMO', 'DELAY-RESPONSE']
# '
algs_names = ['A2C', 'ACER', 'ACKTR', 'PPO2']
algs = [A2C, ACER, ACKTR, PPO2]
for ind_tr in range(num_tr):
    for ind_alg, algorithm in enumerate(algs):
        # RDM
        alg = algs_names[ind_alg]
        for ind_t, task in enumerate(tasks):
            folder = main_folder + task + '_' + alg + '_' + str(ind_tr) + '/'
            if not os.path.exists(folder):
                os.makedirs(folder)
            print('---------------')
            print(task)
            hist = False
            dual_task = False
            cv = False
            if task == 'RDM':
                timing = {'fixation': [100, 100, 100],
                          'stimulus': [500, 200, 800],
                          'delay_btw_stim': [0, 0, 0],  # for sake of clarity
                          'delay_aft_stim': [0, 0, 0],
                          'decision': [200, 200, 200]}
                simultaneous_stim = True
                gng = False
                cohs = [0, 6.4, 12.8, 25.6, 51.2]
                ng_task = 'GenTask-v0'
            elif task == 'ROMO':
                timing = {'fixation': [100, 100, 100],
                          'stimulus': [500, 200, 800],
                          'delay_btw_stim': [300, 200, 400],
                          'delay_aft_stim': [0, 0, 0],
                          'decision': [200, 200, 200]}
                simultaneous_stim = False  # MAIN CHANGE WRT RDM TASK
                gng = False
                cohs = [0, 6.4, 12.8, 25.6, 51.2]
                ng_task = 'GenTask-v0'
            elif task == 'DELAY-RESPONSE':
                timing = {'fixation': [100, 100, 100],
                          'stimulus': [500, 200, 800],
                          'delay_btw_stim': [0, 0, 0],
                          'delay_aft_stim': [0, 0, 0],  # MAIN CHANGE
                          'decision': [200, 200, 200]}
                simultaneous_stim = True
                gng = False
                cohs = [51.2]
                ng_task = 'GenTask-v0'
            elif task == 'GNG':
                timing = {'fixation': [100, 100, 100],
                          'stimulus': [500, 200, 800],
                          'delay_btw_stim': [0, 0, 0],
                          'delay_aft_stim': [200, 100, 300],
                          'decision': [200, 200, 200]}
                gng = True  # MAIN CHANGE
                cohs = [100]  # MAIN CHANGE
                simultaneous_stim = True
                ng_task = 'GenTask-v0'
            elif task == 'DPA':
                timing = {'fixation': [100, 100, 100],
                          'stimulus': [200, 100, 300],
                          'delay_btw_stim': [200, 100, 300],
                          'delay_aft_stim': [200, 100, 300],
                          'decision': [200, 200, 200]}
                ng_task = 'DPA-v1'  # MAIN CHANGE
                simultaneous_stim = False
            elif task == 'RDM-hist':
                timing = {'fixation': [100, 100, 100],
                          'stimulus': [500, 200, 800],
                          'delay_btw_stim': [0, 0, 0],  # for sake of clarity
                          'delay_aft_stim': [0, 0, 0],
                          'decision': [200, 200, 200]}
                simultaneous_stim = True
                gng = False
                cohs = [0, 6.4, 12.8, 25.6, 51.2]
                ng_task = 'GenTask-v0'
                hist = True
            elif task == 'ROMO-hist':
                timing = {'fixation': [100, 100, 100],
                          'stimulus': [500, 200, 800],
                          'delay_btw_stim': [300, 200, 400],
                          'delay_aft_stim': [0, 0, 0],
                          'decision': [200, 200, 200]}
                simultaneous_stim = False  # MAIN CHANGE WRT RDM TASK
                gng = False
                cohs = [0, 6.4, 12.8, 25.6, 51.2]
                ng_task = 'GenTask-v0'
                hist = True
            elif task == 'DELAY-RESPONSE-hist':
                timing = {'fixation': [100, 100, 100],
                          'stimulus': [500, 200, 800],
                          'delay_btw_stim': [0, 0, 0],
                          'delay_aft_stim': [200, 100, 300],  # MAIN CHANGE
                          'decision': [200, 200, 200]}
                simultaneous_stim = True
                gng = False
                cohs = [0, 6.4, 12.8, 25.6, 51.2]
                ng_task = 'GenTask-v0'
                hist = True
            elif task == 'DUAL-TASK':
                timing = {'fixation': [100, 100, 100],
                          'stimulus': [200, 100, 300],
                          'delay_btw_stim': [200, 100, 300],
                          'delay_aft_stim': [200, 100, 300],
                          'decision': [200, 200, 200]}
                ng_task = 'DPA-v1'  # MAIN CHANGE
                simultaneous_stim = False
                timing_2 = {'fixation': [100, 100, 100],
                            'stimulus': [500, 200, 800],
                            'delay_btw_stim': [0, 0, 0],
                            'delay_aft_stim': [200, 100, 300],
                            'decision': [200, 200, 200]}
                gng_2 = True  # MAIN CHANGE
                cohs_2 = [100]  # MAIN CHANGE
                simultaneous_stim_2 = True
                ng_task_2 = 'GenTask-v0'
                dual_task = True
            elif task == 'DELAY-RESPONSE-cv':
                timing = {'fixation': [100, 100, 100],
                          'stimulus': [500, 200, 800],
                          'delay_btw_stim': [0, 0, 0],
                          'delay_aft_stim': [100, 100, 100],  # MAIN CHANGE
                          'decision': [200, 200, 200]}
                simultaneous_stim = True
                gng = False
                cohs = [0, 6.4, 12.8, 25.6, 51.2]
                ng_task = 'GenTask-v0'
                cv = True
            else:
                sys.exit("'NO TASK!!'")

            # combinations and wrappers
            if dual_task:
                params_1 = {'timing': timing}
                params_2 = {'timing': timing_2, 'gng': gng_2, 'cohs': cohs_2,
                            'simultaneous_stim': simultaneous_stim_2}

                all_params = {'env_name_1': ng_task, 'env_name_2': ng_task_2,
                              'params_1': params_1, 'params_2': params_2,
                              'delay': 800, 'mix': [.3, .3, .4],
                              'share_action_space': True, 'defaults': [0, 0]}
                env = gym.make('Combine-v0', **all_params)
            else:
                env_args = {'timing': timing, 'gng': gng, 'cohs': cohs,
                            'simultaneous_stim': simultaneous_stim}
                env = gym.make(ng_task, **env_args)
                if hist:
                    thn.TrialHistory_NAlt(env, n_ch=n_ch, tr_prob=tr_prob,
                                          block_dur=block_dur, trans=trans)
                if cv:
                    env = cv_l.CurriculumLearning(env, th=th, perf_w=perf_w,
                                                  init_ph=init_ph)
            env = md.manage_data(env, folder=folder, num_tr_save=n_tr_sv)
            env = DummyVecEnv([lambda: env])
            try:
                model = algorithm(LstmPolicy, env, verbose=0)
                model.learn(total_timesteps=n_stps_tr)  # 50000)
            except:
                print('could not train')
            env.close()
            plt.close('all')

# plot
nc = 100
perfs = {alg: np.zeros((len(tasks), num_tr)) for alg in algs_names}
plt.figure()
for ind_tr in range(num_tr):
    for ind_alg, algorithm in enumerate(algs):
        # RDM
        alg = algs_names[ind_alg]
        for ind_t, task in enumerate(tasks):
            folder = main_folder + task + '_' + alg + '_' + str(ind_tr) + '/'
            files = glob.glob(folder + '/*bhvr_data*')
            files = ut.order_by_sufix(files)
            for file in files:
                data = np.load(file)
                if 'first_rew' in list(data):
                    plt.plot(np.convolve(data['first_rew'],
                             np.ones((nc,))/nc, mode='same'))
                else:
                    plt.plot(np.convolve(data['reward'],
                             np.ones((nc,))/nc, mode='same'))
