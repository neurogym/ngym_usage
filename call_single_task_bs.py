#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:05:09 2019

@author: manuel
"""
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, ACER, ACKTR, PPO2
import gym
import neurogym
import matplotlib.pyplot as plt
from neurogym.ops import tasktools
from neurogym.wrappers import trial_hist_nalt as thn
import numpy as np
plt.close('all')
num_tr = 2
n_stps_tr = 50000
n_stps_tst = 5000
# trial-history tasks params
n_ch = 2
tr_prob = 0.8
block_dur = 200
trans = 'RepAlt'
tasks = ['RDM-hist', 'ROMO-hist', 'DELAYRESPONSE-hist',
         'DPA', 'GNG', 'RDM', 'ROMO', 'DELAYRESPONSE']
algs_names = ['A2C', 'ACER', 'ACKTR', 'PPO2']
perfs = {alg: np.zeros((len(tasks), num_tr)) for alg in algs_names}
algs = [A2C, ACER, ACKTR, PPO2]
for ind_tr in range(num_tr):
    for ind_alg, algorithm in enumerate(algs):
        # RDM
        alg = algs_names[ind_alg]
        for ind_t, task in enumerate(tasks):
            hist = False
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
            elif task == 'DELAY RESPONSE':
                timing = {'fixation': [100, 100, 100],
                          'stimulus': [500, 200, 800],
                          'delay_btw_stim': [0, 0, 0],
                          'delay_aft_stim': [200, 100, 300],  # MAIN CHANGE
                          'decision': [200, 200, 200]}
                simultaneous_stim = True
                gng = False
                cohs = [0, 6.4, 12.8, 25.6, 51.2]
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
            if task == 'RDM-hist':
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
            elif task == 'DELAY RESPONSE-hist':
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
            env_args = {'timing': timing, 'gng': gng, 'cohs': cohs,
                        'simultaneous_stim': simultaneous_stim}
            env = gym.make(ng_task, **env_args)
            if hist:
                thn.TrialHistory_NAlt(env, n_ch=n_ch, tr_prob=tr_prob,
                                      block_dur=block_dur, trans=trans)
            env = DummyVecEnv([lambda: env])
            model = algorithm(MlpPolicy, env, verbose=1)
            model.learn(total_timesteps=n_stps_tr)  # 50000)
            perfs[alg][ind_t, ind_tr] =\
                tasktools.plot_struct(env, num_steps_env=n_stps_tst,
                                      model=model, name=alg+' '+task)
            env.close()
            plt.close('all')
plt.figure()
for name in algs_names:
    plt.errorbar(np.arange(len(tasks)), np.mean(perfs[name], axis=1),
                 np.std(perfs[name], axis=1))
