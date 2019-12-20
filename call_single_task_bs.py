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
import numpy as np
plt.close('all')
num_tr = 10
n_stps_tr = 50000
n_stps_tst = 5000
tasks = ['GNG', 'DPA', 'RDM', 'ROMO', 'DELAYRESPONSE']
algs_names = ['A2C', 'ACER', 'ACKTR', 'PPO2']
perfs = {alg: np.zeros((len(tasks), num_tr)) for alg in algs_names}
algs = [A2C, ACER, ACKTR, PPO2]
for ind_tr in range(num_tr):
    for ind_alg, algorithm in enumerate(algs):
        # RDM
        alg = algs_names[ind_alg]
        for ind_t, task in enumerate(tasks):
            simultaneous_stim = True
            gng = False
            cohs = [0, 6.4, 12.8, 25.6, 51.2]
            ng_task = 'GenTask-v0'
            if task == 'RDM':
                timing = {'fixation': [100, 100, 100],
                          'stimulus': [500, 200, 800],
                          'delay_btw_stim': [0, 0, 0],  # for sake of clarity
                          'delay_aft_stim': [0, 0, 0],
                          'decision': [200, 200, 200]}
            elif task == 'ROMO':
                timing = {'fixation': [100, 100, 100],
                          'stimulus': [500, 200, 800],
                          'delay_btw_stim': [300, 200, 400],
                          'delay_aft_stim': [0, 0, 0],
                          'decision': [200, 200, 200]}
                simultaneous_stim = False
            elif task == 'DELAY RESPONSE':
                timing = {'fixation': [100, 100, 100],
                          'stimulus': [500, 200, 800],
                          'delay_btw_stim': [0, 0, 0],
                          'delay_aft_stim': [200, 100, 300],
                          'decision': [200, 200, 200]}
            elif task == 'GNG':
                timing = {'fixation': [100, 100, 100],
                          'stimulus': [500, 200, 800],
                          'delay_btw_stim': [0, 0, 0],
                          'delay_aft_stim': [200, 100, 300],
                          'decision': [200, 200, 200]}
                gng = True
                cohs = [100]
            elif task == 'DPA':
                timing = {'fixation': [100, 100, 100],
                          'stimulus': [200, 100, 300],
                          'delay_btw_stim': [200, 100, 300],
                          'delay_aft_stim': [200, 100, 300],
                          'decision': [200, 200, 200]}
                ng_task = 'DPA-v1'
                simultaneous_stim = False

            env_args = {'timing': timing, 'gng': gng, 'cohs': cohs,
                        'simultaneous_stim': simultaneous_stim}
            env = gym.make(ng_task, **env_args)
            env = DummyVecEnv([lambda: env])
            model = algorithm(MlpPolicy, env, verbose=1)
            model.learn(total_timesteps=n_stps_tr)  # 50000)
            perfs[alg][ind_t, ind_tr] =\
                tasktools.plot_struct(env, num_steps_env=n_stps_tst,
                                      model=model, name=alg+' '+task)
            env.close()
            sdf
            plt.close('all')
plt.figure()
for name in algs_names:
    plt.errorbar(np.arange(len(tasks)), np.mean(perfs[name], axis=1),
                 np.std(perfs[name], axis=1))
