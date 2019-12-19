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
num_tr = 1
n_stps_tr = 50000
n_stps_tst = 5000
tasks = ['RDM', 'ROMO', 'DELAYRESPONSE']
algs_names = ['A2C', 'ACER', 'ACKTR', 'PPO2']
perfs = {alg: np.zeros((len(tasks), num_tr)) for alg in algs_names}
algs = [A2C, ACER, ACKTR, PPO2]
for ind_tr in range(num_tr):
    for ind_alg, algorithm in enumerate(algs):
        # RDM
        alg = algs_names[ind_alg]
        for task in tasks:
            if task == 'RDM':
                timing = {'fixation': [100, 100, 100],
                          'stimulus': [500, 200, 800],
                          'delay_aft_stim': [0, 0, 0],
                          'decision': [200, 200, 200]}
                simultaneous_stim = True
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
                          'delay_aft_stim': [200, 100, 300],
                          'decision': [200, 200, 200]}
                simultaneous_stim = True

            env_args = {'timing': timing,
                        'simultaneous_stim': simultaneous_stim}
            env = gym.make('2AFC-v0', **env_args)
            env = DummyVecEnv([lambda: env])
            model = algorithm(MlpPolicy, env, verbose=1)
            model.learn(total_timesteps=n_stps_tr)  # 50000)
            print(task)
            print(alg)
            perfs[alg][0, ind_tr] =\
                tasktools.plot_struct(env,
                                      num_steps_env=n_stps_tst,
                                      model=model,
                                      name=alg+' '+task)
            env.close()
            plt.close('all')
