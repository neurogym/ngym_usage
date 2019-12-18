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
num_tr = 5
n_stps_tst = 5000
perfs = {'A2C': np.zeros((3, num_tr)), 'ACER': np.zeros((3, num_tr)),
         'ACKTR': np.zeros((3, num_tr)), 'PPO2': np.zeros((3, num_tr))}
algs_names = ['A2C', 'ACER', 'ACKTR', 'PPO2']
algs = [A2C, ACER, ACKTR, PPO2]

for ind_tr in range(num_tr):
    for ind_alg, algorithm in enumerate(algs):
        # RDM
        alg = algs_names[ind_alg]
        task = 'RDM'
        timing = {'fixation': [500, 500, 500], 'stimulus': [500, 200, 800],
                  'delay_aft_stim': [0, 0, 0], 'decision': [100, 100, 100]}
        simultaneous_stim = True
        env_args = {'timing': timing, 'simultaneous_stim': simultaneous_stim}
        env = gym.make('2AFC-v0', **env_args)
        env = DummyVecEnv([lambda: env])
        model = algorithm(MlpPolicy, env, verbose=1)
        model.learn(total_timesteps=50000)  # 50000)
        perfs[alg][0, ind_tr] = tasktools.plot_struct(env,
                                                      num_steps_env=n_stps_tst,
                                                      model=model,
                                                      name=alg+' '+task)
        env.close()
        
        # ROMO
        task = 'ROMO'
        timing = {'fixation': [500, 500, 500], 'stimulus': [500, 200, 800],
                  'delay_btw_stim': [500, 200, 800],
                  'delay_aft_stim': [0, 0, 0], 'decision': [100, 100, 100]}
        simultaneous_stim = False
        env_args = {'timing': timing, 'simultaneous_stim': simultaneous_stim}
        env = gym.make('2AFC-v0', **env_args)
        env = DummyVecEnv([lambda: env])
        model = algorithm(MlpPolicy, env, verbose=1)
        model.learn(total_timesteps=50000)  # 50000)
        perfs[alg][1, ind_tr] = tasktools.plot_struct(env,
                                                      num_steps_env=n_stps_tst,
                                                      model=model,
                                                      name=alg+' '+task)
        env.close()
        
        # DELAY RESPONSE
        task = 'DELAY RESPONSE'
        timing = {'fixation': [300, 300, 300], 'stimulus': [500, 200, 800],
                  'delay_aft_stim': [200, 100, 300], 'decision': [100, 100, 100]}
        simultaneous_stim = True
        env_args = {'timing': timing, 'simultaneous_stim': simultaneous_stim}
        env = gym.make('2AFC-v0', **env_args)
        env = DummyVecEnv([lambda: env])
        model = algorithm(MlpPolicy, env, verbose=1)
        model.learn(total_timesteps=50000)  # 50000)
        perfs[alg][2, ind_tr] = tasktools.plot_struct(env,
                                                      num_steps_env=n_stps_tst,
                                                      model=model,
                                                      name=alg+' '+task)
        plt.close('all')
        env.close()
        
        