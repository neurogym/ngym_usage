#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:59:08 2020

@author: molano
"""
import gym
import neurogym  # need to import it so ngym envs are registered
from neurogym.wrappers import trial_hist
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C  # , ACER, ACKTR, PPO2

kwargs = {'dt': 100}
env = gym.make('RDM-v0', **kwargs)
env = trial_hist(env, PARAMS)
env = DummyVecEnv([lambda: env])
model = A2C(LstmPolicy, env, verbose=0)
model.learn(total_timesteps=100000)
print('model trained!')
