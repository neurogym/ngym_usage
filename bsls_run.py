#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test training according to params (model, task, seed)
"""
import os
import sys
sys.path.append(os.path.expanduser('~/gym'))
sys.path.append(os.path.expanduser('~/stable-baselines'))
sys.path.append(os.path.expanduser('~/neurogym'))

if len(sys.argv)!=4:
    raise ValueError('usage: bsls_run.py [model] [task] [seed]')

# ARGS
alg = sys.argv[1] # a2c acer acktr or ppo2
task = sys.argv[2] # ngym task (neurogym.all_tasks.keys())
seed = int(sys.argv[3])

# other relevant vars
tot_timesteps = 1e5
n_cpu_tf = 1 # (ppo2 will crash)
n_steps = 20 # use 20 if short periods, else 100
dts = 100

states_list = """fixation
stim1
delay1
stim2
delay2
go1
go2
reach
delay_btw_stim
delay_aft_stim
decision
sample
first_delay
test
second_delay
delay
test1
test2
delay3
test3
go_cue
resp_delay
target
cue
set
ready
measure
f1
f2
offer_on
pre_sure"""

states_list = states_list.split('\n')

short_states = {'timing': dict(zip(states_list, [10]*len(states_list)))}


savpath = os.path.expanduser(f'~/Jan2020/data/{alg}_{task}_{seed}.npz')

import gym
import neurogym  # need to import it so ngym envs are registered
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv

if alg=='A2C':
    from stable_baselines import A2C as algo  # , ACER, ACKTR, PPO2
elif alg=='ACER':
    from stable_baselines import ACER as algo
elif alg=='ACKTR':
    from stable_baselines import ACKTR as algo
elif alg=='PPO2':
    from stable_baselines import PPO2 as algo

env = gym.make(task, {**{'dt': dts}, **shorts_states})
env.seed(seed=seed)
env = DummyVecEnv([lambda: env])
model = algo(LstmPolicy, env, verbose=0, seed=seed, n_steps=n_steps, n_cpu_tf_sess=n_cpu_tf, savpath=savpath) # 1 to have reproducible results
model.learn(total_timesteps=tot_timesteps, seed=seed)
