#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 07:36:43 2020

@author: molano
"""


import numpy as np
explore = {'seed': np.arange(1),
           'alg': ['A2C', 'ACER', 'ACKTR', 'PPO2'],
           'n_ch': [2, 4, 8, 16]}

# other
experiment = 'tests'
general_params = {'seed': None, 'alg': None,
                  'task': 'NAltPerceptualDecisionMaking-v0', 'n_lstm': 256,
                  'rollout': 40, 'num_trials': 1000000, 'num_cpu': 20,
                  'run_time': 20}
#
algs = {'A2C': {}, 'ACER': {}, 'ACKTR': {}, 'PPO2': {'nminibatches': 4}}


# task
task_kwargs = {'NAltPerceptualDecisionMaking-v0': {'n_ch': None, 'timing': {
                'fixation': ('constant', 200),
                'stimulus': ('truncated_exponential', [330, 100, 1000]),
                'decision': ('constant', 500)}}}

# wrappers
wrapps = {'Monitor-v0': {'folder': '', 'sv_fig': False, 'sv_per': 10000,
                         'fig_type': 'svg'}}
