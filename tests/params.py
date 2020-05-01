#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 07:36:43 2020

@author: molano
"""


import numpy as np
explore = {'seed': np.arange(3),
           'alg': ['A2C', 'ACER', 'ACKTR', 'PPO2'],
           'n_ch': [2, 4, 8]}

# other
experiment = 'tests'
general_params = {'seed': None, 'alg': None,
                  'task': 'NAltPerceptualDecisionMaking-v0', 'n_lstm': 256,
                  'rollout': 40, 'num_trials': 1000000, 'num_cpu': 20,
                  'run_time': 20}
#
algs = {'A2C': {}, 'ACER': {}, 'ACKTR': {}, 'PPO2': {'nminibatches': 4}}


# task
task_kwargs = {'NAltPerceptualDecisionMaking-v0': {'n_ch': None, 'ob_nch': True,
                                                   'timing': { 'fixation': ('constant', 200),
                                                              'stimulus': ('truncated_exponential', [330, 100, 1000]),
                                                              'decision': ('constant', 500)}}}

# wrappers
wrapps = {'TrialHistory-v0': {'block_dur': 4, 'probs': 0.99},
          'Variable_nch-v0': {'block_nch': 10},
          'PassAction-v0': {},
          'PassReward-v0': {},
          'Monitor-v0': {'folder': '', 'sv_fig': False, 'sv_per': 10000,
                         'fig_type': 'svg'}}  # XXX: monitor always last
