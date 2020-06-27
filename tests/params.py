#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 07:36:43 2020

@author: molano
"""


import numpy as np
# each Node has 4 GPUs, each with 40 cpus
# here we do: 8 seed x 2 n_ch x 10 num_cpus
# num_thrds is the number of agents to run
explore = {'seed': np.arange(0, 8),
           'alg': ['ACER'],
           'n_ch': [8]}

# other
experiment = 'diff_probs_nch_blocks'
general_params = {'seed': None, 'alg': None,
                  'task': 'NAltPerceptualDecisionMaking-v0', 'n_lstm': 256,
                  'rollout': 40, 'num_trials': 20000000, 'num_cpu': 20,
                  'run_time': 24, 'num_thrds': 20}
#
algs = {'A2C': {}, 'ACER': {}, 'ACKTR': {}, 'PPO2': {'nminibatches': 4}}


# task
task_kwargs = {'NAltPerceptualDecisionMaking-v0': {'n_ch': None, 'ob_nch': True,
                                                   'timing': { 'fixation': ('constant', 100),
                                                              'stimulus': ('truncated_exponential', [200, 100, 500]),
                                                              'decision': ('constant', 300)}}}

# wrappers
wrapps = {'TrialHistory-v0': {'block_dur': 200, 'probs': 0.9, 'num_blocks': 3},
          'Variable_nch-v0': {'block_nch': 1000, 'blocks_probs': [0.07246752, 0.08851203, 0.10810883, 0.13204443, 0.16127943, 0.19698714, 0.24060063]},  # np.flip(np.exp(-np.arange(7)/5))
          'PassAction-v0': {},
          'PassReward-v0': {},
          'Monitor-v0': {'folder': '', 'sv_fig': False, 'sv_per': 100000,
                         'fig_type': 'svg'}}  # XXX: monitor always last
