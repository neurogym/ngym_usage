#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue April 17  16:09:16 2020

@author: molano
"""


import numpy as np
all_tasks = ['ContextDecisionMaking-v0',
             'DelayedComparison-v0',
             'PerceptualDecisionMaking-v0',
             'EconomicDecisionMaking-v0',
             'PostDecisionWager-v0',
             'DelayPairedAssociation-v0',
             'GoNogo-v0',
             'ReadySetGo-v0',
             'OneTwoThreeGo-v0',
             'DelayedMatchSample-v0',
             'DelayedMatchCategory-v0',
             'DawTwoStep-v0',
             'HierarchicalReasoning-v0',
             'MatchingPenny-v0',
             'MotorTiming-v0',
             'MultiSensoryIntegration-v0',
             'Bandit-v0',
             'PerceptualDecisionMakingDelayResponse-v0',
             'NAltPerceptualDecisionMaking-v0',
             'Reaching1D-v0',
             'Reaching1DWithSelfDistraction-v0',
             'AntiReach-v0',
             'DelayedMatchToSampleDistractor1D-v0',
             'IntervalDiscrimination-v0',
             'AngleReproduction-v0',
             'Detection-v0',
             'ReachingDelayResponse-v0',
             'ChangingEnvironment-v0',
             'ProbabilisticReasoning-v0',
             'DualDelayedMatchSample-v0',
             'PulseDecisionMaking-v0']
explore = {'seed': np.arange(1),
           'alg': ['A2C', 'ACER', 'ACKTR', 'PPO2'],
           'task': all_tasks}

# other
experiment = 'RL_training'
general_params = {'seed': None, 'alg': None,
                  'task': None,
                  'rollout': 40, 'num_trials': 1000000, 'num_cpu': 20,
                  'run_time': 20}
#
algs = {'A2C': {}, 'ACER': {}, 'ACKTR': {}, 'PPO2': {'nminibatches': 4}}


# task
task_kwargs = {k: {} for k in all_tasks}

# wrappers
wrapps = {'Monitor-v0': {'folder': '', 'sv_fig': False, 'sv_per': 10000,
                         'fig_type': 'svg'}}
