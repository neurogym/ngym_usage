#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from neurogym.envs import ALL_ENVS

ALL_ENVS_MINIMAL_TIMINGS =\
     {'ContextDecisionMaking-v0':
         {'timing': {'fixation': ('constant', 200),
                     'stimulus': ('constant', 200),
                     'delay': ('choice', [200, 400, 600]),
                     'decision': ('constant', 100)},
          'loss': 'sparse_categorical_crossentropy',
          'SL': True},
      'DelayedComparison-v0':
          {'timing': {'fixation': ('choice', (100, 200, 300)),
                      'f1': ('constant', 200),
                      'delay': ('constant', 200),
                      'f2': ('constant', 200),
                      'decision': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': True},
      'PerceptualDecisionMaking-v0':
          {'timing': {'fixation': ('constant', 200),
                      'stimulus': ('constant', 200),
                      'decision': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': True},
      'EconomicDecisionMaking-v0':
          {'timing': {'fixation': ('constant', 200),
                      'offer_on': ('choice', [100, 200]),
                      'decision': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': False},  # ground truth not implemented
      'PostDecisionWager-v0':
          {'timing': {'fixation': ('constant', 100),
                      'stimulus': ('choice', [100, 200, 300]),
                      'delay': ('choice', [100, 200, 300]),
                      'pre_sure': ('choice', [100, 200]),
                      'decision': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': False},  # task to test confidence
      'DelayPairedAssociation-v0':
          {'timing': {'fixation': ('constant', 0),
                      'stim1': ('constant', 200),
                      'delay_btw_stim': ('constant', 200),
                      'stim2': ('constant', 200),
                      'delay_aft_stim': ('constant', 200),
                      'decision': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': True},
      'GoNogo-v0':
          {'timing': {'fixation': ('constant', 0),
                      'stimulus': ('constant', 200),
                      'resp_delay': ('constant', 200),
                      'decision': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': True},
      'ReadySetGo-v0':
          {'timing': {'fixation': ('constant', 200),
                      'ready': ('constant', 200),
                      'measure': ('choice', [100, 200]),
                      'set': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': True},
      'DelayedMatchSample-v0':
          {'timing': {'fixation': ('constant', 200),
                      'sample': ('constant', 200),
                      'delay': ('constant', 200),
                      'test': ('constant', 200),
                      'decision': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': True},
      'DelayedMatchCategory-v0':
          {'timing': {'fixation': ('constant', 200),
                      'sample': ('constant', 200),
                      'first_delay': ('constant', 200),
                      'test': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': True},
      'DawTwoStep-v0':
          {'timing': {},
           'loss': 'sparse_categorical_crossentropy',
           'SL': False},  # observation depends on action
      'MatchingPenny-v0':
          {'timing': {},
           'loss': 'sparse_categorical_crossentropy',
           'SL': False},  # either responses depend on actns or task is trivial
      'MotorTiming-v0':
          {'timing': {'fixation': ('constant', 200),
                      'cue': ('choice', [100, 200, 300]),
                      'set': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': True},
      'Bandit-v0':
          {'timing': {},
           'loss': 'sparse_categorical_crossentropy',
           'SL': True},
      'PerceptualDecisionMakingDelayResponse-v0':
          {'timing': {'fixation': ('constant', 0),
                      'stimulus': ('constant', 200),
                      'delay': ('choice', [100, 200, 300, 400]),
                      'decision': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': True},
      'NAltPerceptualDecisionMaking-v0':
          {'timing': {'fixation': ('constant', 200),
                      'stimulus': ('choice', [100, 200, 300]),
                      'decision': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': True},
      'Reaching1D-v0':
          {'timing': {'fixation': ('constant', 200),
                      'reach': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': False},  # actions have incremental effect
      'Reaching1DWithSelfDistraction-v0':
          {'timing': {'fixation': ('constant', 200),
                      'reach': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': False},  # actions have incremental effect
      'AntiReach-v0':
          {'timing': {'fixation': ('constant', 200),
                      'reach': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': False},  # actions have incremental effect
      'DelayedMatchToSampleDistractor1D-v0':
          {'timing': {'fixation': ('constant', 200),
                      'sample': ('constant', 200),
                      'delay1': ('constant', 200),
                      'test1': ('constant', 100),
                      'delay2': ('constant', 200),
                      'test2': ('constant', 100),
                      'delay3': ('constant', 200),
                      'test3': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': True},  # trials end when agent responds
      'IntervalDiscrimination-v0':
          {'timing': {'fixation': ('constant', 200),
                      'stim1': ('choice', [100, 200, 300]),
                      'delay1': ('choice', [200, 400]),
                      'stim2': ('choice', [100, 200, 300]),
                      'delay2': ('constant', 200),
                      'decision': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': True},
      'AngleReproduction-v0':
          {'timing': {'fixation': ('constant', 200),
                      'stim1': ('constant', 200),
                      'delay1': ('constant', 200),
                      'stim2': ('constant', 200),
                      'delay2': ('constant', 200),
                      'go1': ('constant', 100),
                      'go2': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': False},  # actions have incremental effect
      'Detection-v0':
          {'timing': {'fixation': ('constant', 200),
                      'stimulus': ('constant', 200)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': True},  # trials end when agent responds
      'ReachingDelayResponse-v0':
          {'timing':
              {'stimulus': ('constant', 100),
               'delay': ('choice', [0, 100, 200]),
               'decision': ('constant', 100)},
           'loss': 'mean_squared_error',
           'SL': False},
      'CVLearning-v0':
          {'timing': {'fixation': ('constant', 200),
                      'stimulus': ('constant', 200),
                      'delay': ('choice', [100, 200, 300]),
                      'decision': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': True},  # block task
      'ChangingEnvironment-v0':
          {'timing': {'fixation': ('constant', 200),
                      'stimulus': ('choice', [100, 200, 300]),
                      'decision': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': True}}  # block task

assert len(list(set(ALL_ENVS)-set(ALL_ENVS_MINIMAL_TIMINGS))) == 0
assert len(list(set(ALL_ENVS_MINIMAL_TIMINGS)-set(ALL_ENVS))) == 0
