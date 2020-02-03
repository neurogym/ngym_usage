#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

ALL_WRAPPERS_MINIMAL_RL = {
    'CatchTrials-v0': {}, # default should work
    # 'Monitor-v0': {}, # this will work always, so no extra params 
    'Noise-v0': {},
    'PassReward-v0': {},
    'PassAction-v0': {},
    'ReactionTime-v0': {}, # broken
    'SideBias-v0': {'prob':np.array([[0.75, 0.25], [0.25, 0.75]])},
    'TrialHistory-v0': {}
    # 'MissTrialReward-v0': 'neurogym.wrappers.miss_trials_reward:MissTrialReward', # obsolete
    # 'TTLPulse-v0': 'neurogym.wrappers.ttl_pulse:TTLPulse',
    #'Combine-v0': {}'
}