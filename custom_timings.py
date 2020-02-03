#!/usr/bin/env python3
# -*- coding: utf-8 -*-

ALL_ENVS_MINIMAL_TIMINGS = {
    'ContextDecisionMaking-v0':{
        'fixation': ('constant', 200),
        'stimulus': ('constant', 200),
        'delay': ('choice', [200,400,600]),
        'decision': ('constant', 100)
    },
    'DelayedComparison-v0': {
        'fixation': ('choice', (100, 200,300)),
        'f1': ('constant', 200),
        'delay': ('constant', 200),
        'f2': ('constant', 200),
        'decision': ('constant', 100)
    },
    'PerceptualDecisionMaking-v0': {
        'fixation': ('constant', 200),  # TODO: depends on subject
        'stimulus': ('constant', 200),
        'decision': ('constant', 100)
    },
    'EconomicDecisionMaking-v0': {
        'fixation': ('constant', 200),
        'offer_on': ('choice', [100, 200]),
        'decision': ('constant', 100)
    },
    'PostDecisionWager-v0':{
        'fixation': ('constant', 100),  # XXX: not specified
        'stimulus': ('choice', [100, 200, 300]),
        'delay': ('choice', [100,200,300]),
        'pre_sure': ('choice', [100, 200]),
        'decision': ('constant', 100)
    },
    'DelayPairedAssociation-v0': {
        'fixation': ('constant', 0),
        'stim1': ('constant', 200),
        'delay_btw_stim': ('constant', 200),
        'stim2': ('constant', 200),
        'delay_aft_stim': ('constant', 200),
        'decision': ('constant', 100)
    },
    'GoNogo-v0': {
        'fixation': ('constant', 0),
        'stimulus': ('constant', 200),
        'resp_delay': ('constant', 200),
        'decision': ('constant', 100),
    },
    'ReadySetGo-v0': {
        'fixation': ('constant', 200),
        'ready': ('constant', 200),
        'measure': ('choice', [100, 200]),
        'set': ('constant', 100)
    },
    'DelayedMatchSample-v0': {
        'fixation': ('constant', 200),
        'sample': ('constant', 200),
        'delay': ('constant', 200),
        'test': ('constant', 200),
        'decision': ('constant', 100)
    },
    'DelayedMatchCategory-v0':{
        'fixation': ('constant', 200),
        'sample': ('constant', 200),
        'first_delay': ('constant', 200),
        'test': ('constant', 100),
    },
    'DawTwoStep-v0':{ 
    }, 
    'MatchingPenny-v0': { # no timing! ~ empty dict should be handled OK
    },
    'MotorTiming-v0': {
        'fixation': ('constant', 200),  # XXX: not specified
        'cue': ('choice', [100, 200, 300]),
        'set': ('constant', 100)
    },
    'Bandit-v0': { # no timing! ~ empty dict should be handled OK
    },
    'PerceptualDecisionMakingDelayResponse-v0':{
        'fixation': ('constant', 0),
        'stimulus': ('constant', 200),
        'delay': ('choice', [100, 200, 300, 400]),
        'decision': ('constant', 100)
    },
    'NAltPerceptualDecisionMaking-v0': {
        'fixation': ('constant', 200),
        'stimulus': ('choice', [100,200,300]),
        'decision': ('constant', 100)
    },
    'Reaching1D-v0': {
        'fixation': ('constant', 200),
        'reach': ('constant', 100)
    },
    'Reaching1DWithSelfDistraction-v0':{
        'fixation': ('constant', 200),
        'reach': ('constant', 100)
    },
    'AntiReach-v0': {
        'fixation': ('constant', 200),
        'reach': ('constant', 100)
    },
    'DelayedMatchToSampleDistractor1D-v0':{
        'fixation': ('constant', 200),
        'sample': ('constant', 200),
        'delay1': ('constant', 200),
        'test1': ('constant', 100),
        'delay2': ('constant', 200),
        'test2': ('constant', 100),
        'delay3': ('constant', 200),
        'test3': ('constant', 100)
    },
    'IntervalDiscrimination-v0':{
        'fixation': ('constant', 200),
        'stim1': ('choice', [100, 200, 300]),
        'delay1': ('choice', [200, 400]),
        'stim2': ('choice', [100, 200, 300]),
        'delay2': ('constant', 200),
        'decision': ('constant', 100)
    },
    'AngleReproduction-v0':{
        'fixation': ('constant', 200),
        'stim1': ('constant', 200),
        'delay1': ('constant', 200),
        'stim2': ('constant', 200),
        'delay2': ('constant', 200),
        'go1': ('constant', 100),
        'go2': ('constant', 100)
    },
    'Detection-v0':{ # beware hidden delay within stimulus epoch
        'fixation': ('constant', 200),
        'stimulus': ('constant', 200)
    },
    'ReachingDelayResponse-v0': {
        'stimulus': ('constant', 100),
        'delay': ('choice', [0, 100, 200]),
        'decision': ('constant', 100)
    },
    'CVLearning-v0':{
        'fixation': ('constant', 200),
        'stimulus': ('constant', 200),
        'delay': ('choice', [100, 200, 300]),
        'decision': ('constant', 100)
    },
    'ChangingEnvironment-v0':{
        'fixation': ('constant', 200),
        'stimulus': ('choice', [100, 200, 300]),
        'decision': ('constant', 100)
    }
}

all_tasks_bsc_timings = {
    'Mante-v0': 
        {
            'fixation': ('constant', 200),
            'target': ('constant', 200),  # TODO: not implemented
            'stimulus': ('constant', 200),
            'delay': ('choice', [100, 200]),
            'decision': ('constant', 100)
        },
    'Romo-v0': 
    {
            'fixation': ('choice', (100, 200, 300)),
            'f1': ('constant', 200),
            'delay': ('constant', 200),
            'f2': ('constant', 200),
            'decision': ('constant', 100)
    },
    'RDM-v0': 
    {
        'fixation': ('constant', 100),  # TODO: depends on subject
        'stimulus': ('constant', 200),
        'decision': ('constant', 100)
    },
    'padoaSch-v0': {
        'fixation': ('constant', 200),
        'offer_on': ('choice', [100, 200]),
        'decision': ('constant', 100)
    },
    'pdWager-v0': {
        'fixation': ('constant', 100),  # XXX: not specified
        # 'target':  ('constant', 0), # XXX: not implemented, not specified
        'stimulus': ('choice', [100, 200, 300, 400]),
        'delay': ('choice', [100, 200, 300, 400]),
        'pre_sure': ('choice', [100, 200]),
        'decision': ('constant', 100)  # XXX: not specified
    },
    'DPA-v0': {
        'fixation': ('constant', 0),
        'stim1': ('constant', 200),
        'delay_btw_stim': ('constant', 200),
        'stim2': ('constant', 200),
        'delay_aft_stim': ('constant', 200),
        'decision': ('constant', 100)
    },
    'GNG-v0': {
        'fixation': ('constant', 0),
        'stimulus': ('constant', 200),
        'resp_delay': ('constant', 200),
        'decision': ('constant', 100)
    },
    'ReadySetGo-v0': {
        'fixation': ('constant', 200),
        'ready': ('constant', 200),
        'measure': ('choice', [100, 200]),
        'set': ('constant', 100)
    },
    'DelayedMatchSample-v0': {
        'fixation': ('constant', 200),
        'sample': ('constant', 200),
        'delay': ('constant', 200),
        'test': ('constant', 200),
        'decision': ('constant', 100)
    },
    'DelayedMatchCategory-v0':{
        'fixation': ('constant', 200),
        'sample': ('constant', 200),
        'first_delay': ('constant', 200),
        'test': ('constant', 100),
    },
    'DawTwoStep-v0':{ 
    }, 
    'MatchingPenny-v0': { # no timing! ~ empty dict should be handled OK
    },
    'MotorTiming-v0': {
        'fixation': ('constant', 200),  # XXX: not specified
        'cue': ('choice', [100, 200, 300]),
        'set': ('constant', 100)
    },
    'Bandit-v0': { # no timing! ~ empty dict should be handled OK
    },
    'DelayedResponse-v0': {
        'fixation': ('constant', 0),
        'stimulus': ('constant', 200),
        'delay': ('choice', [100, 200, 300]), # prunning original: [(2**x)*100 for x in range(8)] 
        'go_cue': ('constant', 200),
        'decision': ('constant', 100)
    },
    'NAltRDM-v0': {
        'fixation': ('constant', 200),
        'stimulus': ('choice', [100,200,300]),
        'decision': ('constant', 100)
    },
    # 'GenTask-v0': 'neurogym.envs.generaltask:GenTask',
    'Combine-v0': { # this is not a task
    },
    # 'IBL-v0': 'neurogym.envs.ibl:IBL',
    #'MemoryRecall-v0': {} # not working atm (29thJan2020),
    'Reaching1D-v0': {
        'fixation': ('constant', 200),
        'reach': ('constant', 100)
    },
    'Reaching1DWithSelfDistraction-v0':{
        'fixation': ('constant', 200),
        'reach': ('constant', 100)
    },
    'AntiReach-v0': {
        'fixation': ('constant', 200),
        'reach': ('constant', 100)
    },
    'DelayedMatchToSampleDistractor1D-v0':{
        'fixation': ('constant', 200),
        'sample': ('constant', 200),
        'delay1': ('constant', 200),
        'test1': ('constant', 100),
        'delay2': ('constant', 200),
        'test2': ('constant', 100),
        'delay3': ('constant', 200),
        'test3': ('constant', 100)
    },
    'IntervalDiscrimination-v0':{
        'fixation': ('constant', 200),
        'stim1': ('choice', [100, 200, 300]),
        'delay1': ('choice', [200, 400]),
        'stim2': ('choice', [100, 200, 300]),
        'delay2': ('constant', 200),
        'decision': ('constant', 100)
    },
    'AngleReproduction-v0':{
        'fixation': ('constant', 200),
        'stim1': ('constant', 200),
        'delay1': ('constant', 200),
        'stim2': ('constant', 200),
        'delay2': ('constant', 200),
        'go1': ('constant', 100),
        'go2': ('constant', 100)
    },
    'Detection-v0':{ # beware hidden delay within stimulus epoch
        'fixation': ('constant', 200),
        'stimulus': ('constant', 200)
    },
    'Serrano-v0':{
        'stimulus': ('constant', 100),
        'delay': ('choice', [0, 100, 200]),
        'decision': ('constant', 100)
    },
    'CVLearning-v0':{
        'fixation': ('constant', 200),
        'stimulus': ('constant', 200),
        'delay': ('choice', [100, 200, 300]),
        'decision': ('constant', 100)
    },
    'ChangingEnvironment-v0':{
        'fixation': ('constant', 200),
        'stimulus': ('choice', [100, 200, 300]),
        'decision': ('constant', 100)
    },
    'ReachingDelayResponse-v0': {
        'stimulus': ('constant', 100),
        'delay': ('choice', [0, 100, 200]),
        'decision': ('constant', 100)
    },
}