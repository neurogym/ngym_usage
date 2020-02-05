#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:51:03 2020

@author: molano
"""
import os
import gym
import matplotlib.pyplot as plt
import numpy as np
import neurogym as ngym
from neurogym.envs import ALL_ENVS
from neurogym.utils import plotting
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Input

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
           'SL': True},
      'PostDecisionWager-v0':
          {'timing': {'fixation': ('constant', 100),
                      'stimulus': ('choice', [100, 200, 300]),
                      'delay': ('choice', [100, 200, 300]),
                      'pre_sure': ('choice', [100, 200]),
                      'decision': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': True},
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
           'SL': False},
      'MatchingPenny-v0':
          {'timing': {},
           'loss': 'sparse_categorical_crossentropy',
           'SL': True},
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
           'SL': False},
      'Reaching1DWithSelfDistraction-v0':
          {'timing': {'fixation': ('constant', 200),
                      'reach': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': False},
      'AntiReach-v0':
          {'timing': {'fixation': ('constant', 200),
                      'reach': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': False},
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
           'SL': True},
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
           'SL': False},
      'Detection-v0':
          {'timing': {'fixation': ('constant', 200),
                      'stimulus': ('constant', 200)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': True},
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
           'SL': True},
      'ChangingEnvironment-v0':
          {'timing': {'fixation': ('constant', 200),
                      'stimulus': ('choice', [100, 200, 300]),
                      'decision': ('constant', 100)},
           'loss': 'sparse_categorical_crossentropy',
           'SL': True}}

assert len(list(set(ALL_ENVS)-set(ALL_ENVS_MINIMAL_TIMINGS))) == 0
assert len(list(set(ALL_ENVS_MINIMAL_TIMINGS)-set(ALL_ENVS))) == 0


def run_all_envs(main_folder):
    for ind_t, task in enumerate(ALL_ENVS_MINIMAL_TIMINGS.keys()):
        if ALL_ENVS_MINIMAL_TIMINGS[task]['SL']:
            task_params = ALL_ENVS_MINIMAL_TIMINGS[task]
            task_params['dt'] = 100
            print(task)
            if not os.path.exists(main_folder+'/figs/'+task+'env_struct.png'):
                #                try:
                perf = run_env(task=task, task_params=task_params,
                               main_folder=main_folder)
                ALL_ENVS_MINIMAL_TIMINGS[task]['perf'] = [perf]
                print('Performance: ', perf)
                print('-------------------')
                #                except Exception as e:
                #                    print('Failure in ', task)
                #                    print(e)
            else:
                print('DONE')
    np.savez(main_folder + 'all_data.npz', **ALL_ENVS_MINIMAL_TIMINGS)


def run_env(task, task_params, main_folder, **kwargs):
    """
    task: name of task
    task_params is a dict with items:
        dt: timestep (ms, int)
        timing: duration of periods forming trial (ms)
    main_folder: main folder where the task folder will be stored
    training_params is a dict with items:
        seq_len: rollout (def: 20 timesteps, int)
        num_h: number of units (def: 256 units, int)
        steps_per_epoch: (def: 2000, int)
    """
    folder = main_folder + task + '/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    figs_folder = main_folder + '/figs/'
    if not os.path.exists(figs_folder):
        os.mkdir(figs_folder)

    kwargs = {'dt': task_params['dt'], 'timing': task_params['timing']}
    training_params = {'seq_len': 20, 'num_h': 256, 'steps_per_epoch': 5000,
                       'batch_size': 16}
    training_params.update(kwargs)
    # Make supervised dataset
    dataset = ngym.Dataset(task, env_kwargs=kwargs,
                           batch_size=training_params['batch_size'],
                           seq_len=training_params['seq_len'], cache_len=1e5)
    inputs, targets = dataset()
    env = dataset.env
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.n

    # Model
    # from https://www.tensorflow.org/guide/keras/rnn
    xin = Input(batch_shape=(None, None, obs_size), dtype='float32')
    seq = LSTM(training_params['num_h'], return_sequences=True)(xin)
    mlp = TimeDistributed(Dense(act_size, activation='softmax'))(seq)
    model = Model(inputs=xin, outputs=mlp)
    # model.summary()
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train network
    data_generator = (dataset()
                      for i in range(training_params['steps_per_epoch']))
    model.fit(data_generator, verbose=1,
              steps_per_epoch=training_params['steps_per_epoch'])
    model.save(folder+task)
    # evaluate
    perf = eval_net_in_task(model, task, kwargs, dataset, show_fig=True,
                            folder=figs_folder)
    return perf


def eval_net_in_task(model, env_name, kwargs, dataset, num_trials=1000,
                     show_fig=False, folder='', seed=0, n_stps_plt=100):
    if env_name == 'CVLearning-v0':
        kwargs['init_ph'] = 4
    env = gym.make(env_name, **kwargs)
    env.seed(seed=seed)
    env.reset()
    # first trial
    obs, gt = env.obs, env.gt
    obs = obs[np.newaxis]
    action_pred = model.predict(obs)
    action_pred = np.argmax(action_pred, axis=-1)
    actions_mat = action_pred.T
    gt_test = gt.reshape(-1, 1)
    # perf = 0
    for ind_ep in range(num_trials-1):
        env.new_trial()
        obs, gt = env.obs, env.gt
        obs = obs[np.newaxis]
        action_pred = model.predict(obs)
        action_pred = np.argmax(action_pred, axis=-1)
        # perf += gt[-1] == action_pred[0, -1]
        actions_mat = np.concatenate((actions_mat, action_pred.T), axis=0)
        gt_test = np.concatenate((gt_test, gt.reshape(-1, 1)), axis=0)
    actions_mat = actions_mat[1:]
    # run environment step by step
    env = gym.make(env_name, **kwargs)
    env.seed(seed=seed)
    obs = env.reset()
    perf = []
    actions_plt = []
    observations = []
    rewards = []
    gt_mat = []
    rew_cum = 0
    for ind_stp in range(actions_mat.shape[0]):
        observations.append(obs)
        action = actions_mat[ind_stp]
        obs, rew, _, info = env.step(action)
        rew_cum += rew
        if info['new_trial']:
            perf.append(rew_cum)
            rew_cum = 0
        if show_fig:
            rewards.append(rew)
            gt_mat.append(info['gt'])
            actions_plt.append(action)
    #    print(np.mean(perf))
    #    plt.figure()
    #    plt.plot(gt_test[1:])
    #    plt.plot(gt_mat)
    #    plt.plot(actions_mat, '--')
    #    asdasd
    if show_fig:
        observations = np.array(observations)
        plotting.fig_(obs=observations[:n_stps_plt],
                      actions=actions_plt[:n_stps_plt], gt=gt_mat[:n_stps_plt],
                      rewards=rewards[:n_stps_plt], mean_perf=np.mean(perf),
                      legend=True, name=env_name, folder=folder)
    return np.mean(perf)


if __name__ == '__main__':
    plt.close('all')
    main_folder = '/home/molano/ngym_usage/results/SL_tests/'
    run_all_envs(main_folder=main_folder)

    #    task = 'ContextDecisionMaking-v0'  # 'PerceptualDecisionMaking-v0'
    #    task_params = ALL_ENVS_MINIMAL_TIMINGS[task]
    #    task_params['dt'] = 100

#    run_env(task, task_params, main_folder)

#    task = 'DelayPairedAssociation-v0'
#    task_params = {'timing': {'fixation': ('constant', 0),
#                              'stim1': ('constant', 100),
#                              'delay_btw_stim': ('constant', 200),
#                              'stim2': ('constant', 100),
#                              'delay_aft_stim': ('constant', 100),
#                              'decision': ('constant', 100)},
#                   'dt': 100}
#    run_env(task=task, task_params=task_params)
