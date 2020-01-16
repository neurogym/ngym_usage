#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:05:09 2019

@author: manuel
"""
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib
import sys
import os
import glob
import numpy as np
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, ACER, ACKTR, PPO2
import gym
import neurogym  # need to import it so ngym envs are registered
from neurogym import all_tasks
import matplotlib.pyplot as plt
from neurogym.wrappers import trial_hist_nalt as thn
from neurogym.wrappers import cv_learning as cv_l
from neurogym.wrappers import manage_data as md
from priors.codes.ops import utils as ut
matplotlib.use('Qt5Agg')
plt.close('all')


def test_env(env, num_steps=100):
    """Test if all one environment can at least be run."""
    kwargs = {'dt': 100}
    env = gym.make(env, **kwargs)
    env.reset()
    for stp in range(num_steps):
        action = env.action_space.sample()
        state, rew, done, info = env.step(action)
        if done:
            env.reset()
    return env


def get_mean_trial_duration(num_steps=1000):
    num_tr = {}
    for i, env_name in enumerate(sorted(all_tasks.keys())):
        if env_name in ['Combine-v0']:
            continue
        env = test_env(env_name, num_steps=num_steps)
        num_tr[env_name] = env.num_tr
        print(env_name)
        print(env.num_tr)
        print('xxxxxxxxxxxxxxxxxxxx')
    return num_tr


def run_env(algorithm, env, env_args=None, folder='', n_stps_tr=2000000):
    env = DummyVecEnv([lambda: env])
    model = algorithm(LstmPolicy, env, verbose=0)
    model.learn(total_timesteps=n_stps_tr)  # 50000)
    if env_args is not None:
        np.savez(folder + '/params.npz', **env_args)
    env.close()
    plt.close('all')


def run_predefined_envs():
    num_tr = 3
    n_stps_tr = 2000000
    n_tr_sv = 10000
    # trial-history tasks params
    n_ch = 2
    tr_prob = 0.8
    block_dur = 200
    trans = 'RepAlt'
    # dual-task params
    debug = False
    delay = 200
    mix = [.3, .3, .4]
    share_action_space = True
    defaults = [0, 0]
    # cv task parameters
    th = 0.8
    perf_w = 100
    init_ph = 0
    main_folder = '/home/molano/ngym_usage/results/'
    # TODO: include other task, e.g. 'ROMO-hist', 'DELAY-RESPONSE-hist'
    tasks = ['DPA', 'GNG', 'RDM', 'ROMO', 'DELAY-RESPONSE',
             'DELAY-RESPONSE-cv', 'DUAL-TASK', 'RDM-hist']
    # '
    algs_names = ['A2C', 'ACER', 'ACKTR', 'PPO2']
    algs = [A2C, ACER, ACKTR, PPO2]
    for ind_tr in range(num_tr):
        for ind_alg, algorithm in enumerate(algs):
            # RDM
            alg = algs_names[ind_alg]
            for ind_t, task in enumerate(tasks):
                folder = main_folder + task + '_' +\
                    alg + '_' + str(ind_tr) + '/'
                if not os.path.exists(folder):
                    os.makedirs(folder)
                if not os.path.exists(folder + 'params.npz'):
                    print('---------------')
                    print(task)
                    hist = False
                    dual_task = False
                    cv = False
                    if task == 'RDM':
                        timing = {'fixation': [100, 100, 100],
                                  'stimulus': [200, 100, 300],
                                  'delay_btw_stim': [0, 0, 0],  # for clarity
                                  'delay_aft_stim': [0, 0, 0],
                                  'decision': [200, 200, 200]}
                        simultaneous_stim = True
                        gng = False
                        cohs = [0, 6.4, 12.8, 25.6, 51.2]
                        ng_task = 'GenTask-v0'
                    elif task == 'ROMO':
                        timing = {'fixation': [100, 100, 100],
                                  'stimulus': [200, 100, 300],
                                  'delay_btw_stim': [200, 100, 300],
                                  'delay_aft_stim': [0, 0, 0],
                                  'decision': [200, 200, 200]}
                        simultaneous_stim = False  # MAIN CHANGE WRT RDM TASK
                        gng = False
                        cohs = [0, 6.4, 12.8, 25.6, 51.2]
                        ng_task = 'GenTask-v0'
                    elif task == 'DELAY-RESPONSE':
                        timing = {'fixation': [100, 100, 100],
                                  'stimulus': [200, 100, 300],
                                  'delay_btw_stim': [0, 0, 0],
                                  'delay_aft_stim': [200, 100, 300],
                                  'decision': [200, 200, 200]}
                        simultaneous_stim = True
                        gng = False
                        cohs = [0, 6.4, 12.8, 25.6, 51.2]
                        ng_task = 'GenTask-v0'
                    elif task == 'GNG':
                        timing = {'fixation': [100, 100, 100],
                                  'stimulus': [200, 100, 300],
                                  'delay_btw_stim': [0, 0, 0],
                                  'delay_aft_stim': [200, 100, 300],
                                  'decision': [200, 200, 200]}
                        gng = True  # MAIN CHANGE
                        cohs = [100]  # MAIN CHANGE
                        simultaneous_stim = True
                        ng_task = 'GenTask-v0'
                    elif task == 'DPA':
                        timing = {'fixation': [100, 100, 100],
                                  'stimulus': [200, 100, 300],
                                  'delay_btw_stim': [600, 500, 700],
                                  'delay_aft_stim': [200, 100, 300],
                                  'decision': [200, 200, 200]}
                        ng_task = 'DPA-v1'  # MAIN CHANGE
                        simultaneous_stim = False
                        gng = None
                        cohs = None
                    elif task == 'RDM-hist':
                        timing = {'fixation': [100, 100, 100],
                                  'stimulus': [200, 100, 300],
                                  'delay_btw_stim': [0, 0, 0],  # for clarity
                                  'delay_aft_stim': [0, 0, 0],
                                  'decision': [200, 200, 200]}
                        simultaneous_stim = True
                        gng = False
                        cohs = [0, 6.4, 12.8, 25.6, 51.2]
                        ng_task = 'GenTask-v0'
                        hist = True
                    elif task == 'ROMO-hist':
                        timing = {'fixation': [100, 100, 100],
                                  'stimulus': [200, 100, 300],
                                  'delay_btw_stim': [200, 100, 300],
                                  'delay_aft_stim': [0, 0, 0],
                                  'decision': [200, 200, 200]}
                        simultaneous_stim = False  # MAIN CHANGE WRT RDM TASK
                        gng = False
                        cohs = [0, 6.4, 12.8, 25.6, 51.2]
                        ng_task = 'GenTask-v0'
                        hist = True
                    elif task == 'DELAY-RESPONSE-hist':
                        timing = {'fixation': [100, 100, 100],
                                  'stimulus': [200, 100, 300],
                                  'delay_btw_stim': [0, 0, 0],
                                  'delay_aft_stim': [200, 100, 300],
                                  'decision': [200, 200, 200]}
                        simultaneous_stim = True
                        gng = False
                        cohs = [0, 6.4, 12.8, 25.6, 51.2]
                        ng_task = 'GenTask-v0'
                        hist = True
                    elif task == 'DUAL-TASK':
                        timing = {'fixation': [100, 100, 100],
                                  'stimulus': [200, 100, 300],
                                  'delay_btw_stim': [600, 500, 700],
                                  'delay_aft_stim': [200, 100, 300],
                                  'decision': [200, 200, 200]}
                        ng_task = 'DPA-v1'  # MAIN CHANGE
                        simultaneous_stim = False
                        timing_2 = {'fixation': [100, 100, 100],
                                    'stimulus': [100, 100, 100],
                                    'delay_btw_stim': [0, 0, 0],
                                    'delay_aft_stim': [0, 0, 0],
                                    'decision': [100, 100, 100]}
                        gng_2 = True  # MAIN CHANGE
                        cohs_2 = [100]  # MAIN CHANGE
                        simultaneous_stim_2 = True
                        ng_task_2 = 'GenTask-v0'
                        dual_task = True
                    elif task == 'DELAY-RESPONSE-cv':
                        timing = {'fixation': [100, 100, 100],
                                  'stimulus': [200, 100, 300],
                                  'delay_btw_stim': [0, 0, 0],
                                  'delay_aft_stim': [200, 100, 300],
                                  'decision': [200, 200, 200]}
                        simultaneous_stim = True
                        gng = False
                        cohs = [0, 6.4, 12.8, 25.6, 51.2]
                        ng_task = 'GenTask-v0'
                        cv = True
                    else:
                        sys.exit("'NO TASK!!'")
                    # combinations and wrappers
                    if dual_task:
                        params_1 = {'timing': timing}
                        params_2 = {'timing': timing_2, 'gng': gng_2,
                                    'cohs': cohs_2,
                                    'simultaneous_stim': simultaneous_stim_2}
                        env_args = {'env_name_1': ng_task,
                                    'env_name_2': ng_task_2,
                                    'params_1': params_1, 'params_2': params_2,
                                    'delay': delay, 'mix': mix,
                                    'share_action_space': share_action_space,
                                    'defaults': defaults,
                                    'debug': debug}
                        env = gym.make('Combine-v0', **env_args)
                    else:
                        env_args = {'timing': timing, 'gng': gng, 'cohs': cohs,
                                    'simultaneous_stim': simultaneous_stim}
                        env = gym.make(ng_task, **env_args)
                        if hist:
                            thn.TrialHistory_NAlt(env, n_ch=n_ch,
                                                  tr_prob=tr_prob,
                                                  block_dur=block_dur,
                                                  trans=trans)
                            env_args['hist'] = True
                            env_args['n_ch'] = n_ch
                            env_args['tr_prob'] = tr_prob
                            env_args['block_dur'] = block_dur
                            env_args['trans'] = trans
                        if cv:
                            env = cv_l.CurriculumLearning(env, th=th,
                                                          perf_w=perf_w,
                                                          init_ph=init_ph)
                            env_args['cv'] = True
                            env_args['th'] = th
                            env_args['perf_w'] = perf_w
                            env_args['init_ph'] = init_ph
                    env = md.manage_data(env, folder=folder,
                                         num_tr_save=n_tr_sv)
                    # RL
                    run_env(algorithm, env, env_args, folder,
                            n_stps_tr=n_stps_tr)
    # plot
    nc = 100
    # perfs = {alg: np.zeros((len(tasks), num_tr)) for alg in algs_names}
    plt.figure()
    for ind_tr in range(num_tr):
        for ind_alg, algorithm in enumerate(algs):
            alg = algs_names[ind_alg]
            for ind_t, task in enumerate(tasks):
                folder = main_folder + task +\
                    '_' + alg + '_' + str(ind_tr) + '/'
                files = glob.glob(folder + '/*bhvr_data*')
                files = ut.order_by_sufix(files)
                for file in files:
                    data = np.load(file)
                    if 'first_rew' in list(data):
                        plt.plot(np.convolve(data['first_rew'],
                                 np.ones((nc,))/nc, mode='same'))
                    else:
                        plt.plot(np.convolve(data['reward'],
                                 np.ones((nc,))/nc, mode='same'))


def run_original_envs(num_tr=3, n_tr=1000000, n_tr_sv=10000,
                      main_folder=''):
    """Test if all environments can at least be run with baselines-stable."""
    nsts = 1000
    num_tr_in_tasks = get_mean_trial_duration(num_steps=nsts)
#    success_count = 0
    total_count = 0
    algs = [A2C, ACER, ACKTR, PPO2]
    algs_names = ['A2C', 'ACER', 'ACKTR', 'PPO2']

    for ind_tr in range(num_tr):
        for ind_alg, algorithm in enumerate(algs):
            alg = algs_names[ind_alg]
            for ind_env, env_name in enumerate(sorted(all_tasks.keys())):
                if env_name in ['Combine-v0']:
                    continue
                total_count += 1
                folder = main_folder + env_name + '_' + alg + '_' +\
                    str(ind_tr) + '/'
                if not os.path.exists(folder):
                    os.makedirs(folder)
                if not os.path.exists(folder + 'params.npz'):
                    print('Running env: {:s}'.format(env_name))
                    n_stps_tr = int(nsts*n_tr/num_tr_in_tasks[env_name])
                    #                print(num_tr_in_tasks[env_name])
                    #                print(n_stps_tr)
                    # try:
                    kwargs = {'dt': 100}
                    env = gym.make(env_name, **kwargs)
                    env = md.manage_data(env, folder=folder,
                                         num_tr_save=n_tr_sv)
                    kwargs['env'] = env_name
                    kwargs['n_tr_sv'] = n_tr_sv
                    kwargs['n_tr'] = n_tr
                    kwargs['n_stps_tr'] = n_stps_tr
                    env.reset()
                    run_env(algorithm, env, n_stps_tr=n_stps_tr,
                            env_args=kwargs, folder=folder)
#                print('Success')
#                print(env)
#               success_count += 1
#                except BaseException as e:
#                    print('Failure at running env: {:s}'.format(env_name))
#                    print(e)
#            print('Success {:d}/{:d} tasks'.format(success_count,
#                  total_count))


def get_dataset_for_SL(env_name='RDM-v0', n_tr=1000000, dt=100,
                       nstps_test=1000, n_stps_sample=128):
    env = test_env(env_name, num_steps=nstps_test)
    num_steps = int(nstps_test*n_tr/env.num_tr)
    num_samples = int(np.floor(num_steps/n_stps_sample))
    num_steps = n_stps_sample*num_samples
    num_steps_per_trial = int(nstps_test/env.num_tr)
    kwargs = {'dt': dt}
    env = gym.make(env_name, **kwargs)
    env.reset()
    # TODO: this assumes 1-D observations
    samples = np.empty((num_samples, n_stps_sample,
                        env.observation_space.shape[0]))
    target = np.empty((num_samples, n_stps_sample))
    print('Task: ', env_name)
    print('Producing dataset with {0} steps'.format(num_steps) +
          'and {0} trials'.format(n_tr) +
          '({0} steps per trial)'.format(num_steps_per_trial))
    print('Number of samples: ', num_samples)
    for stp in range(num_steps):
        action = env.action_space.sample()
        state, rew, done, info = env.step(action)
        samples[int(stp/n_stps_sample), stp % n_stps_sample, :] = state
        target[int(stp/n_stps_sample), stp % n_stps_sample] = info['gt']
        if stp % (10000*n_stps_sample) == 0:
            print(int(stp/n_stps_sample))

    return samples, target


if __name__ == '__main__':
    batch_size = 128
    num_h = 258
    # run_original_envs(main_folder='/home/molano/ngym_usage/results/')
    samples, target = get_dataset_for_SL(env_name='RDM-v0', n_tr=1000000,
                                         dt=100, nstps_test=1000,
                                         n_stps_sample=batch_size)
    plt.figure()
    plt.imshow(samples[0, :, :].T,  aspect='auto')
    plt.figure()
    plt.plot(target[0, :])

    # from https://www.tensorflow.org/guide/keras/rnn
    model = tf.keras.Sequential()

    # Add a LSTM layer with 128 internal units.
    model.add(layers.LSTM(num_h, input_shape=(1, 3),
                          activation='relu'))

    # Add a Dense layer with 10 units and softmax activation.
    model.add(layers.Dense(3, activation='softmax'))

    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')
    model.fit(samples, target, batch_size=batch_size, epochs=5)
