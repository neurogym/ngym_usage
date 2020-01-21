#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:05:09 2019

@author: manuel
"""
import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Input
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
# from priors.codes.ops import utils as ut
matplotlib.use('Qt5Agg')
plt.close('all')
loss_functions = {'AntiReach-v0': 'mean_squared_error',
                  'Bandit-v0': 'sparse_categorical_crossentropy',
                  'DPA-v0': 'sparse_categorical_crossentropy',
                  'DawTwoStep-v0': 'sparse_categorical_crossentropy',
                  'DelayedMatchCategory-v0': 'sparse_categorical_crossentropy',
                  'DelayedMatchSample-v0': 'sparse_categorical_crossentropy',
                  'DelayedMatchToSampleDistractor1D-v0':
                      'sparse_categorical_crossentropy',
                  'DelayedResponse-v0': 'sparse_categorical_crossentropy',
                  'GNG-v0': 'sparse_categorical_crossentropy',
                  'Mante-v0': 'sparse_categorical_crossentropy',
                  'MatchingPenny-v0': 'sparse_categorical_crossentropy',
                  'MemoryRecall-v0': 'sparse_categorical_crossentropy',
                  'MotorTiming-v0': 'mean_squared_error',
                  'NAltRDM-v0': 'sparse_categorical_crossentropy',
                  'RDM-v0': 'categorical_crossentropy',
                  'Reaching1D-v0': 'mean_squared_error',
                  'Reaching1DWithSelfDistraction-v0': 'mean_squared_error',
                  'ReadySetGo-v0': 'no-SL',
                  'Romo-v0': 'sparse_categorical_crossentropy',
                  'padoaSch-v0': 'sparse_categorical_crossentropy',
                  'pdWager-v0': 'sparse_categorical_crossentropy'}
ROLLOUT = 100
KWARGS = {'dt': 100, 'stimEv': 1000,
          'timing': {'stimulus': ('constant', 200),
                     'decision': ('constant', 100)}}


def test_env(env, num_steps=100):
    """Test if all one environment can at least be run."""
    env = gym.make(env, **KWARGS)
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


def run_env(algorithm, env, n_stps_tr=2000000, verbose=0):
    env = DummyVecEnv([lambda: env])
    model = algorithm(LstmPolicy, env, verbose=verbose)
    model.learn(total_timesteps=n_stps_tr)
    return model


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
                            env = thn.TrialHistory_NAlt(env, n_ch=n_ch,
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


def run_original_envs(main_folder='', **kwargs):
    """
    Test if all environments can at least be learn with sup. learning
    """
    params = {'num_instances': 3, 'num_h': 256, 'b_size': 128,
              'num_tr': 200000, 'tr_per_ep': 100, 'dt': 100}
    params.update(kwargs)
    success_count = 0
    total_count = 0
    for ind_tr in range(params['num_instances']):
        for ind_env, env_name in enumerate(sorted(all_tasks.keys())):
            if env_name in ['Combine-v0']:
                continue
            total_count += 1
            if (env_name in loss_functions.keys() and
               loss_functions[env_name] != 'no-SL'):
                folder = main_folder + env_name + '_' + str(ind_tr) + '/'
                if not os.path.exists(folder):
                    os.makedirs(folder)
                if not os.path.exists(folder + 'params.npz'):
                    print('Running env: {:s}'.format(env_name))
                    # try:
                    train_env_keras_net(env_name, folder,
                                        num_h=params['num_h'],
                                        b_size=params['b_size'],
                                        num_tr=params['num_tr'],
                                        tr_per_ep=params['tr_per_ep'],
                                        dt=params['dt'])
                    print('Success')
                    success_count += 1
                    np.savez(folder + 'params.npz', **params)
    #                except BaseException as e:
    #                print('Failure at running env: {:s}'.format(env_name))
    #                print(e)
    print('Success {:d}/{:d} tasks'.format(success_count,
          total_count))


def get_dataset_for_SL(env_name='RDM-v0', n_tr=1000000,
                       nstps_test=1000, verbose=0, seed=None):
    env = test_env(env_name, num_steps=nstps_test)
    obs_sh = env.observation_space.shape[0]
    act_sh = env.action_space.n
    num_steps = int(nstps_test*n_tr/(env.num_tr))
    num_steps_per_trial = int(nstps_test/env.num_tr)
    env = gym.make(env_name, **KWARGS)
    env.seed(seed)
    env.reset()
    # TODO: this assumes 1-D observations
    samples = np.empty((num_steps, obs_sh))
    target = np.empty((num_steps, act_sh))
    if verbose:
        print('Task: ', env_name)
        print('Producing dataset with {0} steps'.format(num_steps) +
              ' and {0} trials'.format(n_tr) +
              ' ({0} steps per trial)'.format(num_steps_per_trial))
    count_stps = 0
    for tr in range(n_tr):
        obs = env.obs
        gt = env.gt
        samples[count_stps:count_stps+obs.shape[0], :] = obs
        target[count_stps:count_stps+gt.shape[0], :] = np.eye(act_sh)[gt]
        count_stps += obs.shape[0]
        assert obs.shape[0] == gt.shape[0]
        env.new_trial()

    samples = samples[:count_stps, :]
    target = target[:count_stps, :]
    samples = samples.reshape((-1, ROLLOUT, obs_sh))
    target = target.reshape((-1, ROLLOUT, act_sh))
#    print(samples.shape)
#    print(target.shape)
#    plt.figure()
#    plt.subplot(2, 1, 1)
#    plt.imshow(samples[0, :, :].T, aspect='auto')
#    plt.subplot(2, 1, 2)
#    plt.imshow(target[0, :, :].T, aspect='auto')
#    asd
    return samples, target, env


def train_env_keras_net(env_name, folder, num_h=256, b_size=128,
                        num_tr=200000, tr_per_ep=1000, verbose=1):
    env = test_env(env_name, num_steps=1)
    # from https://www.tensorflow.org/guide/keras/rnn
    xin = Input(batch_shape=(None, ROLLOUT, env.observation_space.shape[0]),
                dtype='float32')
    seq = LSTM(num_h, return_sequences=True)(xin)
    mlp = TimeDistributed(Dense(env.action_space.n, activation='softmax'))(seq)
    model = Model(inputs=xin, outputs=mlp)
    model.summary()
    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    num_ep = int(num_tr/tr_per_ep)
    loss_training = []
    acc_training = []
    perf_training = []
    for ind_ep in range(num_ep):
        start_time = time.time()
        # train
        samples, target, _ = get_dataset_for_SL(env_name=env_name,
                                                n_tr=tr_per_ep)
        model.fit(samples, target, epochs=1, verbose=0)
        # test
        samples, target, env = get_dataset_for_SL(env_name=env_name,
                                                  n_tr=tr_per_ep, seed=ind_ep)
        loss, acc = model.evaluate(samples, target, verbose=0)
        loss_training.append(loss)
        acc_training.append(acc)
        perf = eval_net_in_task(model, env_name=env_name,
                                tr_per_ep=tr_per_ep, samples=samples,
                                target=target, folder=folder,
                                show_fig=(ind_ep % 100) == 0, seed=ind_ep)
        perf_training.append(perf)
        if verbose and ind_ep % 100 == 0:
            print('Accuracy: ', acc)
            print('Performance: ', perf)
            rem_time = (num_ep-ind_ep)*(time.time()-start_time)/3600
            print('epoch {0} out of {1}'.format(ind_ep, num_ep))
            print('remaining time: {:.2f}'.format(rem_time))
            print('-------------')

    data = {'acc': acc_training, 'loss': loss_training,
            'perf': perf_training}
    np.savez(folder + 'training.npz', **data)
    fig = plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(acc_training)
    plt.subplot(1, 3, 2)
    plt.plot(loss_training)
    plt.subplot(1, 3, 3)
    plt.plot(perf_training)

    fig.savefig(folder + 'performance.png')
    plt.close(fig)
    return model


def eval_net_in_task(model, env_name, tr_per_ep, sl=True, samples=None,
                     target=None, seed=0, show_fig=False, folder=''):
    if samples is None:
        samples, target, _ = get_dataset_for_SL(env_name=env_name,
                                                n_tr=tr_per_ep, seed=seed)
        if show_fig:
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.imshow(samples[0, :, :].T, aspect='auto')
            plt.subplot(2, 1, 2)
            plt.imshow(target[0, :]. T, aspect='auto')
            plt.tight_layout()

    if sl:
        actions = model.predict(samples)
    env = gym.make(env_name, **KWARGS)
    env.seed(seed=seed)
    obs = env.reset()
    perf = []
    actions_plt = []
    rew_temp = []
    observations = []
    rewards = []
    gt = []
    target_mat = []
    action = 0

    for ind_act in range(tr_per_ep):
        index = ind_act + 1
        observations.append(obs)
        if sl:
            action = actions[int(np.floor(index/ROLLOUT)),
                             (index % ROLLOUT), :]
            action = np.argmax(action)
        else:
            action, _ = model.predict([obs])
        obs, rew, _, info = env.step(action)
        if info['new_trial']:
            perf.append(rew)
        if show_fig:
            rew_temp.append(rew)
            rewards.append(rew)
            gt.append(info['gt'])
            target_mat.append(target[int(np.floor(index/ROLLOUT)),
                                     index % ROLLOUT])
            actions_plt.append(action)

    if show_fig:
        n_stps_plt = 100
        observations = np.array(observations)
        f = plt.figure()
        plt.subplot(3, 1, 1)
        plt.imshow(observations[:n_stps_plt, :].T, aspect='auto')
        plt.title('observations')
        plt.subplot(3, 1, 2)
        plt.plot(np.arange(n_stps_plt)+1, actions_plt[:n_stps_plt], marker='+')
        gt = np.array(gt)
        if len(gt.shape) == 2:
            gt = np.argmax(gt, axis=1)
        plt.plot(np.arange(n_stps_plt)+1, gt[:n_stps_plt], 'r')
        # plt.plot(np.arange(n_stps_plt)+1, target_mat[:n_stps_plt], '--y')
        plt.title('actions')
        plt.xlim([-0.5, n_stps_plt+0.5])
        plt.subplot(3, 1, 3)
        plt.plot(np.arange(n_stps_plt)+1, rewards[:n_stps_plt], 'r')
        plt.title('reward')
        plt.xlim([-0.5, n_stps_plt+0.5])
        plt.title(str(np.mean(perf)))
        plt.tight_layout()
        plt.show()
        if folder != '':
            f.savefig(folder + 'task_struct.png')
            plt.close(f)

    return np.mean(perf)


if __name__ == '__main__':
    sl = True
    main_folder = '/home/molano/ngym_usage/results/'
    env_name = 'RDM-v0'
    ROLLOUT = 20
    KWARGS = {'dt': 100,
              'timing': {'decision': ('constant', 100)}}
    if sl:
        # Supervised Learning
        model = train_env_keras_net(env_name, main_folder + '/tests_short/',
                                    num_h=256, b_size=128, num_tr=500000,
                                    tr_per_ep=1000, verbose=1)
        eval_net_in_task(model, 'RDM-v0', tr_per_ep=100, show_fig=True)
    else:
        # RL
        env = gym.make(env_name, **KWARGS)
        obs = env.reset()
        # model = run_env(A2C, env, n_stps_tr=200000, verbose=0)
        eval_net_in_task(model, 'RDM-v0', tr_per_ep=100, show_fig=True, sl=sl)
