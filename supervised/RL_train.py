#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 09:33:08 2020

@author: manuel
"""

import os
from pathlib import Path
import json
import importlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from neurogym.wrappers import ALL_WRAPPERS
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.callbacks import CheckpointCallback
import gym
import glob
import neurogym as ngym


def get_modelpath(envid):
    # Make a local file directories
    path = Path('.') / 'files'
    os.makedirs(path, exist_ok=True)
    path = path / envid
    os.makedirs(path, exist_ok=True)
    return path


def apply_wrapper(env, wrap_string, params):
    wrap_str = ALL_WRAPPERS[wrap_string]
    wrap_module = importlib.import_module(wrap_str.split(":")[0])
    wrap_method = getattr(wrap_module, wrap_str.split(":")[1])
    return wrap_method(env, **params)


def make_env(env_id, rank, seed=0, wrapps={}, **kwargs):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """
    def _init():
        env = gym.make(env_id, **kwargs)
        env.seed(seed + rank)
        for wrap in wrapps.keys():
            if not (wrap == 'MonitorExtended-v0' and rank != 0):
                env = apply_wrapper(env, wrap, wrapps[wrap])
        return env
    set_global_seeds(seed)
    return _init


def get_alg(alg):
    if alg == "A2C":
        from stable_baselines import A2C as algo
    elif alg == "ACER":
        from stable_baselines import ACER as algo
    elif alg == "ACKTR":
        from stable_baselines import ACKTR as algo
    elif alg == "PPO2":
        from stable_baselines import PPO2 as algo
    return algo

def train_network(envid, alg='ACER', num_steps=100000, seed=0, wrappers_kwargs={},
                  task_kwargs={}, n_thrds=1, rollout=20, n_lstm=64, alg_kwargs={}):
    """Supervised training networks.

    Save network in a path determined by environment ID.

    Args:
        envid: str, environment ID.
    """
    modelpath = get_modelpath(envid)
    config = {
        'dt': 100,
        'hidden_size': n_lstm,
        'lr': 1e-2,
        'alg': alg,
        'rollout': rollout,
        'n_thrds': n_thrds,
        'wrappers_kwargs': wrappers_kwargs,
        'seed': seed,
        'envid': envid,
    }

    env_kwargs = {'dt': config['dt']}
    env_kwargs.update(task_kwargs)
    config['env_kwargs'] = env_kwargs

    # Save config
    with open(modelpath / 'config.json', 'w') as f:
        json.dump(config, f)
    algo = get_alg(alg)
    # Make supervised dataset
    env = SubprocVecEnv([make_env(env_id=envid, rank=i, seed=seed,
                                  wrapps=wrappers_kwargs, **env_kwargs)
                         for i in range(n_thrds)])
    model = algo(LstmPolicy, env, verbose=0, n_steps=rollout,
                 n_cpu_tf_sess=n_thrds, tensorboard_log=None,
                 policy_kwargs={"feature_extraction": "mlp",
                                "n_lstm": n_lstm}, **alg_kwargs)
    chckpnt_cllbck = CheckpointCallback(save_freq=int(num_steps/10),
                                        save_path=modelpath,
                                        name_prefix='model')
    model.learn(total_timesteps=num_steps, callback=chckpnt_cllbck)
    model.save(modelpath / 'model_{num_steps}_steps.zip')
    print('Finished Training')


def infer_test_timing(env):
    """Infer timing of environment for testing."""
    timing = {}
    for period in env.timing.keys():
        period_times = [env.sample_time(period) for _ in range(100)]
        timing[period] = np.median(period_times)
    return timing


def extend_obs(ob, num_threads):
    sh = ob.shape
    return np.concatenate((ob, np.zeros((num_threads-sh[0], sh[1]))))


def run_network(envid, num_steps=10**5):
    """Run trained networks for analysis.

    Args:
        envid: str, Environment ID

    Returns:
        activity: a list of activity matrices
        info: pandas dataframe, each row is information of a trial
        config: dict of network, training configurations
    """
    modelpath = get_modelpath(envid)
    files = glob.glob(modelpath+'/*')
    if len(files) > 0:
        with open(modelpath / 'config.json') as f:
            config = json.load(f)
        env_kwargs = config['env_kwargs']
        wrappers_kwargs = config['wrappers_kwargs']
        seed = config['seed']
        # Run network to get activity and info
        sorted_models, last_model = order_by_sufix(files)
        model_name = sorted_models[-1]
        algo = get_alg(config['alg'])
        model = algo.load(modelpath+'/'+model_name, tensorboard_log=None,
                          custom_objects={'verbose': 0})

        # Environment
        env = make_env(env_id=envid, rank=0, seed=seed, wrapps=wrappers_kwargs,
                       **env_kwargs)()
        env.timing = infer_test_timing(env)
        env.reset(no_step=True)
        # Instantiate the network and print information
        activity = list()
        info = pd.DataFrame()
        state_mat = []
        gt = []
        ob = env.reset()
        _states = None
        done = False
        info_df = pd.DataFrame()
        for stp in range(int(num_steps)):
            ob = np.reshape(ob, (1, ob.shape[0]))
            done = [done] + [False for _ in range(config['num_threads']-1)]
            action, _states = model.predict(extend_obs(ob, config['num_threads']),
                                            state=_states, mask=done)
            action = action[0]
            ob, rew, done, info = env.step(action)
            if done:
                env.reset()
            if isinstance(info, (tuple, list)):
                info = info[0]
                action = action[0]
            state_mat.append(_states)
            if info['new_trial']:
                gt = env.gt_now
                correct = action == gt
                # Log trial info
                trial_info = env.trial
                trial_info.update({'correct': correct, 'choice': action})
                info_df = info_df.append(trial_info, ignore_index=True)
                # Log stimulus period activity
                activity.append(np.array(state_mat))
                state_mat = []
        env.close()


def order_by_sufix(file_list):
    file_list = [os.path.basename(x) for x in file_list]
    flag = 'model.zip' in file_list
    file_list = [x for x in file_list if x != 'model.zip']
    sfx = [int(x[x.find('_')+1:x.rfind('_')]) for x in file_list]
    sorted_list = [x for _, x in sorted(zip(sfx, file_list))]
    if flag:
        sorted_list.append('model.zip')
    return sorted_list, np.max(sfx)


def analysis_average_activity(activity, info, config):
    # Load and preprocess results
    plt.figure(figsize=(1.2, 0.8))
    t_plot = np.arange(activity.shape[1]) * config['dt']
    plt.plot(t_plot, activity.mean(axis=0).mean(axis=-1))

    # os.makedirs(FIGUREPATH / path, exist_ok=True)
    # plt.savefig(FIGUREPATH / path / 'meanactivity.png', transparent=True, dpi=600)


def get_conditions(info):
    """Get a list of task conditions to plot."""
    conditions = info.columns
    # This condition's unique value should be less than 5
    new_conditions = list()
    for c in conditions:
        try:
            n_cond = len(pd.unique(info[c]))
            if 1 < n_cond < 5:
                new_conditions.append(c)
        except TypeError:
            pass
        
    return new_conditions


def analysis_activity_by_condition(activity, info, config):
    conditions = get_conditions(info)
    for condition in conditions:
        values = pd.unique(info[condition])
        plt.figure(figsize=(1.2, 0.8))
        t_plot = np.arange(activity.shape[1]) * config['dt']
        for value in values:
            a = activity[info[condition] == value]
            plt.plot(t_plot, a.mean(axis=0).mean(axis=-1), label=str(value))
        plt.legend(title=condition, loc='center left', bbox_to_anchor=(1.0, 0.5))
        
        # os.makedirs(FIGUREPATH / path, exist_ok=True)
        

def analysis_example_units_by_condition(activity, info, config):
    conditions = get_conditions(info)
    if len(conditions) < 1:
        return

    example_ids = np.array([0, 1])    
    for example_id in example_ids:        
        example_activity = activity[:, :, example_id]
        fig, axes = plt.subplots(
                len(conditions), 1,  figsize=(1.2, 0.8 * len(conditions)),
                sharex=True)
        for i, condition in enumerate(conditions):
            ax = axes[i]
            values = pd.unique(info[condition])
            t_plot = np.arange(activity.shape[1]) * config['dt']
            for value in values:
                a = example_activity[info[condition] == value]
                ax.plot(t_plot, a.mean(axis=0), label=str(value))
            ax.legend(title=condition, loc='center left', bbox_to_anchor=(1.0, 0.5))
            ax.set_ylabel('Activity')
            if i == len(conditions) - 1:
                ax.set_xlabel('Time (ms)')
            if i == 0:
                ax.set_title('Unit {:d}'.format(example_id + 1))


def analysis_pca_by_condition(activity, info, config):
    # Reshape activity to (N_trial x N_time, N_neuron)
    activity_reshape = np.reshape(activity, (-1, activity.shape[-1]))
    pca = PCA(n_components=2)
    pca.fit(activity_reshape)
    
    conditions = get_conditions(info)
    for condition in conditions:
        values = pd.unique(info[condition])
        fig = plt.figure(figsize=(2.5, 2.5))
        ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
        for value in values:
            # Get relevant trials, and average across them
            a = activity[info[condition] == value].mean(axis=0)
            a = pca.transform(a)  # (N_time, N_PC)
            plt.plot(a[:, 0], a[:, 1], label=str(value))
        plt.legend(title=condition, loc='center left', bbox_to_anchor=(1.0, 0.5))
    
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')


if __name__ == '__main__':
    envid = 'PerceptualDecisionMaking-v0'
    train_network(envid)
    activity, info, config = run_network(envid)
    analysis_average_activity(activity, info, config)
    analysis_activity_by_condition(activity, info, config)
    analysis_example_units_by_condition(activity, info, config)
    analysis_pca_by_condition(activity, info, config)
    
