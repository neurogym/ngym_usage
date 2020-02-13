#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:09:30 2020

@author: molano
"""

import gym
import numpy as np
import neurogym as ngym
from neurogym.utils import plot_env
import pandas as pd
# import seaborn as sns


class randomAgent():
    """
    The random agent just outputs a random action following a probability
    distribution policy

    :param policy: list, probability distribution to sample the action space
                  alternatively, policy can be a list with only one value
                  indicating the probability of the action 0 (usually fixate)
    :param env: task
    """

    def __init__(self, env, policy, **kwargs):
        continuous_actions = isinstance(env.action_space, gym.spaces.Box)
        discrete_actions = isinstance(env.action_space, gym.spaces.Discrete)
        if continuous_actions:
            raise NotImplementedError('Not implemented for continuous actions')
        elif not discrete_actions:
            raise TypeError('Unknown action space')
        self.n = int(env.action_space.n)
        if len(policy) == self.n:  # all probabilities are specified
            self.policy = np.array(policy)
        elif len(policy) == 1:  # only probability of fixating is specified
            probs_alt_acts = (1-policy[0])/(self.n-1)
            self.policy = np.concatenate((np.array(policy),
                                          np.ones((self.n-1,))*probs_alt_acts))
        else:
            raise TypeError('Wrong policy format')

        self.env = env

    def predict(self, obs):
        ch = int(np.random.choice(np.arange(self.n), p=self.policy))
        return ch, []


if __name__ == '__main__':
    main_folder = '/home/molano/ngym_usage/results/random_agent/'
    n_stps = 2000000
    envs = sorted(ngym.all_envs())
    fixate_probs = [.1, .33, .5, .666, .9]
    parameters = ['mean_rew', 'max_rew', 'min_rew',
                  'mean_perf', 'max_perf', 'min_perf']
    funcs = [np.mean, np.max, np.min]*2
    items = ['rewards']*3 + ['perf']*3
    df = pd.DataFrame({'tasks': envs})
    for pr in fixate_probs:
        for par in parameters:
            df['prob'+str(pr)+' '+par] = np.nan
    for ind_pr, pr in enumerate(fixate_probs):
        for ind_env, env_name in enumerate(envs):
            env = gym.make(env_name)
            try:
                model = randomAgent(env=env, policy=[.9])
                data = plot_env(env, num_steps_env=n_stps, model=model,
                                show_fig=False)
                for par, f, it in zip(parameters, funcs, items):
                    df.loc[df.tasks == env_name,
                           'prob'+str(pr)+' '+par] = f(data[it])
            except Exception as e:
                print('Failure at running env: {:s}'.format(env_name))
                print(e)
    df.to_pickle(main_folder + 'random_agent')


#    # plot the heatmap
#    sns.heatmap(df,  xticklabels=df.columns, yticklabels=df.columns)
