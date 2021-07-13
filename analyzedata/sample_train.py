import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn

import gym
import neurogym as ngym
from neurogym import spaces
import pynwb
from pynwb import NWBFile
from pynwb import NWBHDF5IO
import datetime

import sys
sys.path.append('..')  # TODO: Temporary hack
from training.supervised_train import train_network, Net


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Steinmetz19(ngym.TrialEnv):
    """Two-alternative forced choice task in which the subject has to
    integrate two stimuli to decide which one is higher on average.

    A noisy stimulus is shown during the stimulus period. The strength (
    coherence) of the stimulus is randomly sampled every trial. Because the
    stimulus is noisy, the agent is encouraged to integrate the stimulus
    over time.

    Args:
        cohs: list of float, coherence levels controlling the difficulty of
            the task
        sigma: float, input noise level
        dim_ring: int, dimension of ring input and output
    """
    metadata = {
        'paper_link': 'https://www.jneurosci.org/content/12/12/4745',
        'paper_name': '''The analysis of visual motion: a comparison of
        neuronal and psychophysical performance''',
        'tags': ['perceptual', 'two-alternative', 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, cohs=None,
                 sigma=1.0, dim_ring=2):
        super().__init__(dt=dt)
        if cohs is None:
            self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2])
        else:
            self.cohs = cohs
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': [100, 200, 300],
            'stimulus': [500, 1000, 2000],
            'delay': 0,
            'decision': 100}
        if timing:
            self.timing.update(timing)

        self.abort = False

        self.theta = np.linspace(0, 2*np.pi, dim_ring+1)[:-1]
        self.choices = np.arange(dim_ring)

        name = {'fixation': 0, 'stimulus': range(1, dim_ring+1)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1+dim_ring,), dtype=np.float32, name=name)
        name = {'fixation': 0, 'choice': range(1, dim_ring+1)}
        self.action_space = spaces.Discrete(1+dim_ring, name=name)

    def _new_trial(self, **kwargs):
        """
        new_trial() is called when a trial ends to generate the next trial.
        The following variables are created:
            durations, which stores the duration of the different periods (in
            the case of perceptualDecisionMaking: fixation, stimulus and
            decision periods)
            ground truth: correct response for the trial
            coh: stimulus coherence (evidence) for the trial
            obs: observation
        """
        # Trial info
        trial = {
            'ground_truth': self.rng.choice(self.choices),
            'coh': self.rng.choice(self.cohs),
        }
        trial.update(kwargs)

        coh = trial['coh']
        ground_truth = trial['ground_truth']
        stim_theta = self.theta[ground_truth]

        # Periods
        self.add_period(['fixation', 'stimulus', 'delay', 'decision'])

        # Observations
        self.add_ob(1, period=['fixation', 'stimulus', 'delay'], where='fixation')
        stim = np.cos(self.theta - stim_theta) * (coh/200) + 0.5
        self.add_ob(stim, 'stimulus', where='stimulus')
        self.add_randn(0, self.sigma, 'stimulus', where='stimulus')

        # Ground truth
        self.set_groundtruth(ground_truth, period='decision', where='choice')

        return trial

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # observations
        if self.in_period('fixation'):
            if action != 0:  # action = 0 means fixating
                new_trial = self.abort
                reward += self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward += self.rewards['correct']
                    self.performance = 1
                else:
                    reward += self.rewards['fail']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


ngym.register('Steinmetz19-v0', entry_point='sample_train:Steinmetz19')


def get_modelpath(envid):
    # Make a local file directories
    path = Path('.') / 'files'
    os.makedirs(path, exist_ok=True)
    path = path / envid
    os.makedirs(path, exist_ok=True)
    return path


def infer_test_timing(env):
    """Infer timing of environment for testing."""
    timing = {} 
    for period in env.timing.keys():
        period_times = [env.sample_time(period) for _ in range(100)]
        timing[period] = np.median(period_times)
    return timing


def sample_spikes(rate, dt):
    """Return a list of spike times given a list of rate.
    
    Args:
        rate: a list of rate values to sample spikes from
        dt: the temporal interval of each rate value, in unit second
        
    Returns:
        spike_times: a list of spike times
    """
    spike_dt = 0.001
    probs = np.repeat(rate, int(dt/spike_dt)) * spike_dt
    spikes = np.random.rand(len(probs)) < probs
    spike_times = np.where(spikes)[0] * spike_dt
    
    return spike_times
    

def run_trial_torch_sl(model, env):
    """Run model on env for one trial.
    
    Args:
        model: a model file, here a Pytorch nn.Module
        env: ngym env
        
    Returns:
        model: same as above
        env: same as above
        hidden: activity matrix (N_time, N_unit)  # TODO: Need to support multi-area
        trial_info: dictionary of trial information
    """
    env.new_trial()
    ob, gt = env.ob, env.gt
    inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float)
    action_pred, hidden = model(inputs)

    # Compute performance
    action_pred = action_pred.detach().numpy()
    choice = np.argmax(action_pred[-1, 0, :])
    correct = choice == gt[-1]
    
    # Log stimulus period activity
    hidden = np.array(hidden)[:, 0, :]
    
    trial_info = env.trial
    trial_info.update({'correct': correct, 'choice': choice})
    
    return model, env, hidden, trial_info


def run_network(model, env, run_trial, num_trial=1000, file=None):
    """Run trained networks for analysis on trial-based tasks.

    Args:
        model: model of arbitrary format, must provide a run_one_trial function
            that works with it
        env: neurogym environment
        run_trial: function handle for running model for one trial,
            takes (model, env) as inputs and 
            returns (model, env, activity, trial_info), where activity has 
            shape (N_time, N_unit)
        num_trial: int, number of trials to run
        file: str or None, file name to save

    Returns:
        activity: a list of activity matrices, each matrix has shape (
        N_time, N_neuron)
        info: pandas dataframe, each row is information of a trial
        config: dict of network, training configurations
    """
    env.reset(no_step=True)
    
    # Make NWB file
    nwbfile = NWBFile(session_description=str(env),  # required
                  identifier='NWB_default',  # required
                  session_start_time=datetime.datetime.now(),  # required
                  file_create_date=datetime.datetime.now())

    info = pd.DataFrame()

    spike_times = defaultdict(list)
    start_time = 0.
    for i in range(num_trial):
        model, env, hidden, trial_info = run_trial(model, env)

        # Log trial info
        for key, val in env.start_t.items():
            # NWB time default unit is second, ngym default is ms
            trial_info['start_' + key] = val / 1000. + start_time
        for key, val in env.end_t.items():
            trial_info['end_' + key] = val / 1000. + start_time
        
        info = info.append(trial_info, ignore_index=True)
        
        # Store results to NWB file
        if i == 0:
            for key in trial_info.keys():
                nwbfile.add_trial_column(name=key, description=key)

        stop_time = start_time + hidden.shape[0] * env.dt / 1000.
        
        # Generate simulated spikes from rates
        scale_rate = 10.
        for j in range(hidden.shape[-1]):
            spikes = sample_spikes(hidden[:, j] * scale_rate, 
                                   dt=env.dt / 1000.) + start_time
            spike_times[j].append(spikes)
        
        nwbfile.add_trial(start_time=start_time, 
                          stop_time=stop_time, **trial_info)
        start_time = stop_time  # Assuming continous trials

    try:
        print('Average performance', np.mean(info['correct']))
    except:
        pass

    for j in range(hidden.shape[-1]):  # For each neuron
        nwbfile.add_unit(id=j, spike_times=np.concatenate(spike_times[j]))
    # TODO: Check why the file.units['spike_times'] is weird

    if file is None:
        file = str(get_modelpath(envid) / (envid + '.nwb'))
    with pynwb.NWBHDF5IO(file, 'w') as io:
        io.write(nwbfile)


if __name__ == '__main__':
    # envid = 'Steinmetz19-v0'
    envid = 'ReadySetGo-v0'
    torch.set_grad_enabled(True)
    train_network(envid)
    # activity, info, config = run_network('Steinmetz19-v0')
    
    modelpath = get_modelpath(envid)
    with open(modelpath / 'config.json') as f:
        config = json.load(f)

    env_kwargs = config['env_kwargs']

    # Run network to get activity and info
    # Environment
    env = gym.make(envid, **env_kwargs)
    # env.timing = infer_test_timing(env)

    torch.set_grad_enabled(False)
    net = Net(input_size=env.observation_space.shape[0],
              hidden_size=config['hidden_size'],
              output_size=env.action_space.n)
    net = net.to(device)
    net.load_state_dict(torch.load(modelpath / 'net.pth'))

    run_network(net, env, run_trial_torch_sl)
