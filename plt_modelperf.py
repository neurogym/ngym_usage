#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# this requires custom_timings dict- ()
# just copypasting what I did to reuse models trained in bsc
import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
from neurogym.utils.plotting import plot_env
import importlib
from custom_timings import ALL_ENVS_MINIMAL_TIMINGS
import gym
from stable_baselines.common.vec_env import DummyVecEnv

def custom_plot_env(modelpath, num_steps_env=200):
    root_str = os.path.split(modelpath)[0].split('/')[-1]
    algo = root_str.split('_')[0]
    task = root_str.split('_')[1]
    seed = root_str.split('_')[-1]
    ngym_kwargs = {'dt':100, 'timing': ALL_ENVS_MINIMAL_TIMINGS[task]}
    env = gym.make(task, **ngym_kwargs)
    env = DummyVecEnv([lambda: env])
    pkg = importlib.import_module('stable_baselines') #+algo) 
    module = getattr(pkg, algo)
    model = module.load(modelpath)
    plot_env(env, num_steps_env=num_steps_env, model=model, name=f'{algo} on {task}', fig_kwargs={'figsize':(10, 12)})
    plt.show()

if __name__ == '__main__':
    print('take as an example:')
    files = sorted([str(x) for x in Path('/home/jordi/Repos/pkgs/data/3rd/').glob('*0/model.zip')])
    custom_plot_env(files[0])
