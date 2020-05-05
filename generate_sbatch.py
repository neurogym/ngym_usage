#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:28:51 2020

@author: molano
"""

import os
import sys
import importlib
import itertools
# scriptsdir = '/home/hcli64/hcli64745/shaping/scripts/'
# workdir = '/home/hcli64/hcli64745/'
# script = 'shaping_run.py'
scriptsdir = '/home/hcli64/hcli64348/anna_k/scripts/'
workdir = '/home/hcli64/hcli64348/'
script = 'bsc_run.py'

commontxt = (
    'module purge\n'
    'module load gcc/6.4.0\n'
    'module load cuda/9.1\n'
    'module load cudnn/7.1.3\n'
    'module load openmpi/3.0.0\n'
    'module load atlas/3.10.3\n'
    'module load scalapack/2.0.2\n'
    'module load fftw/3.3.7\n'
    'module load szip/2.1.1\n'
    'module load ffmpeg\n'
    'module load opencv/3.4.1\n'
    'module load python/3.6.5_ML\n'
)


def get_name_and_command_from_dict(d):
    name = ''
    cmd = ''
    for k in d.keys():
        if k != 'folder':
            if isinstance(d[k], list):
                name += k + '_'
                cmd += ' --' + k
                for el in d[k]:
                    name += str(el)
                    cmd += ' ' + str(el)
                name += '_'
            else:
                name += k + '_' + str(d[k]) + '_'
                cmd += ' --' + k + ' ' + str(d[k])

    return name[:-1], cmd


def gen_file(exp, n_cpu, run_time, **kwargs):
    # "To ensure fair and reliable CPU usage accounting information, we’ve
    # enforced the need to use at least 40 threads for each GPU requested.
    # In your job scripts, make sure that the amount of threads used meet the
    # requirements for your GPU needs. Note that Slurm does refer to each thread
    # as if it was a physical CPU.
    # https://www.bsc.es/user-support/power.php#submittingjobs
    n_cpu = int(40/n_cpu)
    name, cmd = get_name_and_command_from_dict(kwargs)
    with open(f'{scriptsdir}/{exp}/{name}.sh', 'w') as f:
        f.write('#!/bin/sh\n')
        f.write(f'#SBATCH --job-name={name}\n')
        f.write(f'#SBATCH --output={dirtosave}/logs/{name}.out\n')
        f.write(f'#SBATCH -D {workdir}\n')
        f.write(f'#SBATCH --gres=gpu:1\n')
        f.write(f'#SBATCH --cpus-per-task={n_cpu}\n')
        f.write(f'#SBATCH --time={run_time}:00:00\n')
        f.write(commontxt)
        f.write(f'{workdir}{script} --folder {dirtosave} {cmd}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("usage: gen_scripts.py [path]")

    dirtosave = sys.argv[1]
    if not os.path.exists(dirtosave):
        raise ValueError("Provided path does not exist")

    # load parameters
    sys.path.append(os.path.expanduser(dirtosave))
    spec = importlib.util.spec_from_file_location("params",
                                                  dirtosave+"/params.py")
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)
    explore = params.explore
    experiment = params.experiment
    run_time = params.general_params['run_time']
    num_cpu = params.general_params['num_cpu']
    if not os.path.exists(scriptsdir + '/' + experiment):
        os.makedirs(scriptsdir + '/' + experiment)
    if not os.path.exists(dirtosave + '/logs'):
        os.makedirs(dirtosave + '/logs')
        

    combinations = itertools.product(*explore.values())
    for ind_c, comb in enumerate(combinations):
        pars = {key: comb[ind] for ind, key in enumerate(explore.keys())}
        print(pars)
        gen_file(exp=experiment, n_cpu=num_cpu, run_time=run_time, **pars)

    print('done')
