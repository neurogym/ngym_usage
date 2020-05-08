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
# SCRIPTSDIR = '/home/hcli64/hcli64745/shaping/scripts/'
# WORKDIR = '/home/hcli64/hcli64745/'
# SCRIPT = 'shaping_run.py'
SCRIPTSDIR = '/home/hcli64/hcli64348/anna_k/scripts/'
WORKDIR = '/home/hcli64/hcli64348/'
SCRIPT = 'bsc_run.py'

commontxt_1 = (
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
    '# COMPUTE NUBMER OF TASKS PER GPU\n'
    'REQUESTED_GPUS=4 # ask for 4 gpus (each node has 4 gpus)\n'
    'if [[ $SLURM_TASKS_PER_NODE -lt $REQUESTED_GPUS ]]; then\n'
    'NTASKS_PER_GPU=1\n'
    'else\n'
    'NTASKS_PER_GPU=$((SLURM_TASKS_PER_NODE/$REQUESTED_GPUS))\n'
    'fi\n'
    '# LAUNCH RUNS'
    "DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\\n')\n"
    )
commontxt_2 = (
    'for DEVICE in $DEVICES; do\n'
    'export CUDA_VISIBLE_DEVICES=$DEVICE\n'
    'I=0\n'
    'while [[ $I -lt $NTASKS_PER_GPU ]]; do\n'
    )

commontxt_3 = (
    "declare \"PID${DEVICE}${I}=$!\"\n"
    'I=$(($I+1))\n'
    'SEED=$(($SEED+1))\n'
    'done\n'
    'done\n'
    '# CHECK THAT EVERYTHING IS FINE (?)\n'
    'RET=0\n'
    'for DEVICE in $DEVICES; do\n'
    'I=0\n'
    'while [[ $I -lt $NTASKS_PER_GPU ]]; do\n'
    "PID=\"PID${DEVICE}${I}\"\n"
    'wait ${!PID}\n'
    'RET=$(($RET+$?))\n'
    'I=$(($I+1))\n'
    'done\n'
    'done\n'
    'exit $RET #if 0 all ok, else something failed\n'
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


def gen_file(exp, n_cpu, run_time, dirtosave, num_tasks_per_node,
             **kwargs):
    # "To ensure fair and reliable CPU usage accounting information, we’ve
    # enforced the need to use at least 40 threads for each GPU requested.
    # In your job scripts, make sure that the amount of threads used meet the
    # requirements for your GPU needs. Note that Slurm does refer to each thread
    # as if it was a physical CPU.
    # https://www.bsc.es/user-support/power.php#submittingjobs
    name, cmd = get_name_and_command_from_dict(kwargs)
    seed = num_tasks_per_node*kwargs['seed']  # each seed will actually correspond
    with open(f'{SCRIPTSDIR}/{exp}/{name}.sh', 'w') as f:
        f.write('#!/bin/sh\n')
        f.write('#SBATCH -N 1 # 1 node\n')
        f.write('#SBATCH --gres=gpu:4 # 4 GPUs\n')
        f.write(f'#SBATCH --job-name={name}\n')
        f.write(f'#SBATCH --output={dirtosave}/logs/{name}.out\n')
        f.write(f'#SBATCH -D {WORKDIR}\n')
        f.write(f'#SBATCH --cpus-per-task={n_cpu}\n')
        f.write(f'#SBATCH -n {num_tasks_per_node} # number of task per node\n')
        f.write(f'#SBATCH --time={run_time}:00:00\n')
        f.write(commontxt_1)
        f.write(f'SEED={seed}\n')
        f.write(commontxt_2)
        f.write(f'{WORKDIR}{SCRIPT} --folder {dirtosave} {cmd} &\n')
        f.write(commontxt_3)


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
    num_tasks_per_node = params.general_params['num_tasks_per_node']
    num_cpu = int(160/num_tasks_per_node)  # 160= 4 gpus-per-node x 40 cpus-per-gpu
    if not os.path.exists(SCRIPTSDIR + '/' + experiment):
        os.makedirs(SCRIPTSDIR + '/' + experiment)
    if not os.path.exists(dirtosave + '/logs'):
        os.makedirs(dirtosave + '/logs')

    combinations = itertools.product(*explore.values())
    for ind_c, comb in enumerate(combinations):
        pars = {key: comb[ind] for ind, key in enumerate(explore.keys())}
        print(pars)
        gen_file(exp=experiment, n_cpu=num_cpu, run_time=run_time,
                 dirtosave=dirtosave, num_tasks_per_node=num_tasks_per_node,
                 **pars)

    print('done')
