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
import numpy as np
# SCRIPTSDIR = '/home/hcli64/hcli64745/shaping/scripts/'
# WORKDIR = '/home/hcli64/hcli64745/'
# SCRIPT = 'shaping_run.py'
# SCRIPTSDIR = '/home/molano/ngym_usage/tests/'
SCRIPTSDIR = '/home/hcli64/hcli64348/anna_k/scripts/'
WORKDIR = '/home/hcli64/hcli64348/'
SCRIPT = 'bsc_run.py'
NUM_CPUS_PER_GPU = 40
NUM_GPUS_PER_NODE = 4
NUM_CPUS_PER_NODE = NUM_GPUS_PER_NODE*NUM_CPUS_PER_GPU
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
    '# COMPUTE NUMBER OF TASKS PER GPU\n'
    'REQUESTED_GPUS=4 # ask for 4 gpus (each node has 4 gpus)\n'
    'if [[ $SLURM_TASKS_PER_NODE -lt $REQUESTED_GPUS ]]; then\n'
    'NTASKS_PER_GPU=1\n'
    'else\n'
    'NTASKS_PER_GPU=$((SLURM_TASKS_PER_NODE/$REQUESTED_GPUS))\n'
    'fi\n'
    '# LAUNCH RUNS\n'
    )

commontxt_2 = (
    '# CHECK THAT EVERYTHING IS FINE (?)\n'
    'RET=0\n'
    )

commontxt_3 = (
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


def gen_file(combs, exp, n_cpu, num_gpus, run_time, dirtosave, num_tasks_per_node,
             num_tasks_per_gpu, keys):
    # "To ensure fair and reliable CPU usage accounting information, we’ve
    # enforced the need to use at least 40 threads for each GPU requested.
    # In your job scripts, make sure that the amount of threads used meet the
    # requirements for your GPU needs. Note that Slurm does refer to each thread
    # as if it was a physical CPU.
    # https://www.bsc.es/user-support/power.php#submittingjobs
    comb_tmp = {key: combs[0][ind] for ind, key in enumerate(keys)}
    name, _ = get_name_and_command_from_dict(comb_tmp)
    with open(f'{SCRIPTSDIR}/{exp}/{name}.sh', 'w') as f:
        f.write('#!/bin/sh\n')
        # f.write('#SBATCH -N 1 # 1 node\n')
        f.write(f'#SBATCH --gres=gpu:{num_gpus} # num GPUs\n')
        f.write(f'#SBATCH --job-name={name}\n')
        f.write(f'#SBATCH --output={dirtosave}/logs/{name}.out\n')
        f.write(f'#SBATCH -D {WORKDIR}\n')
        f.write(f'#SBATCH --cpus-per-task={n_cpu}\n')
        f.write(f'#SBATCH -n {num_tasks_per_node} # number of task per node\n')
        f.write(f'#SBATCH --time={run_time}:00:00\n')
        f.write(commontxt_1)
        run_cnt = 0
        for ind_gpu in range(num_gpus):
            f.write('################\n')
            f.write(f'export CUDA_VISIBLE_DEVICES={ind_gpu}\n')
            for ind_run in range(num_tasks_per_gpu):
                c = {key: combs[run_cnt][ind] for ind, key in enumerate(keys)}
                _, cmd = get_name_and_command_from_dict(c)
                f.write(f'{WORKDIR}{SCRIPT} --folder {dirtosave} {cmd} &\n')
                f.write(f'PID{ind_gpu}{ind_run}=$!\n')
                run_cnt += 1
        f.write(commontxt_2)
        for ind_gpu in range(num_gpus):
            f.write(f'export CUDA_VISIBLE_DEVICES={ind_gpu}\n')
            for ind_run in range(num_tasks_per_gpu):
                f.write(f'PID="PID{ind_gpu}{ind_run}"\n')
                f.write('wait ${!PID}\n')
                f.write('RET=$(($RET+$?))\n')
        f.write(commontxt_3)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        dirtosave = sys.argv[1]
    elif len(sys.argv) == 1:
        dirtosave = '/home/molano/ngym_usage/tests/'
    else:
        raise ValueError("usage: gen_scripts.py [path]")

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
    num_cpus_per_task = params.general_params['num_cpu']
    num_tasks_per_node = int(NUM_CPUS_PER_NODE/num_cpus_per_task)
    # num_cpus_per_node = num_cpus_per_task*num_tasks_per_node
    # 160 = 4 gpus-per-node x 40 cpus-per-gpu
    # assert num_cpus_per_node <= NUM_CPUS_PER_NODE, 'Asking for more than 1 node'
    if not os.path.exists(SCRIPTSDIR + '/' + experiment):
        os.makedirs(SCRIPTSDIR + '/' + experiment)
    if not os.path.exists(dirtosave + '/logs'):
        os.makedirs(dirtosave + '/logs')
    combinations = list(itertools.product(*explore.values()))
    try:
        explore_bis = params.explore_bis
        combinations_bis = list(itertools.product(*explore_bis.values()))
        combinations = combinations + combinations_bis
    except AttributeError:
        print('no explore bis attribute')
    assert ((len(combinations)*num_cpus_per_task) % NUM_CPUS_PER_GPU) == 0
    num_nodes = int(np.ceil(len(combinations)*num_cpus_per_task/NUM_CPUS_PER_NODE))
    for ind_node in range(num_nodes):
        combs = combinations[ind_node*num_tasks_per_node:
                             (ind_node+1)*num_tasks_per_node]
        print(combs)
        print('----------')
        num_task_in_node = len(combs)
        num_gpus = int(num_task_in_node*num_cpus_per_task/NUM_CPUS_PER_GPU)
        assert ((num_task_in_node*num_cpus_per_task) % NUM_CPUS_PER_GPU) == 0
        num_tasks_per_gpu = int(num_task_in_node/num_gpus)
        assert (num_task_in_node % num_gpus) == 0
        gen_file(combs=combs, exp=experiment, n_cpu=num_cpus_per_task,
                 num_gpus=num_gpus, run_time=run_time, dirtosave=dirtosave,
                 num_tasks_per_node=num_task_in_node,
                 num_tasks_per_gpu=num_tasks_per_gpu, keys=explore.keys())

    print('done')
