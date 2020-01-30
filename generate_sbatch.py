#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.expanduser('~/neurogym'))
import neurogym as ngym
dirtosave = '/home/molano/Jan2020/generated_scripts/'
if not os.path.exists(dirtosave):
    os.makedirs(dirtosave)

workdir = '/home/hcli64/hcli64348/'
logdir = '/home/hcli64/hcli64348/Jan2020/logs/'

commontxt = (
    '#SBATCH --cpus-per-task=4\n'
    '#SBATCH --time=8:00:00\n'
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


def gen_file(modelname, task, instance, num_tr, rollout):
    fullname = f'{modelname}_{task}_{instance}'
    with open(f'{dirtosave}{modelname}_{task}_{instance}.sh', 'a') as f:
        f.write('#!/bin/sh\n')
        f.write(f'#SBATCH --job-name={fullname}\n')
        f.write(f'#SBATCH --output={logdir}{fullname}.out\n')
        f.write(f'#SBATCH -D {workdir}\n')
        f.write(commontxt)
        f.write(f'/home/hcli64/hcli64348/bsls_run.py {modelname} {task} {instance} {num_tr} {rollout}')


if __name__ == '__main__':
    for model in ['A2C', 'ACER', 'ACKTR', 'PPO2', 'SL']:
        for tsk in ngym.all_tasks.keys():
            for i in range(2):
                for ntr in [100000]:
                    for rll in [20]:
                        gen_file(model, tsk, i, ntr, rll)

    print('done')
