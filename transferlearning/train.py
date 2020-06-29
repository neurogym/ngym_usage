"""Transfer Learning"""

import os
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn

import gym
import neurogym as ngym

from models import RNNNet, get_performance


def get_filename(task, pretrain_task=None):
    if pretrain_task is None:
        return task
    else:
        return pretrain_task + '_' + task


def get_modelname(task, pretrain_task=None):
    return 'model_' + get_filename(task, pretrain_task) + '.pt'


def get_logname(task, pretrain_task=None):
    return 'log_' + get_filename(task, pretrain_task) + '.npz'


def train_task(task, pretrain_task=None):
    # Environment
    kwargs = {'dt': 100}
    seq_len = 100

    env = gym.make(task, **kwargs)
    dataset = ngym.Dataset(env, batch_size=4, seq_len=seq_len)

    env = dataset.env
    ob_size = env.observation_space.shape[0]
    act_size = env.action_space.n

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = RNNNet(input_size=ob_size, hidden_size=256, output_size=act_size,
                   dt=env.dt).to(device)

    if pretrain_task is not None:
        fname = os.path.join('files', get_modelname(pretrain_task))
        model.load_state_dict(torch.load(fname, map_location=torch.device(device)))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print_step = 200
    running_loss = 0.0
    running_task_time = 0
    running_train_time = 0
    log = defaultdict(list)
    for i in range(2000):
        task_time_start = time.time()
        inputs, labels = dataset()
        running_task_time += time.time() - task_time_start
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
        labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)

        train_time_start = time.time()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, _ = model(inputs)

        loss = criterion(outputs.view(-1, act_size), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_train_time += time.time() - train_time_start
        # print statistics
        running_loss += loss.item()
        if i % print_step == (print_step - 1):
            running_loss /= print_step
            log['step'].append(i)
            log['loss'].append(running_loss)
            print('{:d} loss: {:0.5f}'.format(i + 1, running_loss))
            running_loss = 0.0
            if True:
                print('Task/Train time {:0.1f}/{:0.1f} ms/step'.format(
                        running_task_time / print_step * 1e3,
                        running_train_time / print_step * 1e3))
                running_task_time, running_train_time = 0, 0

            perf = get_performance(model, env, num_trial=200, device=device)
            log['perf'].append(perf)
            print('{:d} perf: {:0.2f}'.format(i + 1, perf))

            fname = os.path.join('files', get_logname(task, pretrain_task))
            np.savez_compressed(fname, **log)

            fname = os.path.join('files', get_modelname(task, pretrain_task))
            torch.save(model.state_dict(), fname)

    print('Finished Training')


if __name__ == '__main__':
    # Make supervised dataset
    tasks = ngym.get_collection('yang19')
    # tasks = tasks[:3]
    for task in tasks:
        train_task(task)
    for task in tasks:
        for pretrain_task in tasks:
            train_task(task, pretrain_task=pretrain_task)
