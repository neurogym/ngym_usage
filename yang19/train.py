"""Train networks for reproducing multi-cognitive-tasks from

Task representations in neural networks trained to perform many cognitive tasks
https://www.nature.com/articles/s41593-018-0310-2
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn

import gym
import neurogym as ngym
from neurogym.wrappers import ScheduleEnvs
from neurogym.utils.scheduler import RandomSchedule

from models import RNNNet, get_performance

# Environment
kwargs = {'dt': 100}
# kwargs = {'dt': 100, 'sigma': 0, 'dim_ring': 2, 'cohs': [0.1, 0.3, 0.6, 1.0]}
seq_len = 100

# Make supervised dataset
tasks = ngym.get_collection('yang19')
envs = [gym.make(task, **kwargs) for task in tasks]
schedule = RandomSchedule(len(envs))
env = ScheduleEnvs(envs, schedule=schedule, env_input=True)
dataset = ngym.Dataset(env, batch_size=4, seq_len=seq_len)

env = dataset.env
ob_size = env.observation_space.shape[0]
act_size = env.action_space.n

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = RNNNet(input_size=ob_size, hidden_size=256, output_size=act_size,
               dt=env.dt).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print_step = 200
running_loss = 0.0
running_task_time = 0
running_train_time = 0
for i in range(40000):
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
        print('{:d} loss: {:0.5f}'.format(i + 1, running_loss / print_step))
        running_loss = 0.0
        if True:
            print('Task/Train time {:0.1f}/{:0.1f} ms/step'.format(
                    running_task_time / print_step * 1e3,
                    running_train_time / print_step * 1e3))
            running_task_time, running_train_time = 0, 0

        perf = get_performance(model, env, num_trial=200, device=device)
        print('{:d} perf: {:0.2f}'.format(i + 1, perf))

        fname = os.path.join('files', 'model.pt')
        torch.save(model.state_dict(), fname)

print('Finished Training')
