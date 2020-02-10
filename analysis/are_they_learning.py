#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
import seaborn as sns
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))


# In[2]:


# path 
data_pth = '/home/jordi/Repos/pkgs/data/'
files = [str(x) for x in Path(data_pth).glob('**/*.npz')]
files


# ### 1st

# In[3]:


sublist = sorted([x for x in files if 'Romo-v0' in x]) # 'ReadySetGo'
sublist


# In[5]:


for algo in ['A2C', 'ACER', 'ACKTR']:
    targ = sorted([x for x in sublist if algo in x])[0]
    test = np.load(targ, allow_pickle=True)
    print(algo)
    for item in test.files:
        print(f'{item} shape {test[item].shape}')


# In[6]:


test=np.load('/home/jordi/Repos/pkgs/data/A2C_Romo-v0_0.npz', allow_pickle=True)
test.files


# In[7]:


test['rewards'].shape


# In[8]:


plt.figure(figsize=(16,4))
for i in range(test['rewards'].shape[1]):
    plt.plot(pd.Series(test['rewards'][:,i]).rolling(25).mean())

plt.axhline(y=0, c='k', ls=':')
plt.xlabel('dt')
plt.ylabel('reward rolling average (win=25)')
plt.show()


# In[9]:


plt.figure(figsize=(16,4))
for i in range(test['rewards'].shape[1]):
    plt.plot(test['rewards'][:,i])

plt.axhline(y=0, c='k', ls=':')
plt.xlabel('dt')
plt.ylabel('raw reward')
plt.show()


# In[10]:


pd.DataFrame(test['rewards']).boxplot()


# In[12]:


import neurogym as ngym
len(ngym.all_tasks.keys())


# In[43]:


toplot = []
for task in ngym.all_tasks.keys():
    try:
        targfile = [x for x in files if (task in x) and ('ACER' in x) and (x.endswith('0.npz'))][0] # 'A2C'
        toplot += [targfile]
    except:
        print(f'{task} not found')
        continue
    print(targfile)


# In[44]:


test['rewards'].shape


# In[45]:


len(toplot)


# In[39]:


f, ax = plt.subplots(nrows=len(toplot), sharex=True, figsize=(16, 4*len(toplot)))
ax=ax.flatten()

for j,item in enumerate(toplot):
    test=np.load(item, allow_pickle=True)
    rew = test['rewards'].squeeze()
    ax[j].axhline(y=0, c='k', ls=':')
    for i in range(rew.shape[1]):
        ax[j].plot(pd.Series(rew[:,i]).rolling(25).mean())

    ax[j].set_title(item.split('/')[-1])
    ax[j].set_ylabel('rolling(w=25)reward')

ax[j].set_xlabel('dt')
plt.show()


# In[40]:


f, ax = plt.subplots(nrows=len(toplot),ncols=2,sharex=True, figsize=(16*2, 4*len(toplot)))
ax=ax.flatten()

for j,item in enumerate(toplot):
    j = j*2
    test=np.load(item, allow_pickle=True)
    
    rew = test['rewards'].squeeze()
    ax[j].axhline(y=0, c='k', ls=':')
    ax[j+1].axhline(y=0, c='k', ls=':')
    for i in range(rew.shape[1]):
        ax[j].plot(rew[:,i])
        ax[j+1].plot(pd.Series(rew[:,i]).rolling(25).mean())
        
    ax[j].set_title(item.split('/')[-1]+ 'reward per timestep')
    ax[j].set_ylabel('rewards')
    ax[j+1].set_title(item.split('/')[-1]+ ' rolling mean')
    ax[j+1].set_ylabel('rolling(w=25)reward')

ax[j].set_xlabel('dt')
ax[j+1].set_xlabel('dt')
plt.show()


# In[46]:


for j,item in enumerate(toplot):
    test=np.load(item, allow_pickle=True)
    print(test['rewards'].shape)


# ### 2nd batch

# In[13]:


sl_list = sorted([x for x in files if os.path.split(x)[1].startswith('training')])
sl_list


# In[9]:


d = np.load(sl_list[0], allow_pickle=True)
d.files


# In[17]:


f, ax = plt.subplots(nrows=int(len(sl_list)/2), ncols=3, sharex=True, figsize=(16*2, 4*len(sl_list)/2))
#ax=ax.flatten()

for j,item in enumerate(sl_list):
    j = j//2
    test=np.load(item, allow_pickle=True)
    fname = item.split('/')[-2]
    #ax[j].axhline(y=0, c='k', ls=':')
    #ax[j+1].axhline(y=0, c='k', ls=':')
    for i, metric in enumerate(['acc','loss', 'perf']):
        #ax[j][].axhline(y=0, c='k', ls=':')
        ax[j][i].plot(test[metric], label=f'{fname} {metric}')
        ax[j][i].legend()
        #ax[j+1].plot(pd.Series(rew[:,i]).rolling(25).mean())
        
    # ax[j].set_title(item.split('/')[-1]+ 'reward per timestep')
    # ax[j].set_ylabel('rewards')
    #ax[j+1].set_title(item.split('/')[-1]+ ' rolling mean')
    ax[j][0].set_ylabel(fname[:-5])

# ax[j].set_xlabel('dt')
# ax[j+1].set_xlabel('dt')
plt.show()


# In[18]:


# there are several tasks missing
sl_list_working = [sl_list[i].split('/')[-2][:-5] for i in range(0, len(sl_list), 2)]
sl_list_working


# In[21]:


import neurogym as ngym
sl_not_working = [x[:-3] for x in ngym.all_tasks.keys() if 'SL_'+x[:-3] not in sl_list_working]
sl_not_working


# In[34]:


a2cworking = list(set([os.path.split(x)[1][4:-6] for x in files if os.path.split(x)[1].startswith('A2C')]))


# In[40]:


a2notworking = [x for x in ngym.all_tasks.keys() if x not in a2cworking]


# In[41]:


len(a2cworking)


# In[42]:


a2cworking


# In[43]:


len(ngym.all_tasks.keys())


# In[46]:


a2notworking


# ### 3rd, RL (a2c)

# In[49]:


a2cworking = sorted(a2cworking)
a2cworking


# In[85]:


print(files[0])
test = np.load(files[0], allow_pickle=True)
test.files


# In[86]:


test['rewards'].shape


# In[92]:


f, ax = plt.subplots(ncols=2, nrows=10, figsize=(9,28))
ax=ax.flatten()
for i in range(20):
    ax[i].plot(test['rewards'][70000:,i])
    ax[i].set_title(f'dim (:,{i})')
    
plt.show()


# In[80]:


print(files[2])
test = np.load(files[2], allow_pickle=True)
print(test['rewards'].shape)
test.files


# In[93]:


kek = np.array([[0,2,4,6],[1,3,5,7]])

kek


# In[94]:


kek.shape


# In[95]:


kek.flatten()


# In[96]:


test['rewards'].flatten().size


# In[100]:


f, ax = plt.subplots(ncols=2, nrows=1, figsize=(15,6))
ax[0].plot(np.arange(3000),test['rewards'].flatten()[:3000])
ax[0].set_title('early training')
ax[1].plot(np.arange(test['rewards'].size-3000,test['rewards'].size),test['rewards'].flatten()[-3000:])
ax[1].set_title('late training')


# In[103]:


[x for x in files if (a2cworking[0] in x) and ('A2C' in x)]


# In[107]:


# final list
flist = sorted([x for x in files if x.endswith('0.npz') and ('A2C' in x)])
len(flist)


# In[114]:


f, ax = plt.subplots(nrows=20, ncols=2, sharey='row',figsize=(18,4*20))

for i, targfile in enumerate(flist):
    target = np.load(targfile, allow_pickle=True)
    rews = target['rewards'].flatten()
    taskname = os.path.split(targfile)[1][4:-6]
    ax[i][0].plot(np.arange(1000), rews[:1000])
    ax[i][1].plot(np.arange(rews.size-1000,rews.size), rews[-1000:])
    ax[i][0].set_ylabel(f'{taskname} rewards')
    
ax[0][0].set_title('early training')
ax[0][1].set_title('late training')
ax[-1][0].set_xlabel('timestep')
ax[-1][1].set_xlabel('timestep')
plt.show()


# In[127]:


flist


# In[126]:


f, ax = plt.subplots(nrows=20,figsize=(18, 4*20))

for i, targfile in enumerate(flist):
    target = np.load(targfile, allow_pickle=True)
    rews = target['rewards'].flatten()
    window = int(rews.size*0.01)
    taskname = os.path.split(targfile)[1][4:-6]
    ax[i].plot(pd.Series(rews).rolling(window).mean())
    #ax[i][1].plot(np.arange(rews.size-1000,rews.size), rews[-1000:])
    ax[i].set_ylabel(f'{taskname} rewards')
    ax[i].set_title(f'{taskname} rewards (rolling window = {window})')
    
#x[0][0].set_title('early training')
#x[0][1].set_title('late training')
ax[-1].set_xlabel('step')
plt.show()


# In[129]:


# test with flist[2]
test = np.load(flist[2], allow_pickle=True)
test.files


# In[130]:


for item in test.files:
    print(item, test[item].shape)


# In[133]:


kek = np.arange(4*3*2).reshape(4,3,2)
kek


# In[138]:


rews = test['rewards'].flatten()
obs = test['observations'].reshape(-1,5)
acts = test['actions'].flatten()


# In[147]:


nsteps=100
f, ax = plt.subplots(nrows=3, ncols=1, figsize=(9,10), sharex=True)
ax[0].imshow(obs[-nsteps:,:].T, aspect='auto')
ax[0].set_title('task')
ax[0].set_ylabel('obs')
ax[1].plot(acts[-nsteps:], c='tab:blue', marker='+')
ax[1].set_ylabel('acts')
ax[2].plot(rews[-nsteps:], c='tab:red')
ax[2].set_ylabel('rews')
plt.show()


# In[149]:


flist[2]


# In[152]:


for item in flist:
    test = np.load(item, allow_pickle=True)
    rews = test['rewards'].flatten()
    obs = test['observations'].reshape(-1,test['observations'].shape[-1])
    acts = test['actions'].flatten()
    f, ax = plt.subplots(nrows=3, ncols=1, figsize=(9,10), sharex=True)
    ax[0].imshow(obs[-nsteps:,:].T, aspect='auto')
    ax[0].set_title(os.path.split(item)[1][:-6])
    ax[0].set_ylabel('obs')
    ax[1].plot(acts[-nsteps:], c='tab:blue', marker='+')
    ax[1].set_ylabel('acts')
    ax[2].plot(rews[-nsteps:], c='tab:red')
    ax[2].set_ylabel('rews')
    plt.show()


# In[180]:


for item in test.files:
    print(item, test[item].shape)


# In[181]:


for item in [rews, obs, acts]:
    print(item.shape)


# In[182]:


len(test['rewards'].shape)


# In[6]:


nsteps = 100
initial_s = 0
test = np.load('/home/jordi/Repos/pkgs/neurotests/out/serrano2.npz', allow_pickle=True)
if len(test['rewards'].shape)==3:
    rews = test['rewards'].reshape(-1,test['rewards'].shape[-1]) #flatten()
else:
    rews = test['rewards'].flatten()
    
obs = test['observations'].reshape(-1,test['observations'].shape[-1])
acts = test['actions'].reshape(-1,test['actions'].shape[-1])
f, ax = plt.subplots(nrows=3, ncols=1, figsize=(9,10), sharex=True)
ax[0].imshow(obs[initial_s:initial_s+nsteps,:].T, aspect='auto')
ax[0].set_title('serrano')
ax[0].set_ylabel('obs')
ax[1].plot(acts[initial_s:initial_s+nsteps, 0], c='tab:blue', marker='+')
ax[1].plot(acts[initial_s:initial_s+nsteps, 1], c='tab:orange', marker='+')
ax[1].set_ylabel('acts')
ax[1].axhline(y=0, c='grey', ls=':')
ax[2].plot(rews[initial_s:initial_s+nsteps], c='tab:red')
ax[2].set_ylabel('rews')
ax[0].set_xlim([0,nsteps])
plt.show()


# In[210]:


(rews==rews.max()).sum()


# In[219]:


sns.distplot(acts[:,1])


# In[213]:


rews.shape[0]/100


# In[195]:


a = np.random.uniform(0.8, 1.2)


print(round(a-0.01, 3), round(a+0.01, 3))

print(round(a-0.01-0.03, 3), round(a+0.01+0.03, 3))
print(f'range = {(0.01+0.03)*2} represents {(100*(0.01+0.03)*2)/0.4} %')


# In[161]:


100*np.isnan(acts).sum()/acts.size


# In[165]:


f, ax = plt.subplots(figsize=(9,3))
plt.plot(pd.Series(rews).rolling(int(1e7*0.01)).mean())
plt.ylabel('reward')
plt.show()


# In[2]:


from tqdm import tqdm
def our_reward(action, gt):
    r = 1/((1+abs(action-gt))**2)
    return r

for i,j in tqdm(zip(np.random.uniform(-100,100,int(1e8)),np.random.uniform(0, 1., int(1e8)))):
    if our_reward(i,j)>1:
        print(f'got it with action={i} and gt={j}')
    


# ### 2nd BATCH (30th)

# In[9]:


import neurogym as ngym


# In[8]:


# check integrity...
# complete training should have .png
ok_run_tasks = []
pngs = list(set([str(x).split('/')[-2][:-2] for x in Path(data_pth).glob('**/*.png')]))
print(pngs)


# In[19]:


df = pd.DataFrame(np.zeros((6,len(ngym.all_tasks.keys()))).T)


# In[21]:


df.columns = ['task', 'A2C', 'ACER', 'ACKTR', 'PPO2', 'SL']
df.shape


# In[22]:


df['task'] = sorted(list(ngym.all_tasks.keys()))


# In[25]:


df.iloc[:,1:]=False
df.head()


# In[29]:


for algo in ['A2C', 'ACER', 'ACKTR', 'PPO2', 'SL']:
    subset = [x[len(algo)+1:] for x in pngs if x.startswith(algo)]
    #print(subset)
    for item in subset:
        df.loc[df.task==item, algo] = True
        
df.head()


# In[44]:


## PPO2: (old)minibatches error ~ not included in new bsls
## ACKTR: still not enough time to finnish

plt.figure(figsize=(2.5,13))
plt.imshow(df.iloc[:,1:].values, aspect='auto')
plt.yticks(np.arange(26),df['task'].values)
plt.xticks(np.arange(5), ['A2C', 'ACER', 'ACKTR', 'PPO2', 'SL'], fontsize=8)
plt.title('plotting integrity')
plt.show()


# In[54]:


import tqdm
errors= {}

# we discard the obvious ones which always fail
for algo in ['A2C', 'ACER', 'SL']:
    errors[algo]={}
    for task in tqdm.tqdm(sorted(ngym.all_tasks.keys())):
        if not df.loc[df.task==task, algo].values:
            errors[algo][task] = 'pending'          
            
errors


# In[ ]:


# idk why im generating this infernal nested dict
errors['A2C']['AngleReproduction-v0'] = 'IndexError: index 5 is out of bounds for axis 1 with size 3 (keras.utils.to_categorical)'
errors['A2C']['AntiReach-v0'] = 'IndexError: index 3 is out of bounds for axis 1 with size 3 (keras.utils.to_categorical)'
errors['A2C']['Bandit-v0'] = "AttributeError: 'Bandit' object has no attribute 'obs' || @ line 63, 42"
errors['A2C']['ChangingEnvironment-v0'] = "fixed"
errors['A2C']['DawTwoStep-v0'] = "AttributeError: 'DawTwoStep' object has no attribute 'hi_state' + TODOED"
# delayed match category:
"""File "/home/hcli64/hcli64348/neurogym/neurogym/envs/delaymatchcategory.py", line 91, in new_trial
    self.add_epoch('decision', after='test', last_epoch=True)
  File "/home/hcli64/hcli64348/neurogym/neurogym/core.py", line 167, in add_epoch
    duration = (self.timing_fn[epoch]() // self.dt) * self.dt
KeyError: 'decision'"""


# In[56]:


list(errors['SL'].keys())


# ### 3rd batch (3rd Feb)

# In[2]:


import neurogym as ngym
from neurogym.utils.plotting import plot_rew_across_training
#builtint
#test = '/home/jordi/Repos/pkgs/data/30th/A2C_ReadySetGo-v0_0'
test = '/home/jordi/Repos/pkgs/data/30th/A2C_GNG-v0_0'

plot_rew_across_training(folder=test, window=1000)


# In[4]:


targ_dir = '/home/jordi/Repos/pkgs/data/30th/SL_CVLearning-v0_0/99/'
targ = targ_dir+[x for x in os.listdir(targ_dir) if x.endswith('.npz')][0]
targ


# In[5]:


n = np.load(targ, allow_pickle=True)
n.files


# In[6]:


n['reward'].mean()


# In[7]:


len(ngym.envs.ALL_ENVS.keys())


# In[8]:


perf = pd.DataFrame(np.ones((len(ngym.envs.ALL_ENVS.keys()), 6)))
perf.columns = ['task', 'A2C', 'ACER', 'ACKTR', 'PPO2', 'SL']
perf['task'] = sorted(list(ngym.envs.ALL_ENVS.keys()))
for col in ['A2C', 'ACER', 'ACKTR', 'PPO2', 'SL']:
    perf[col] = np.nan
perf.head()


# In[9]:


def get_perf(algo, task ,seeds=2, ntr=100000):
    try:
        perf = []
        for i in range(seeds):
            basedir = f'/home/jordi/Repos/pkgs/data/3rd/{algo}_{task}_{i}/{task[:-3]}_bhvr_data_{ntr}.npz' 
            n = np.load(basedir, allow_pickle=True)
            perf += n['reward'].tolist()
        return np.array(perf).mean()
    except:
        #print(f'crash with {algo} & {task}')
        return np.nan
    
def get_perf_SL(task ,seeds=2, ntr=100000):
    try:
        perf = []
        for i in range(seeds):
            basedir = f'/home/jordi/Repos/pkgs/data/3rd/SL_{task}_{i}/{task[:-3]}_bhvr_data_1000.npz'  # forgot to add dir with iter
            n = np.load(basedir, allow_pickle=True)
            perf += n['reward'].tolist()
        return np.array(perf).mean()
    except:
        #print(f'crash with {algo} & {task}')
        return np.nan
    
for current_task in perf.task.values.tolist():
    perf.loc[perf.task==current_task, 'SL'] = get_perf_SL(current_task)
    for alg in ['A2C', 'ACER', 'ACKTR', 'PPO2']: # 'SL']:
        #print(get_perf(alg, current_task))
        perf.loc[perf.task==current_task, alg]=get_perf(alg, current_task)


# In[10]:


# plt.figure(figsize=(4,26))
# plt.imshow(perf[['A2C', 'ACER', 'ACKTR', 'PPO2', 'SL']].values,aspect='auto')
# plt.yticks(np.arange(perf.shape[0]),perf['task'].values)
# plt.xticks(np.arange(5), ['A2C', 'ACER', 'ACKTR', 'PPO2', 'SL'], fontsize=8)
# plt.title('performance @ 100k trials')
# plt.show()


# In[14]:


f, ax = plt.subplots(figsize=(5,26))
sns.heatmap(perf[['A2C', 'ACER', 'ACKTR', 'PPO2', 'SL']].values, annot=True, fmt='.2f',ax=ax, cmap='viridis', cbar=False)
# annot=np.flipud(nmat), fmt='.0f',ax=ax
ax.set_yticks(np.arange(perf.shape[0])+0.5)
ax.set_yticklabels([x[:-3] for x in perf['task'].values], rotation='horizontal')
ax.set_xticks(np.arange(5)+0.5)
ax.set_xticklabels(['A2C', 'ACER', 'ACKTR', 'PPO2', 'SL'], fontsize=8)
ax.set_title('performance @ 100k trials')
plt.show()


# In[12]:


def second_round(idx, col):
    """heheh"""
    curr_task = perf.loc[idx, 'task']
    all_perf = []
    try:
        for i in [0,1]: # seeds
            d = f'/home/jordi/Repos/pkgs/data/3rd/{col}_{curr_task}_{i}/'
            allfiles = [int(x.split('_')[-1][:-4]) for x in os.listdir(d) if x.endswith('.npz')]
            targ = [x for x in os.listdir(d) if str(max(allfiles)) in x]
            #print(targ)
            arr = np.load(d+targ[0], allow_pickle=True)
            all_perf += arr['reward'].tolist()

        return np.array(all_perf).mean()
    except Exception as e:
        print(f'{curr_task} failed\n{e}')
        return np.nan
        
#perf['ACER'].isna()
second_round(25, 'ACER')


# In[13]:


for curr_col in ['A2C', 'ACER', 'PPO2']:
    for i in perf.loc[perf[curr_col].isna()].index.values:
        perf.loc[i, curr_col] = second_round(i, curr_col)


# In[32]:


#f = np.load('/home/jordi/Repos/pkgs/data/3rd/A2C_Detection-v0_0/Detection_bhvr_data_89000.npz', allow_pickle=True)
f = np.load('/home/jordi/Repos/pkgs/data/3rd/A2C_DelayedComparison-v0_0/DelayedComparison_bhvr_data_100000.npz', allow_pickle=True)

#f['reward'].mean()
sns.distplot(f['reward'], kde=False)


# In[15]:


#so apparently model is missing from many folders
#import pathlib
files = sorted([str(x) for x in Path('/home/jordi/Repos/pkgs/data/3rd/').glob('*0/model.zip')])
files


# In[16]:


# should be ok (ie model trained but need dataset >1000trials) in: 
# ReachingDelayResponse
# 


# In[3]:


from neurogym.utils.plotting import plot_env
import importlib
from neurogym.custom_timings import ALL_ENVS_MINIMAL_TIMINGS
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
    


# In[31]:


import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

files = sorted([str(x) for x in Path('/home/jordi/Repos/pkgs/data/3rd/').glob('*0/model.zip')])
for f in files:
    custom_plot_env(f, num_steps_env=100)


# In[26]:


#custom_plot_env('A2C', )
import shutil
dest_dir = '/home/jordi/DATA/Documents/remote_code/share/'

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for f in files:
    mod = f.split('/')[-2]
    if not os.path.exists(dest_dir+mod+'/'):
        os.makedirs(dest_dir+mod+'/')
    shutil.copyfile(f, dest_dir+mod+'/model.zip')


# In[21]:


get_ipython().system('pwd')


# In[4]:


custom_plot_env('/home/jordi/Repos/pkgs/trash/A2C_DelayedMatchCategory-v0_0/model.zip', num_steps_env=100)


# In[14]:


# targ
#toplot = '/home/jordi/Repos/pkgs/trash/A2C_DelayedMatchCategory-v0_0/DelayedMatchCategory_bhvr_data_149000.npz'
#toplot = '/home/jordi/Repos/pkgs/trash/A2C_DelayedComparison-v0_0/DelayedComparison_bhvr_data_140000.npz'
toplot = '/home/jordi/Repos/pkgs/data/3rd/A2C_DelayedComparison-v0_0/DelayedComparison_bhvr_data_100000.npz'
toplot = np.load(toplot, allow_pickle=True)
toplot.files


# In[15]:


acts.shape


# In[16]:


obs = toplot['stimulus']
gt = toplot['gt']
acts = toplot['choice']
rews = toplot['reward']
nsteps=100
f, ax = plt.subplots(nrows=3, ncols=1, figsize=(9,10), sharex=True)
ax[0].imshow(obs[-nsteps:,:].T, aspect='auto')
ax[0].set_title('DelayedComparison')
ax[0].set_ylabel('obs')
ax[1].plot(acts[-nsteps:], c='tab:blue', marker='+')
ax[1].plot(gt[-nsteps:], c='tab:red')
ax[1].set_ylabel('acts')
ax[2].plot(rews[-nsteps:], c='tab:red')
ax[2].set_ylabel('rews')
plt.show()


# In[1]:


# so model actually learns (according to monitor wrapper)the issue is somewhere when reusing plotting functions


# ### what's wrong when reusing models

# In[1]:


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL)

from stable_baselines import A2C
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
import gym
import neurogym 
from neurogym.utils.plotting import plot_env
import importlib

from neurogym.custom_timings import ALL_ENVS_MINIMAL_TIMINGS

def custom_plot_env(modelpath, num_steps_env=100):
    root_str = os.path.split(modelpath)[0].split('/')[-1]
    algo = root_str.split('_')[0]
    task = root_str.split('_')[1]
    seed = root_str.split('_')[-1]
    ngym_kwargs = {'dt':100, 'timing': ALL_ENVS_MINIMAL_TIMINGS[task]}
    env = gym.make(task, **ngym_kwargs)
    env = DummyVecEnv([lambda: env])
    pkg = importlib.import_module('stable_baselines') #+algo) 
    module = getattr(pkg, algo)
    ###new shit
    model = module(LstmPolicy, env, verbose=0, n_steps=20, n_cpu_tf_sess=1, policy_kwargs={'feature_extraction':'mlp'}) 
    ###
    model = module.load(modelpath)
    plot_env(env, num_steps_env=num_steps_env, model=model, name=f'{algo} on {task}', fig_kwargs={'figsize':(10, 12)})


# In[4]:


toplot = '/home/jordi/Repos/pkgs/data/3rd/A2C_DelayedComparison-v0_0/model.zip' #DelayedComparison_bhvr_data_100000.npz'
#toplot = '/home/jordi/Repos/pkgs/data/3rd/A2C_DelayedMatchToSampleDistractor1D-v0_0/model.zip'
#toplot = '/home/jordi/Repos/pkgs/data/3rd/A2C_DelayedMatchCategory-v0_0/model.zip'
custom_plot_env(toplot)


# In[ ]:


# i still do not know why, when saving monitor everyingle iter it looks as it is learning


# ### learning RL

# In[5]:


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL)

from stable_baselines import A2C
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
import gym
import neurogym 
from neurogym.utils.plotting import plot_env
import importlib

from neurogym.custom_timings import ALL_ENVS_MINIMAL_TIMINGS


# In[6]:


tasks = sorted(list(ALL_ENVS_MINIMAL_TIMINGS.keys()))
motherdir='/home/jordi/Repos/pkgs/data/3rd/'
tasks


# In[7]:


pth = motherdir+f'A2C_{tasks[0]}_0/'

f, ax = plt.subplots(figsize=(8,8))

neurogym.utils.plotting.plot_rew_across_training(pth, window=0.05,ax=ax, fkwargs={'c':'tab:orange', 'ls':'--', 'alpha':0.5, 'label':'test'})
plt.legend()
plt.show()


# In[8]:


#len(tasks)
f, ax = plt.subplots(nrows=13, ncols=2, figsize=(18, 13*3))
ax = ax.flatten()
cols = sns.color_palette()
for i, task in enumerate(tasks):
    for j,alg in enumerate(['A2C', 'ACER', 'PPO2']):
        for k in [0,1]:
            pth = f'{motherdir}{alg}_{task}_{k}/'
            try:
                neurogym.utils.plotting.plot_rew_across_training(pth, window=0.05,ax=ax[i], fkwargs={'c':cols[j], 'ls':'--', 'alpha':0.5, 'label':alg},
                                                                 ytitle=task, legend=True, zline=True)
            except:
                continue
            
#plt.legend()
plt.suptitle('bsc training Jan3rd')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('./monster.png')
plt.show()


# In[ ]:




