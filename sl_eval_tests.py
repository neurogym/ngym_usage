import bsc_run
from copy import deepcopy as deepc
import numpy as np
import importlib
import neurogym as ngym

# n_ch = 2
# seed = 0
# main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
#     'var_nch_predef_mats_larger_nets/'

n_ch = 2
seed = 36
main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
    'variable_nch_predef_tr_mats/'

folder = main_folder + 'train_mode_SL_seed_'+str(seed)+'_n_ch_'+str(n_ch)+'/'
expl_params = {'n_ch': n_ch, 'folder': folder, 'seed': 0}

spec = importlib.util.spec_from_file_location("params", main_folder+"/params.py")
params = importlib.util.module_from_spec(spec)
spec.loader.exec_module(params)
gen_params = params.general_params
bsc_run.update_dict(gen_params, expl_params)
# update task params
task_params = params.task_kwargs[gen_params['task']]
bsc_run.update_dict(task_params, expl_params)
# update wrappers params
wrappers_kwargs = params.wrapps
for wrap in wrappers_kwargs.keys():
    params_temp = wrappers_kwargs[wrap]
    bsc_run.update_dict(params_temp, expl_params)
# update task params
sl_kwargs = params.sl_kwargs

n_lstm = gen_params['n_lstm']
task = 'NAltPerceptualDecisionMaking-v0'
stps_ep = sl_kwargs['steps_per_epoch']
wraps_sl = deepc(wrappers_kwargs)
del wraps_sl['PassAction-v0']
del wraps_sl['PassReward-v0']
# update wrappers params
env = bsc_run.make_env(env_id=task, rank=0, seed=seed, wrapps=wraps_sl,
                       **task_params)()
obs_size = env.observation_space.shape[0]
act_size = env.action_space.n
model_test = bsc_run.define_model(seq_len=20, batch_size=64,
                                  obs_size=obs_size, act_size=act_size,
                                  stateful=sl_kwargs['stateful'],
                                  num_h=n_lstm,
                                  loss=sl_kwargs['loss'])
ld_f = folder+'model_'+str(stps_ep)+'_steps'.replace('//', '/')
print('loading: ', ld_f)
model_test.load_weights(ld_f)


dataset = ngym.Dataset(env, batch_size=sl_kwargs['btch_s'],
                       seq_len=20, batch_first=True)
data_generator = (dataset()
                  for i in range(stps_ep))
model_test.fit(data_generator, verbose=1,
               steps_per_epoch=stps_ep)



env.reset()
# obs = env.ob[0]
act_mat = []
gt_mat = []
rew_mat = []
obs_mat = []
for ind_stp in range(sl_kwargs['test_steps']):
    if ind_stp % 1000 == 0:
        print(ind_stp)
    obs = env.ob_now
    obs_mat.append(obs)
    obs = obs[np.newaxis]
    obs = obs[np.newaxis]
    action = model_test.predict(obs)
    action = np.argmax(action, axis=-1)[0]
    _, rew, _, info = env.step(action[0])
    act_mat.append(action[0])
    gt_mat.append(info['gt'])
    rew_mat.append(rew)
    # print('--')
    # print(action[0])
    # if info['new_trial']:
    #     print(rew)
    #     print(info['gt'])
    #     print('xxxxxxxxxxxxxxxxxxx')
