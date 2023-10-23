## "Copyright (c) Meta Platforms, Inc. and affiliates"


import torch
import numpy as np
import os
import pickle
import argparse
from einops import rearrange

from constants import CAMERA_NAMES, TEXT_EMBEDDINGS


from policy import ACTPolicy, CNNMLPPolicy


import gym
import click
import numpy as np
import time
import os
from mj_envs.logger.grouped_datasets import Trace

def main(args):

    taskname = args['task_name']

    if 'open_drawer' in taskname:
        task_emb = TEXT_EMBEDDINGS[0]
    elif 'close_drawer' in taskname:
        task_emb = TEXT_EMBEDDINGS[1]
    elif 'pick_butter' in taskname:
        task_emb = TEXT_EMBEDDINGS[2]            
    elif 'place_butter' in taskname:
        task_emb = TEXT_EMBEDDINGS[3]
    elif 'pick_toast' in taskname:
        task_emb = TEXT_EMBEDDINGS[4]
    elif 'place_toast' in taskname:
        task_emb = TEXT_EMBEDDINGS[5] 
    elif 'cap_lid' in taskname:
        task_emb = TEXT_EMBEDDINGS[6] 
    elif 'pick_lid' in taskname:
        task_emb = TEXT_EMBEDDINGS[7]
    elif 'pick_tea' in taskname:
        task_emb = TEXT_EMBEDDINGS[8]
    elif 'place_lid' in taskname:
        task_emb = TEXT_EMBEDDINGS[9] 
    elif 'place_tea' in taskname:
        task_emb = TEXT_EMBEDDINGS[10]
    elif 'uncap_lid' in taskname:
        task_emb = TEXT_EMBEDDINGS[11]
    elif 'close_oven' in taskname:
        task_emb = TEXT_EMBEDDINGS[12]
    elif 'open_oven' in taskname:
        task_emb = TEXT_EMBEDDINGS[13]
    elif 'place_bowl' in taskname:
        task_emb = TEXT_EMBEDDINGS[14]
    elif 'slide_out' in taskname:
        task_emb = TEXT_EMBEDDINGS[15]
    elif "cap_mug" in taskname:
        task_emb = TEXT_EMBEDDINGS[16]
    elif "pick_mug" in taskname:
        task_emb = TEXT_EMBEDDINGS[17]
    elif "pick_towel" in taskname:
        task_emb = TEXT_EMBEDDINGS[18]
    elif "wipe_towel" in taskname:
        task_emb = TEXT_EMBEDDINGS[19]
    elif "pick_cup" in taskname:
        task_emb = TEXT_EMBEDDINGS[20]
    elif "place_cup" in taskname:
        task_emb = TEXT_EMBEDDINGS[21]
    else:
        task_emb = TEXT_EMBEDDINGS[0]
        'SINGLE TASK embedding wont be used'
    print(taskname, len(task_emb))
    task_emb = np.asarray(task_emb)
    task_emb = torch.from_numpy(task_emb).float().cuda()
    task_emb = task_emb.unsqueeze(0)

    ## robohive args
    env_name = args['env_name']

    mode = args['mode']
    horizon = args['horizon']
    num_repeat = args['num_repeat']
    render = args['render']
    camera_name = args['camera_name']
    frame_size = args['frame_size']
    output_dir = args['output_dir']
    output_name = args['output_name']
    save_paths = args['save_paths']
    compress_paths = args['compress_paths']
    plot_paths = args['plot_paths']
    env_args = args['env_args']
    noise_scale = args['noise_scale']

    # command line parameters
    # is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    # dataset_dir = args['dataset_dir']
    policy_class = args['policy_class']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # fixed parameters
    num_episodes = 200 ## VHANGE IT
    state_dim = 8
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'state_dim': state_dim,
        'lr': args['lr'],
        'real_robot': 'TBD',
        'policy_class': policy_class,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg']
    }


    policy_config['camera_names'] = CAMERA_NAMES
    config['camera_names'] = CAMERA_NAMES
    config['real_robot'] = True
    config['episode_len'] = 100

    ckpt_names = [f'policy_best.ckpt']
    ckpt_name = ckpt_names[0]
    # eval_bc(config, ckpt_name, save_episode=True)


    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'main'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    ### evaluation loop
    if temporal_agg:
        all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

    qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
    image_list = [] # for visualization
    qpos_list = []
    target_qpos_list = []
    rewards = []

    np.random.seed(123)
    env = gym.make(env_name) if env_args==None else gym.make(env_name, **(eval(env_args)))
    env.seed(123)
    env.mujoco_render_frames = False

    # Rollout paths
    path_horizon=horizon
    trace = Trace("Rollouts Trajectories")
    for i_loop in range(num_repeat):

        # Rollout path
        print("Starting rollout loop:{}".format(i_loop))
        # for path_name, path_data in paths.items():

        # initialize path -----------------------------
        ep_t0 = time.time()
        path_name = f"Trial{i_loop}"
        # path_name+='-'+str(i_loop)
        print("Starting {} rollout".format(path_name))
        trace.create_group(path_name)

        env.reset()

        # Rollout path --------------------------------
        obs, rwd, done, env_info = env.forward()
        path_horizon = horizon
        print(path_horizon)
        for i_step in range(path_horizon):
            t = i_step


            if mode=='policy':
                with torch.inference_mode():
                    
                    # camera_names = ['rgb_left','rgb_top','rgb_right','rgb_wrist']
                    ### Change these for real robot?
                    camera_names = ['rgb:left_cam:240x424:2d','rgb:right_cam:240x424:2d','rgb:top_cam:240x424:2d','rgb:Franka_wrist_cam:240x424:2d']
                    # camera_names = ['rgb:right_cam:240x424:2d']
                    image_dict = dict()
                    for cam_name in camera_names:
                        #image_dict[cam_name] = path_data[f'{cam_name}'][t]
                        image_dict[cam_name] = env_info['obs_dict'][f'{cam_name}']
                    
                    curr_images = []
                    for cam_name in camera_names:
                        curr_image = rearrange(image_dict[cam_name], 'h w c -> c h w')
                        curr_images.append(curr_image)
                    print("curr_images",len(curr_images))
                    curr_image = np.stack(curr_images, axis=0)
                    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

                    # qpos_numpy = path_data['qp_arm'][t].astype(np.float32)
                    qpos_numpy = env_info['obs_dict']['qp_arm'].astype(np.float32)
                    qpos = pre_process(qpos_numpy)
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    #qpos_history[:, t] = qpos

                    # print('path_data',path_data)
                    # print('qpos',qpos)
                    # print('curr_image',curr_image)
                    # print('t',t)
                    

                    ### query policy
                    if config['policy_class'] == "ACT":
                        if t % query_frequency == 0:
                            all_actions =  policy(qpos, curr_image,task_emb=task_emb)
                            print('SAMPLED ACTION')
                        temporal_agg = True
                        if temporal_agg:
                            all_time_actions[[t], t:t+num_queries] = all_actions
                            actions_for_curr_step = all_time_actions[:, t]
                            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
                            k = 0.01
                            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                            exp_weights = exp_weights / exp_weights.sum()
                            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                            print('TEMPORAL AGG')
                        else:
                            raw_action = all_actions[:, t % query_frequency]
                    elif config['policy_class'] == "CNNMLP":
                        raw_action = policy(qpos, curr_image)
                    else:
                        raise NotImplementedError

                    ### post-process actions
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = post_process(raw_action)
                    target_qpos = action

                    act = target_qpos
                    #add gaussian noise
                    act = act + env.env.np_random.normal(loc=0.0, scale=0.025, size=len(act))
                    print(act)
                    print(f"STEP: {i_step}")

            # nan actions for last log entry
            if i_step == path_horizon:
                act = np.nan*np.ones(env.action_space.shape)

            # log values at time=t ----------------------------------
            if compress_paths:
                obs = [] # don't save obs, env_infos has obs_dict
                del env_info['state']  # don't save state, obs_dict has env necessities

            # log: time, obs, act, rwd, info, done
            datum_dict = dict(
                    time=env.time,
                    observations=obs,
                    actions=act.copy(),
                    rewards=rwd,
                    env_infos=env_info,
                    done=done,
                )
            
            trace.append_datums(group_key=path_name,dataset_key_val=datum_dict)
            if i_step < path_horizon: #incase last step actions (nans) can cause issues in step
                obs, rwd, done, env_info = env.step(act)
                
        print("-- Finished %s rollout in %2.3fs" % (path_name, time.time()-ep_t0))
        # Finish loop
        print("Finished rollout loop:{}".format(i_loop))
        user_cmt = input("Enter 1 for success, 0 for failure, 0.25 for good target, 0.5 for attempted action, 0.75 failed to complete task: ") # a string
        while (isinstance(user_cmt, float)): #ensure valid input and offer correction if mistake
            print('an input other than 1, 0, or -1 was entered; try again: ')
            user_cmt = input()
            while user_cmt == '': #in case Enter key is hit without selecting valid input
                user_cmt = input()
        user_cmt=float(user_cmt)
        datum_dict['user_cmt'] = user_cmt
        trace.append_datums(group_key=path_name,dataset_key_val=datum_dict)
    ti = time.localtime()
    trace.save(f"eval_{ti.tm_hour}_{ti.tm_min}_{num_queries}.h5")   

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset_dir', action='store', type=str, help='dataset_dir', required=False, default="/mnt/raid5/data/roboset/v0.4/setting_table_close_drawer_scene_1/")
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=False, default="ACT")

    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=2)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=False, default=0)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=False, default=1000)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=False, default=1e-04)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False, default=10)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False, default=10)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False, default=256)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False, default=2048)
    parser.add_argument('--temporal_agg', action='store', type=bool, default=True)
    parser.add_argument('-e', '--env_name', type=str, help='environment to load', required=True, default='rpFrankaRobotiqData-v0')

    parser.add_argument('-m', '--mode', type=click.Choice(['record', 'render', 'playback', 'recover','policy']), help='How to examine rollout', default='policy')
    parser.add_argument('-hor', '--horizon', type=int, help='Rollout horizon, when mode is record', default=100)
    parser.add_argument('-num_repeat', '--num_repeat', type=int, help='number of repeats for the rollouts', default=10)
    parser.add_argument('-r', '--render', type=click.Choice(['onscreen', 'offscreen', 'none']), help='visualize onscreen or offscreen', default='none')
    parser.add_argument('-c', '--camera_name', type=str, default=[None,], help=('list of camera names for rendering'))
    parser.add_argument('-fs', '--frame_size', type=tuple, default=(424, 240), help=('Camera frame size for rendering'))
    parser.add_argument('-o', '--output_dir', type=str, default='/checkpoint/homanga/cactiv2/robohivelogs', help=('Directory to save the outputs'))
    parser.add_argument('-on', '--output_name', type=str, default=None, help=('The name to save the outputs as'))
    parser.add_argument('-sp', '--save_paths', type=bool, default=False, help=('Save the rollout paths'))
    parser.add_argument('-cp', '--compress_paths', type=bool, default=True, help=('compress paths. Remove obs and env_info/state keys'))
    parser.add_argument('-pp', '--plot_paths', type=bool, default=False, help=('2D-plot of individual paths'))
    parser.add_argument('-ea', '--env_args', type=str, default="{\'is_hardware\':True}", help=('env args. E.g. --env_args "{\'is_hardware\':True}"'))
    parser.add_argument('-ns', '--noise_scale', type=float, default=0.0, help=('Noise amplitude in randians}"'))

    parser.add_argument('--task_name', type=str, default='open_drawer', help=('task name for multitask'))
    # add this for multi-task embedding condition
    parser.add_argument('--multi_task', action='store_true')

    main(vars(parser.parse_args()))



# python evaluate.py -e rpFrankaRobotiqData-v0 -p /checkpoint/jayvakil/v0.4/setting_table_close_drawer_scene_1/setting_table_close_drawer_scene_1_20230308-120120.h5 -m playback -f RoboSet -r none

#python eval_robot.py -e rpFrankaRobotiqDataRP04-v0 --ckpt_dir ckpt/ --chunk_size $CHUNK -ns 0.01 --num_repeat 10

#python eval_robot.py -e rpFrankaRobotiqDataRP02-v0 --ckpt_dir ckpt/rp02_manga_policies/chunk20/drawer_close --chunk_size 20 --num_repeat 10 --task_name close_drawer

#python eval_robot_multi_task.py -e rpFrankaRobotiqDataRP02-v0 --ckpt_dir ckpt/april7multitask --policy_class ACT --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 --num_repeat 10 --task_name pick_butter --multi_task
