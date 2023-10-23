## Copyright (c) Meta Platforms, Inc. and affiliates

import numpy as np
import torch
import os
import h5py
from torch.utils.data import DataLoader
from constants import CAMERA_NAMES, TEXT_EMBEDDINGS
import random
import glob 

class EpisodicDatasetRobopen(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, norm_stats,num_episodes):
        super(EpisodicDatasetRobopen).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.norm_stats = norm_stats
        self.num_episodes = num_episodes
        self.is_sim = None
        self.h5s = []
        self.trials = []
        self.task_emb_per_trial = []
        self.verbose = True
        self.h5s = {}
        lens = []
        
        files = list()
        randfiles = list()
        n = 100
        for i in glob.glob("{}/*/*/".format(dataset_dir)):
            print(i)
            randfiles.append(sorted(glob.glob(os.path.join(i, "*.h5"))))

        for i in randfiles:
            l = len(i)
            num = int(n/250 * l)
            for file in range(num):
                files.append(i[file])

        files = sorted(files)

        # files = sorted(glob.glob(os.path.join(dataset_dir + "*/*/", '*.h5')))
        for filename in files:
            #for 20 tasks hardcoded, modify as needed
            if 'open_drawer' in filename:
                task_emb = TEXT_EMBEDDINGS[0]
            elif 'close_drawer' in filename:
                task_emb = TEXT_EMBEDDINGS[1]
            elif 'pick_butter' in filename:
                task_emb = TEXT_EMBEDDINGS[2]            
            elif 'place_butter' in filename:
                task_emb = TEXT_EMBEDDINGS[3]
            elif 'pick_toast' in filename:
                task_emb = TEXT_EMBEDDINGS[4]
            elif 'place_toast' in filename:
                task_emb = TEXT_EMBEDDINGS[5] 
            elif 'cap_lid' in filename:
                task_emb = TEXT_EMBEDDINGS[6] 
            elif 'pick_lid' in filename:
                task_emb = TEXT_EMBEDDINGS[7]
            elif 'pick_tea' in filename:
                task_emb = TEXT_EMBEDDINGS[8]
            elif 'place_lid' in filename:
                task_emb = TEXT_EMBEDDINGS[9] 
            elif 'place_tea' in filename:
                task_emb = TEXT_EMBEDDINGS[10]
            elif 'uncap_lid' in filename:
                task_emb = TEXT_EMBEDDINGS[11]
            elif 'close_oven' in filename:
                task_emb = TEXT_EMBEDDINGS[12]
            elif 'open_oven' in filename:
                task_emb = TEXT_EMBEDDINGS[13]
            elif 'place_bowl' in filename:
                task_emb = TEXT_EMBEDDINGS[14]
            elif 'slide_out' in filename:
                task_emb = TEXT_EMBEDDINGS[15]
            elif "cap_mug" in filename:
                task_emb = TEXT_EMBEDDINGS[16]
            elif "pick_mug" in filename:
                task_emb = TEXT_EMBEDDINGS[17]
            elif "pick_towel" in filename:
                task_emb = TEXT_EMBEDDINGS[18]
            elif "wipe_towel" in filename:
                task_emb = TEXT_EMBEDDINGS[19]
            elif "pick_cup" in filename:
                task_emb = TEXT_EMBEDDINGS[20]
            elif "place_cup" in filename:
                task_emb = TEXT_EMBEDDINGS[21]
            else:
                task_emb = TEXT_EMBEDDINGS[0]
                'SINGLE TASK embedding wont be used'

            h5 = h5py.File(filename, 'r')
            for key, trial in h5.items():
                if(trial['data']['time'].shape[0] != 42):
                    continue
                # Open the trial and extract metadata
                lens.append(trial['data']['ctrl_arm'].shape[0])
                # Bookkeeping for all the trials
                self.trials.append(trial)
                self.task_emb_per_trial.append(task_emb)

        self.trial_lengths = np.cumsum(lens)
        self.max_idx = self.trial_lengths[-1]
        print("TOTAL TRIALS",len(self.trials))
        self.trials = self.trials[:num_episodes]

        assert self.num_episodes == len(self.trials) ## sanity check that all files are loaded, remove if needed

        print('TOTAL TRIALS = num_episodes = ',len(self.trials))
        self.__getitem__(0)


    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, idx):
        sample_full_episode = False # hardcode
        trial_idx = self.episode_ids[idx]
        trial = self.trials[trial_idx]
        task_emb = self.task_emb_per_trial[trial_idx]
        camera_names = CAMERA_NAMES

        action = np.concatenate([trial['data']['ctrl_arm'], trial['data']['ctrl_ee']], axis=1).astype(np.float32)
        original_action_shape = action.shape
        cutoff = 2 #10#5 
        episode_len = original_action_shape[0] -cutoff ## cutoff last few

        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)
        # get observation at start_ts only
        qpos = trial['data']['qp_arm'][start_ts].astype(np.float32)
        qvel = trial['data']['qv_arm'][start_ts].astype(np.float32)

        image_dict = dict()
        for cam_name in camera_names:
            image_dict[cam_name] = trial['data'][f'{cam_name}'][start_ts]
        # get all actions after and including start_ts
        action = np.concatenate([trial['data']['ctrl_arm'], trial['data']['ctrl_ee']], axis=1)[max(0, start_ts - 1):].astype(np.float32) # hack, to make timesteps more aligned
        action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned 

        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action[:-cutoff]
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        task_emb = torch.from_numpy(np.asarray(task_emb)).float()

        return image_data, qpos_data, action_data, is_pad, task_emb


def get_norm_stats_robopen(dataset_dir,num_epsiodes):
    # files = []
    # for directory in dataset_dir:
    #     files.append()
    files = list()
    randfiles = list()
    n = 100
    for i in glob.glob("{}/*/*/".format(dataset_dir)):
        print(i)
        randfiles.append(sorted(glob.glob(os.path.join(i, "*.h5"))))

    for i in randfiles:
        l = len(i)
        num = int(n/250 * l)
        for file in range(num):
            files.append(i[file])

    files = sorted(files)

    print('files',files)
    all_qpos_data = []
    all_action_data = []
    cutoff = 2 #10#5 
    for filename in files:
        # Check each file to see how many entires it has
        h5 = h5py.File(filename, 'r')
        # with h5py.File(filename, 'r') as h5:
        for key, trial in h5.items():
            # Open the trial and extract metadata
    
            qpos = trial['data']['qp_arm'][()].astype(np.float32)
            qvel = trial['data']['qv_arm'][()].astype(np.float32)
            camera_names = CAMERA_NAMES
            action = np.concatenate([trial['data']['ctrl_arm'], trial['data']['ctrl_ee']], axis=1).astype(np.float32)


            if(trial['data']['time'].shape[0] != 42):
                continue
            all_qpos_data.append(torch.from_numpy(qpos[:-cutoff]))
            all_action_data.append(torch.from_numpy(action[:-cutoff]))
            # if len(qpos)==41:
                # all_qpos_data.append(torch.from_numpy(qpos[:-(cutoff-1)]))
                # all_action_data.append(torch.from_numpy(action[:-(cutoff-1)]))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, 10) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, 10) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, batch_size_train, batch_size_val):
    # obtain train test split
    train_ratio = 0.8 # change as needed
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]
    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats_robopen(dataset_dir, num_episodes)
    # construct dataset and dataloader
    train_dataset = EpisodicDatasetRobopen(train_indices, dataset_dir, norm_stats,num_episodes)
    val_dataset = EpisodicDatasetRobopen(val_indices, dataset_dir, norm_stats,num_episodes)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
