import os
import torch
from torch.utils.data import Dataset
import numpy as np

import bagpy
from bagpy import bagreader
import rosbag

from scipy.spatial.transform import Rotation as R

import yaml

import pandas as pd
class ScrewingDataset(Dataset):
    """

    """
    # if window_size = -1, produce the full episode !
    def __init__(self, bag_path, target_path_pos_ori_list, window_size, overlapping=True, load_in_mem=False):
        # self.rosbag = None
        self.load_in_mem = load_in_mem

        ## for now hardcode the main topic and topics of interest...
        
        # bag_path = '~/datasets/rosbags/input.bag'
        self.bag_path = os.path.expanduser(bag_path)
        if not os.path.exists(self.bag_path):
            raise AssertionError('bag path does not exist')

        # bag = rosbag.Bag(bag_path)
        # bag_yaml_str = bag._get_yaml_info()
        # bag.close()

        # self.bag_yaml = yaml.safe_load(bag_yaml_str)
        # print(self.bag_yaml)

        # self.main_num_msgs = bag.get_message_count(self.main_topic)
        # print(self.main_num_msgs)
        # main_topic_window = 100 # number of messages

        try:
            self.bagreader = bagreader(bag_path)
        except:
            print('reading bag ' + bag_path + ' failed!')
            raise 

        self.main_topic = '/panda/franka_state_controller_custom/franka_states'

        self.total_T = self.bagreader.end_time - self.bagreader.start_time

        table = self.bagreader.topic_table
        self.main_num_msgs = table[table['Topics'] == self.main_topic]['Message Count'].item()

        ## WINDOW SIZE IS THE SAME FOR ALL STREAMS FOR NOW
        # max_window = max(list(time_window_dict.values()))
        self.window_size = window_size

        assert self.main_num_msgs > 1, "must be at least one number of msgs!" 
        assert self.main_num_msgs >= self.window_size, "number of msgs must be geq window size!" 

        self.overlapping = overlapping 

        if self.window_size == -1:
            self._len = 1
        else:
            ## non-overlapping partitioning of the episode
            if not overlapping:
                self._len = (self.main_num_msgs // window_size)
            ## INSTEAD change to overlapping 
            else:
                self._len = self.main_num_msgs - window_size + 1

        self.pose_topics = []
        for i in range(16):
            topic = 'O_T_EE_' + str(i)
            self.pose_topics.append(topic)

        self.wrench_topics = []
        for i in range(6):
            topic = 'K_F_ext_hat_K_' + str(i)
            self.wrench_topics.append(topic)

        self.action_topics = []
        for i in range(6):
            topic = 'O_F_EE_d_' + str(i)
            self.action_topics.append(topic)

        # table = b.topic_table
        # self.topic_list = []
        # self.topic_csv_list = []
        self.topic_csv_dict = {}
        for topic in self.bagreader.topics:
            # print(topic)
            # self.topic_list.append(topic)
            self.topic_csv_dict[topic] = self.bagreader.message_by_topic(topic)
        # print(self.topic_csv_dict)
        ## Read in the target file
        # self.target_paths = 

        # EE pose: O_T_EE_1 ... O_T_EE_15
        # self.time_window_dict = {
        #     'O_T_EE' : 
        # }

        ## read the label
        self.target = np.append(np.load(target_path_pos_ori_list[0]), np.load(target_path_pos_ori_list[1]))

        if load_in_mem:
            ## get features
            self.main_topic_csv = self.topic_csv_dict[self.main_topic] 
            self.df = pd.read_csv(self.main_topic_csv)
        
        # print('succesfully loaded bag!')

    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        # self.main_num_msgs
        if idx >= self._len:
            raise AssertionError('index out of range')

        ## indexing for non-overlapping indexing
        if self.window_size == -1:
            start_idx = 0
            end_idx = self.main_num_msgs - 1
        else:
            if not self.overlapping:
                start_idx = self.window_size * idx
                end_idx = self.window_size*(idx+1)  - 1
            else: ## indexing for overlapping indexing
                start_idx = idx
                end_idx = idx + self.window_size - 1

        if not self.load_in_mem:
            ## get features
            main_topic_csv = self.topic_csv_dict[self.main_topic] 
            df = pd.read_csv(main_topic_csv)
       
            ## get wrenches
            wrenches_np = df.loc[start_idx:end_idx, self.wrench_topics].to_numpy(dtype='float')

            ##  get poses
            ## convert affine tfs to poses
            tfs_np = df.loc[start_idx:end_idx, self.pose_topics].to_numpy(dtype='float')
            poses_np = []
            for i in range(tfs_np.shape[0]):
                # print(tfs_np[i])
                poses_np.append(self.affine_tf_to_pose(tfs_np[i]))

            poses_np = np.array(poses_np)

            ##  get actions
            actions_np = df.loc[start_idx:end_idx, self.action_topics].to_numpy(dtype='float')

            ## get timestamps 
            times_np = df.loc[start_idx:end_idx, 'Time'].to_numpy(dtype='float')
        else: # TODO fix this so just make a shallow copy of self.df
            ## get wrenches
            wrenches_np = self.df.loc[start_idx:end_idx, self.wrench_topics].to_numpy(dtype='float')

            ##  get poses
            ## convert affine tfs to poses
            tfs_np = self.df.loc[start_idx:end_idx, self.pose_topics].to_numpy(dtype='float')
            poses_np = []
            for i in range(tfs_np.shape[0]):
                # print(tfs_np[i])
                poses_np.append(self.affine_tf_to_pose(tfs_np[i]))

            poses_np = np.array(poses_np)

            ##  get actions
            actions_np = self.df.loc[start_idx:end_idx, self.action_topics].to_numpy(dtype='float')

            ## get timestamps 
            times_np = self.df.loc[start_idx:end_idx, 'Time'].to_numpy(dtype='float')
        
        ## return normalized timestamps for each sample in the input sequence
        # normed_times_np = (times_np -  self.bagreader.start_time)/ self.total_T

        unnormed_times_np = times_np 

        ## massage into tensor
        ## (time window) x (batch) x (pose + wrench) 
        ## batches ommitted at this stage so:
        ## (time window) x (pose + wrench) 
        poses_wrenches_actions_tensor = torch.tensor(np.concatenate((poses_np, wrenches_np, actions_np), axis=1))

        return_dict = {'poses_wrenches_actions_tensor': poses_wrenches_actions_tensor,
        'target': self.target,
        'unnormed_times_np': unnormed_times_np, 
        'start_time': self.bagreader.start_time,
        'total_T': self.total_T,
        'bag_path': self.bag_path,
        'len_samples': self._len,
        'idx_accessed': idx
        }

        # return poses_wrenches_actions_tensor, self.target, normalized_times_np,  self.total_T
        return return_dict

    def affine_tf_to_pose(self, tf):
        
        tf_np = np.reshape(tf, (4,4), order='F')
        # pose_np[0:4, 3]
        rot = tf_np[0:3, 0:3]
        # R @ R.T
        rot = R.from_matrix(rot)
        quat = rot.as_quat()
        quat = np.divide(quat, np.linalg.norm(quat))

        trans = tf_np[0:3, -1]
        pose = np.concatenate((trans, quat))
        return pose
