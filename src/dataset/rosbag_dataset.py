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
class RosbagDataset(Dataset):
    """A Dataset which wraps a RosBag
    Note:
        The dataset can only read data sequentially, and will raise an error when two calls are not consecutives
    Args:
        file_path (str): The path on disk to the rosbag
        main_topic (str): The name of the main topic (which sets the number of frames to be extracted)
        frame_size (int): The number of messages to accumulate in a frame
        topic_mapping (dict): The mapping topic name to key in the data_dict
    """

    def __init__(self, bag_path, window_size):
        # self.rosbag = None

        ## for now hardcode the main topic and topics of interest...
        
        # bag_path = '~/datasets/rosbags/input.bag'
        self.bag_path = os.path.expanduser(bag_path)
        if not os.path.exists(self.bag_path):
            raise AssertionError('bag path does not exist')

        bag = rosbag.Bag(bag_path)
        bag_yaml_str = bag._get_yaml_info()
        self.bag_yaml = yaml.safe_load(bag_yaml_str)

        self.main_topic = 'panda/franka_state_controller_custom/franka_states'

        self.main_num_msgs = bag.get_message_count(self.main_topic)
        # main_topic_window = 100 # number of messages
        
        ## WINDOW SIZE IS THE SAME FOR ALL STREAMS FOR NOW
        # max_window = max(list(time_window_dict.values()))
        self.window_size = window_size
        self._len = (self.main_num_msgs // window_size)

        self.bagreader = bagreader(bag_path)

        self.pose_topics = []
        for i in range(16):
            topic = 'O_T_EE_' + str(i)
            self.pose_topics.append(topic)

        
        self.wrench_topics = []
        for i in range(6):
            topic = 'K_F_ext_hat_K_' + str(i)
            self.wrench_topics.append(topic)

        # table = b.topic_table
        # self.topic_list = []
        # self.topic_csv_list = []
        self.topic_csv_dict = {}
        for topic in self.bagreader.topics:
            # print(topic)
            # self.topic_list.append(topic)
            self.topic_csv_dict[topic] = self.bagreader.message_by_topic(topic)
        
        ## Read in the target file
        # self.target_paths = 

        # EE pose: O_T_EE_1 ... O_T_EE_15
        # self.time_window_dict = {
        #     'O_T_EE' : 
        # }
        
        # self.label = 

        bag.close()

    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        ## get features
        main_topic_csv = self.topic_csv_dict[self.main_topic] 
        df = pd.read_csv(main_topic_csv)

        # self.main_num_msgs
        if idx >= self._len:
            raise AssertionError('index out of range')

        ## indexing
        start_idx = self.window_size * idx
        end_idx = self.window_size*(idx+1)  - 1

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

        ## massage into tensor
        ## (time window) x (batch) x (pose + wrench) 
        ## batches ommitted at this stage so:
        ## (time window) x (pose + wrench) 
        poses_wrenches_tensor = torch.tensor(np.concatenate((poses_np, wrenches_np), axis=1))

        ## get target
        ## TODO

        return poses_wrenches_tensor

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


