## generate csv files from all rosbags in a run directory
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

import glob

base_dset_dir = os.path.expanduser('~/datasets/screwing')
xprmnt_dir = time.strftime("/%Y-%m-%d_%H-%M-%S")

self.bag_path = os.path.expanduser(bag_path)
if not os.path.exists(self.bag_path):
    raise AssertionError('bag path does not exist')

bag = rosbag.Bag(bag_path)
bag_yaml_str = bag._get_yaml_info()
self.bag_yaml = yaml.safe_load(bag_yaml_str)

self.main_topic = '/panda/franka_state_controller_custom/franka_states'

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

## Read in the target file
# self.target_paths = 

# EE pose: O_T_EE_1 ... O_T_EE_15
# self.time_window_dict = {
#     'O_T_EE' : 
# }

# self.label = 

bag.close()