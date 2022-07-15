import os
import torch
from torch.utils.data import Dataset

import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

# import bagpy
# from bagpy import bagreader
import rosbag
# import rosbag_pandas.src.rosbag_pandas
import yaml

import skimage
# import PIL
import torchvision
import cv2

import glob
import natsort

import math

class ContactDataset(Dataset):
    """

    """
    # if window_size = -1, produce the full episode !
    def __init__(self, bag_path, window_size, obj_name, in_cam_frame=True, im_resize=None, centered=False, load_in_mem=True):
        # self.rosbag = None
        self.load_in_mem = load_in_mem

        ## for now hardcode the main topic and topics of interest...
        
        # bag_path = '~/datasets/rosbags/input.bag'
        self.bag_path = os.path.expanduser(bag_path)
        if not os.path.exists(self.bag_path):
            raise AssertionError('bag path does not exist')

        try:
            self.bag = rosbag.Bag(bag_path, "r")
        except:
            print('reading bag ' + bag_path + ' failed!')
            raise 
        
        self.info_dict = yaml.safe_load(self.bag._get_yaml_info())
        self.info_dict['topics']

        for topic_dict in self.info_dict['topics']:
            if 'contact_data' in topic_dict['topic']:
                self.contact_freq = topic_dict['frequency']
                self.contact_num_msgs = topic_dict['messages']
            elif 'franka_states' in topic_dict['topic']:
                # self.contact_freq = topic_dict['frequency']
                self.proprio_num_msgs = topic_dict['messages']
            if 'depth' in topic_dict['topic']:
                # self.contact_freq = topic_dict['frequency']
                self.depth_num_msgs = topic_dict['messages']
        
        self.contact_dt = 1./self.contact_freq

        self.main_topic = '/panda/franka_state_controller_custom/franka_states'

        self.total_T = self.info_dict['duration']

        self.im_type = 'depth'
        self.im_path = os.path.join(self.bag_path.strip('.bag'), self.im_type)
        assert os.path.exists(self.im_path), 'im_path does not exist!!'
        self.im_times = np.load(os.path.join(self.im_path, 'timestamps.npy'), allow_pickle=True)
        self.im_path_list = natsort.natsorted(glob.glob(os.path.join(self.im_path, '*.png')))
        self.im_shape = cv2.imread(self.im_path_list[0]).shape[:2]
        self.in_cam_frame = in_cam_frame

        self.depth_tf_world = np.load(os.path.join(self.im_path, 'D_tf_W.npy'))
        self.K_cam = np.load(os.path.join(self.im_path, 'depth_K.npy'))
    
        assert self.depth_num_msgs == len(self.im_times), 'bag depth num msgs does not match depth timestamp length!'
        self.main_num_msgs = self.depth_num_msgs

        ## WINDOW SIZE IS THE SAME FOR ALL STREAMS FOR NOW
        # max_window = max(list(time_window_dict.values()))
        self.window_size = window_size

        self.obj_name = obj_name

        assert self.main_num_msgs > 1, "must be at least one number of msgs!" 
        assert self.main_num_msgs >= self.window_size, "number of msgs must be geq window size!" 

        self.centered = centered 

        if self.window_size == -1:
            self._len = 1
        else:
            self._len = self.main_num_msgs - window_size + 1

        self.base_proprio_topic = '/panda/franka_state_controller_custom/franka_states/'

        self.pose_topics = []
        for i in range(16):
            topic = self.base_proprio_topic + 'O_T_EE/' + str(i)
            self.pose_topics.append(topic)
        
        # 'EE_T_K'

        # to add EE vel??
        self.EE_vel_topics = []
        for i in range(6):
            topic = self.base_proprio_topic + 'O_dP_EE/' + str(i)
            self.EE_vel_topics.append(topic)
        # O_dP_EE # EE vel computed as J*dq

        self.wrench_topics = []
        for i in range(6):
            topic = self.base_proprio_topic + 'K_F_ext_hat_K/' + str(i)
            self.wrench_topics.append(topic)


        self.im_resize = im_resize #HxW
        self.im_to_tensor = torchvision.transforms.ToTensor() 

        if load_in_mem:
            ## read the label
            self.contact_df = pd.read_pickle(os.path.join(bag_path.strip('.bag'), 'contact_df.pkl'))
            self.contact_filtered_df = self.contact_df.loc[(self.contact_df['/contact_data_throttled/collision1_name'].str.contains(self.obj_name))]

            ## get features
            ## images
            
            self.depth_times = np.load(os.path.join(self.im_path, 'timestamps.npy'), allow_pickle=True)

            ## proprio
            self.proprio_df = pd.read_pickle(os.path.join(bag_path.strip('.bag'), 'proprio_df.pkl'))

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
            end_idx = self.main_num_msgs
        else:
            start_idx = idx
            end_idx = idx + self.window_size
        
        # get images
        images = skimage.io.imread_collection(self.im_path_list[start_idx:end_idx])
        #resize 
        if not self.im_resize:
            im_tensor = torch.stack([self.im_to_tensor(img) for img in images])
        else:
            im_tensor = torch.stack([self.im_to_tensor(cv2.resize(img, self.im_resize, interpolation=cv2.INTER_CUBIC)) for img in images])
        im_times = self.im_times[start_idx:end_idx]


        # get nearest proprio values
        ##  get poses
        ## convert affine tfs to poses
        # print(im_times)
        # print(self.proprio_df.index)
        nrst_proprio_idxs = self.get_nearest_idxs(im_times, self.proprio_df.index)
        nrst_tfs_np = np.array(self.proprio_df.iloc[nrst_proprio_idxs][self.pose_topics].values)
        nrst_poses_np = []
        for i in range(nrst_tfs_np.shape[0]):
            # print(tfs_np[i])
            pose = self.affine_tf_to_pose(nrst_tfs_np[i])
            if self.in_cam_frame:
                pose = self.transform_pose(pose, self.depth_tf_world)
            nrst_poses_np.append(pose)
        nrst_poses_np = np.array(nrst_poses_np)

        ## get wrenches
        nrst_wrench_np = np.array(self.proprio_df.iloc[nrst_proprio_idxs][self.wrench_topics].values)
        if self.in_cam_frame:
            tfed_nrst_wrench_np = []
            for i in range(nrst_wrench_np.shape[0]):
                # print(tfs_np[i])
                wrench = nrst_wrench_np[i]
                R = self.depth_tf_world[:3, :3]
                wrench = np.concatenate((R@wrench[:3], R@wrench[3:]))
                tfed_nrst_wrench_np.append(wrench)
            nrst_wrench_np = np.array(nrst_wrench_np)
        
        # get target contact label
        contact_idx = self.get_nearest_contact_idx(im_times, self.contact_filtered_df.index, centered=False)
        contact_list, contact_times, contact_time_diffs = self.get_contacts(contact_idx, self.contact_dt, self.contact_filtered_df)
        # (T dim x num contacts) of dicts
        # if only one time, contact_list is actually only list of num_contacts, not 2D

        # output the contact location prob map
        # output the contact forces map

        if not self.im_resize:
            contact_prob_map = np.zeros(self.im_shape)
            contact_force_map = np.zeros((3,) + self.im_shape)
            contact_normal_map = np.zeros((3,) + self.im_shape)
        else: 
            contact_prob_map = np.zeros(self.im_resize)
            contact_force_map = np.zeros((3,) + self.im_resize) #fx,fy,fz
            contact_normal_map = np.zeros((3,) + self.im_resize) #fx,fy,fz
        
        for contact in contact_list:
            if contact: # if not a null contact
                contact_pos = contact['position']
                contact_force = contact['force']
                contact_normal = contact['normal']

                contact_pos_prj = self.point_proj(self.K_cam, self.depth_tf_world, contact_pos)
                if not self.im_resize:
                    contact_prob_map[contact_pos_prj[1], contact_pos_prj[0]] = 1.0
                    contact_force_map[:, contact_pos_prj[1], contact_pos_prj[0]] = contact_force
                    contact_normal_map[:, contact_pos_prj[1], contact_pos_prj[0]] = contact_normal
                else:
                    contact_pos_prj_resized = ((self.im_resize[0]/self.im_shape[0])*contact_pos_prj).astype(int)
                    contact_prob_map[contact_pos_prj_resized[1], contact_pos_prj_resized[0]] = 1.0

                    contact_force_map[:, contact_pos_prj_resized[1], contact_pos_prj_resized[0]] = contact_force
                    contact_normal_map[:, contact_pos_prj_resized[1], contact_pos_prj_resized[0]] = contact_normal

################## bookmark

        ## return normalized timestamps for each sample in the input sequence
        # normed_times_np = (times_np -  self.breader.start_time)/ self.total_T

        return_dict = {
        'poses_np': nrst_poses_np,
        'wrench_np': nrst_wrench_np,
        'images_tensor': im_tensor,
        'im_times': im_times,
        'prob_map_np': contact_prob_map,
        'force_map_np': contact_force_map,
        'normal_map_np': contact_normal_map,
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

    def invert_transform(self, tf):
        R = tf[0:3, 0:3]
        T = tf[:3, -1]
        tf_inv = np.diag([0.,0.,0.,1.])
        tf_inv[:3, :3] = R.T
        tf_inv[:3, -1] = -R.T @ T
        return tf_inv
    
    def transform_pose(self, pose_W, C_tf_W):
        C_rot_W = R.from_matrix(C_tf_W[:3, :3])

        pos = pose_W[:3]
        W_rot_EE = R.from_quat(pose_W[3:])
        pos_tfed = (C_tf_W @ np.concatenate((pos, np.array([1.,]))))[:-1]
        
        ori_tfed = C_rot_W * W_rot_EE

        return np.concatenate((pos_tfed, ori_tfed.as_quat())) 

    # TODO return times list
    def get_nearest_idxs(self, times, df_index):
        idxs = df_index.searchsorted(times, side="left")
        idxs_list = []
        for i in range(len(times)):
            if idxs[i] > 0 and (idxs[i] == len(df_index) or math.fabs(times[i] - df_index[idxs[i]-1]) < math.fabs(times[i] - df_index[idxs[i]-1])):
                idxs_list.append(idxs[i]-1)
            else:
                idxs_list.append(idxs[i])
        return idxs_list
    
    def get_nearest_contact_idx(self, times, df_index, centered=False):
        if not centered:
            return self.get_nearest_idxs([times[-1]], df_index)
        else:
            return self.get_nearest_idxs([times[len([times])//2]], df_index)
    
    # TODO change this to only take one time at a time...
    def get_contacts(self, times, contact_dt, contact_df):
        contact_list = []
        contact_times = []
        contact_time_diff = []
        for time in times:
            contacts = [] # list of all contact dicts at each timestep
            contact_idx = self.get_nearest_idxs([time], contact_df.index)[0]
            row = contact_df.iloc[contact_idx].loc[contact_df.iloc[contact_idx].notnull()]
            num_contacts = len(row[row.keys().str.contains('depths')])
            time_diff = abs(contact_df.index[contact_idx] - time)
            contact_time_diff.append(time_diff)
            if time_diff > (contact_dt*1.5):
                contact_list.append(None)
                contact_times.append(None)
            else:
                for contact_id in range(num_contacts):
                    contact_dict = {}

                    contact_pos_idx = '/contact_data_throttled/contact_positions/' + str(contact_id) 
                    contact_pos_cols = [col for col in row.keys() if contact_pos_idx in col]
                    contact_pos = np.array(row[contact_pos_cols].values)
                    
                    contact_dict['position'] = contact_pos

                    contact_nrml_idx = '/contact_data_throttled/contact_normals/' + str(contact_id) 
                    contact_nrml_cols = [col for col in row.keys() if contact_nrml_idx in col]
                    contact_nrml = np.array(row[contact_nrml_cols].values)

                    contact_dict['normal'] = contact_nrml

                    contact_force_idx = '/contact_data_throttled/wrenches/' + str(contact_id) + '/force/'
                    contact_force_cols = [col for col in row.keys() if contact_force_idx in col]
                    contact_force = np.array(row[contact_force_cols].values)

                    contact_dict['force'] = contact_force

                    contact_torque_idx = '/contact_data_throttled/wrenches/' + str(contact_id) + '/torque/'
                    contact_torque_cols = [col for col in row.keys() if contact_torque_idx in col]
                    contact_torque = np.array(row[contact_torque_cols].values)

                    contact_dict['torque'] = contact_torque
                
                    contacts.append(contact_dict)
                
                contact_list.append(contacts)
                contact_times.append(contact_df.index[contact_idx])

        return contact_list, contact_times, contact_time_diff

    def point_proj(self, K, C_tf_W, pos):
        contact_pos_in_depth = (C_tf_W @ np.concatenate((pos, np.array([1]))))[:-1]
        # print(contact_pos_in_depth)
        project_coords = K @ (contact_pos_in_depth)
        return (project_coords[:2]/project_coords[-1]).astype(int)  


