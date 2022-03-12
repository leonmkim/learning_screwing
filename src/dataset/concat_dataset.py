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

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)