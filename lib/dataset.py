# -*- coding: utf-8 -*-

"""Definition of the class Dataset and some function for rossvalidation purposes"""

from os import listdir
import torch
import numpy as np
from torch.utils import data
from tools import get_data, filter_data, get_3d_scan

PATH_DATA = "../data/"

class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, path, indices):
        'Initialization'
        self.indices = indices
        self.list_of_ids = np.array(listdir(path + 'train/'))[self.indices]
        self.data = get_data() #Train CSV File

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indices)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # if self.list_of_ids[index] == 'ID00052637202186188008618' or self.list_of_ids[index] == "ID00105637202208831864134":
        #     index = (index + 14)%len(self.indices)
        scan = get_3d_scan(self.list_of_ids[index])
        misc, fvc, percent,weeks = filter_data(self.data, self.list_of_ids[index])
        scan = torch.tensor(scan).unsqueeze(0)
        return (scan.float(), misc.float(), fvc.float(), percent.float(), weeks.float())
