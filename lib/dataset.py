# -*- coding: utf-8 -*-

"""Definition of the class Dataset and some function for rossvalidation purposes"""

import torch
from torch.utils import data
from utils import get_data, filter_data, get_3d_scan
from scan_processing import process_3d_scan
import numpy as np

class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, indices, train=True):
        'Initialization'
        self.indices = indices
        self.train = train
        self.data = get_data(train=self.train) #CSV File (Train or Test)
        self.list_of_ids = self.data.Patient.unique()[self.indices]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indices)

    def __getitem__(self, index):
        'Generates one sample of data'
        
        try:
            if self.train:
                scan = get_3d_scan(self.list_of_ids[index])
            else:
                scan = process_3d_scan(self.list_of_ids[index], False)
        except:
            print("Error caught in Dataset. Returning zeros")
            scan = np.zeros((32, 256, 256))   
        # scan = np.zeros((32, 256, 256))      
        misc, fvc, percent,weeks, ranger = filter_data(self.data, self.list_of_ids[index])
        scan = torch.tensor(scan).unsqueeze(0)
        return (scan.float(), misc.float(), fvc.float(), percent.float(), weeks.float(), ranger.int())
