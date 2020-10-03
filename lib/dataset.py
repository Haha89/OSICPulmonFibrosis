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
        if self.train:
            scan = get_3d_scan(self.list_of_ids[index])
            misc, fvc, percent, weeks, ranger = filter_data(self.data, self.list_of_ids[index])
            scan = torch.tensor(scan).unsqueeze(0)
            return (scan.float(), misc.float(), fvc.float(), percent.float(), weeks.float(), ranger.int())

        else:
            try:
                scan = process_3d_scan(self.list_of_ids[index], False)
            except:
                print("Error caught in scan creation. Returning zeros")
                with np.load("../input/localosic/OSICPulmonFibrosis-master/data/scans/ID00421637202311550012437.npy") as scan_file:
                    scan = scan_file

            misc, fvc, percent, weeks = filter_data(self.data, self.list_of_ids[index], train=False)
            scan = torch.tensor(scan).unsqueeze(0)
            return (scan.float(), misc.float(), fvc.float(), percent.float(), weeks.float())
