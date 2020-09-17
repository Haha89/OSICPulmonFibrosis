# -*- coding: utf-8 -*-

"""Script to generate the submission"""

import numpy as np # linear algebra
import pandas as pd
import utils
import torch
from pickle import load
from dataset import Dataset
from torch.utils import data

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

test_df = utils.get_data(train=False)
sub = pd.read_csv('../data/sample_submission.csv')

checkpoint = torch.load('../data/model/model-0.pth')
model = checkpoint['model']
model = model.to(DEVICE)
model.load_state_dict(checkpoint['state_dict'])


with open('minmax.pickle', 'rb') as minmax_file:
    min_max = load(minmax_file)['FVC']
    
unscale = lambda x: x*(min_max["max"] - min_max["min"]) + min_max["min"]



#####################
# Loading of data
#####################
a = np.arange(0, len(test_df.Patient.unique()))
testing_set = Dataset(a, train=False)
testing_generator = data.DataLoader(testing_set, batch_size=1, shuffle=False)
print("Train Dataset OK")

for scans, misc, FVC, percent, weeks in testing_generator:
    
    scans, misc = scans.to(DEVICE), misc.to(DEVICE)
    FVC, percent, weeks = FVC.to(DEVICE), percent.to(DEVICE), weeks.to(DEVICE)
    pred = model(scans, misc, FVC, percent, weeks)
    mean = unscale(pred[:, :, 0])
    std = pred[:, :, 1]*500    
    print(mean)
    print(std)
    

    
# sub.rename(columns={"std":"FVC"})[["Patient_Week","FVC","Confidence"]].to_csv("../data/submission.csv", index=False)