# -*- coding: utf-8 -*-

"""Script to generate the submission"""

import numpy as np # linear algebra
import pandas as pd
import utils
import torch
from pickle import load
from dataset import Dataset
from torch.utils import data


MIN_WEEK = -20
MAX_WEEK = 180
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

test_df = utils.get_data(train=False)
sub = pd.read_csv('../data/sample_submission.csv')
sub = sub.iloc[0:0]

checkpoint = torch.load('../data/model/model-0.pth')
model = checkpoint['model']
model = model.to(DEVICE)
model.load_state_dict(checkpoint['state_dict'])


with open('../data/model/minmax.pickle', 'rb') as minmax_file:
    min_max = load(minmax_file)['FVC']

unscale = lambda x: x*(min_max["max"] - min_max["min"]) + min_max["min"]


#####################
# Loading of data
#####################
a = np.arange(0, len(test_df.Patient.unique()))
testing_set = Dataset(a, train=False)
testing_generator = data.DataLoader(testing_set, batch_size=1, shuffle=False)

for i, (scans, misc, fvc, percent, week) in enumerate(testing_generator):
    print(f"{i+1}/{len(testing_generator)}")
    try:
        fvc_std = torch.cat((fvc, torch.ones_like(fvc)*.7),1)
        weeks = torch.torch.from_numpy(np.arange(MIN_WEEK, MAX_WEEK+1)).float()
        scans, misc = scans.to(DEVICE), misc[:,:,0].to(DEVICE)
        fvc_std, percent, weeks = fvc_std.to(DEVICE), percent.to(DEVICE), weeks.to(DEVICE)
        first_week = int(week.cpu().detach().numpy()[0])
        pred = model(scans, misc, fvc_std, percent, weeks-first_week)
        
        #Postprocessing
        mean = unscale(pred[:, :, 0])
        std = pred[:, :, 1]*500    
        goal = unscale(fvc).to(DEVICE)
        mean = mean + (goal[:,0]- mean[:,first_week-MIN_WEEK])
    
        mean = mean.cpu().detach().numpy()[0]
        std = std.cpu().detach().numpy()[0]
        list_weeks = [test_df.Patient.unique()[i] + "_" + str(int(week.cpu().detach().numpy())) for week in weeks]
        
    except:
        weeks = torch.torch.from_numpy(np.arange(MIN_WEEK, MAX_WEEK+1)).float()
        list_weeks = [test_df.Patient.unique()[i] + "_" + str(int(week.cpu().detach().numpy())) for week in weeks]
        mean = np.ones((len(list_weeks)))*2000
        std  = np.ones((len(list_weeks)))*250
        
    df = pd.DataFrame({"Patient_Week": list_weeks, "FVC":mean, "Confidence":std})
    sub = pd.concat([sub, df], axis=0)
    
sub.round(0).to_csv("../data/submission.csv", index=False)
print("Submission save")