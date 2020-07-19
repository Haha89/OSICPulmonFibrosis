# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 21:16:31 2020

@author: Benjamin
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils import data
import torch.nn.functional as F
from Folds_dataset import *
import time
from conv3dNetwork import *
from metrique import laplace_log_likelihood

device = ("0" if torch.cuda.is_available() else "cpu" )

"""
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICE"] = "0"
"""

path= 'C:/Users/Benjamin/Desktop/Kaggle/osic-pulmonary-fibrosis-progression/'
#path= "E:/DataScience/Kaggle Challenges/osic-pulmonary-fibrosis-progression/"


extremums = np.load("minmax.npy")
mini = extremums[0]
maxi = extremums[1]
   
nb_folds = 1
fold_labels = np.load("4-folds-split.npy")

for k in range(nb_folds):  
    indices_train, indices_test = train_test_indices(fold_labels,k)
    learning_rate = 0.001
    model = Convolutionnal_Network(1,10,(128,128,128),16,64,4,64)   
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 5e-5)
    
    #####################
    # Train model
    #####################
    num_epochs = 100
    
    training_set = Dataset(path, indices_train)
    training_generator = data.DataLoader(training_set, batch_size = 4, shuffle = True,num_workers=0)
    
    testing_set = Dataset(path, indices_test)
    testing_generator = data.DataLoader(testing_set, batch_size = 1, shuffle = False,num_workers=0)

    histo = torch.zeros((num_epochs,2))

    for t in range(num_epochs):
        start_time = time.time()
        
        loss_training = 0
        loss_testing = 0
        
        model.train()
        for scans, misc, FVC, percent in training_generator: 
        # Clear stored gradient
            scans = scans.to(device)
            misc = misc.to(device)
            FVC = FVC.to(device)
            percent = percent.to(device)
            
            optimiser.zero_grad()
            pred = model(scans, misc, FVC, percent)
            
            mean = pred[:,:-1,0]*(maxi-mini) + mini
            std = pred[:,:-1,1]*(maxi-mini) + mini
            
            goal = FVC[1:]*(maxi-mini) + mini
            
            loss = laplace_log_likelihood(goal,mean,std)
            loss_training += loss
            # Backward pass
            loss.backward()
            # Update parameters
            optimiser.step()

        with torch.no_grad():
            model.eval()
            for scans, misc, FVC, percent in testing_generator: 
        # Clear stored gradient
                scans = scans.to(device)
                misc = misc.to(device)
                FVC = FVC.to(device)
                percent = percent.to(device)

                pred = model(scans, misc, FVC, percent)
                
                mean = pred[:,:-1,0]*(maxi-mini) + mini
                std = pred[:,:-1,1]*(maxi-mini) + mini
                
                goal = FVC[1:]*(maxi-mini) + mini
                
                loss = laplace_log_likelihood(goal,mean,std)
                loss_testing += loss
        loss_training = loss_training/len(training_generator)    
        loss_testing =  loss_testing/len(testing_generator)
        print(t,loss_training,loss_testing, time.strftime("%H:%M:%S",time.gmtime(time.time()-start_time)))
        histo[t,0] = loss_training
        histo[t,1] = loss_testing
