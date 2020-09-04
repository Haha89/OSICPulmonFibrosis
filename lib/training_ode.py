"""Script to train """

import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils import data
from ODE_network import ODE_Network
import tools
from dataset import Dataset
from pickle import load
from os import remove
from glob import glob

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")


"""
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICE"] = "0"
"""

PATH_DATA = '../data/'
NB_FOLDS = 3
LEARNING_RATE = 0.0001
NUM_EPOCHS = 13


if __name__ == "__main__":

    unscale = lambda x: x*(MAXI_FVC-MINI_FVC) + MINI_FVC
    
    for f in glob(f"{PATH_DATA}/histo-fold/histo-fold-*.pt"): #Removes existing histo-fold-X.pt
        remove(f)

    with open('minmax.pickle', 'rb') as minmax_file:
        dict_extremum = load(minmax_file)
        
    MINI_FVC = dict_extremum['FVC']["min"]
    MAXI_FVC = dict_extremum['FVC']["max"]
    MEAN_week = dict_extremum['Weeks']["mean"]
    STD_week = dict_extremum['Weeks']["std"]
    FOLD_LABELS = np.load("4-folds-split.npy")
    
    for k in range(NB_FOLDS):
        indices_train, indices_test = tools.train_test_indices(FOLD_LABELS, k)
        model = ODE_Network(1, 10, (256, 256, 32), 16, 64, 3, 64)
        model.to(DEVICE)
        optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-8)
    
        #####################
        # Loding of data
        #####################
    
        training_set = Dataset(PATH_DATA, indices_train)
        training_generator = data.DataLoader(training_set, batch_size=1, shuffle=True)
    
        testing_set = Dataset(PATH_DATA, indices_test)
        testing_generator = data.DataLoader(testing_set, batch_size=1, shuffle=False)
        
        histo = torch.zeros((NUM_EPOCHS, 2))
    
        for epoch in range(NUM_EPOCHS):
            start_time = time.time()
            loss_train, loss_test = 0, 0
            model.train()

            #TRAINING    
            for scans, misc, FVC, percent, weeks in training_generator:
                ranger = np.where(weeks != 0)[1]
                misc = misc[:,ranger[0],:].squeeze(1) #Dépend du m
                fvc = FVC[:,ranger[0]]
                percent = percent[:,ranger[0]]
                weeks = weeks[:,ranger]
                weeks = weeks - weeks[:,0]
                scans, misc = scans.to(DEVICE), misc.to(DEVICE)
                fvc, percent, weeks = fvc.to(DEVICE), percent.to(DEVICE), weeks.to(DEVICE)
                # Clear stored gradient
                optimiser.zero_grad()
                
                
                pred = model(scans, misc, fvc, percent,weeks)
                #Deprocessing
                mean = unscale(pred[:, :-1, 0])
                std = pred[:, :-1, 1]*100
                goal = FVC[:,ranger[1:]]
                goal = unscale(goal).to(DEVICE)
                
                # print("Weeks")
                # print(weeks)
                # print("FVC")
                # print(fvc)
                # print("MISC")
                # print(misc)
                # print("Goal")
                # print(goal)
                # print("MEAN"
                # print(mean)
                # print("STD")
                # print(std)
                # print("+++++++++++++++")
                
                loss = tools.ode_laplace_log_likelihood(goal, mean, std)
                loss_train += loss
                loss.backward() # Gradient Computation
                optimiser.step() # Update parameters

            #VALIDATION
            with torch.no_grad():
                model.eval()
                for scans, misc, FVC, percent, weeks in testing_generator:
                    ranger = np.where(weeks != 0)[1]
                    misc = misc[:,ranger[0],:].squeeze(1) #Dépend du m
                    fvc = FVC[:,ranger[0]]
                    percent = percent[:,ranger[0]]
                    weeks = weeks[:,ranger]
                    weeks = weeks - weeks[:,0]
                    scans, misc = scans.to(DEVICE), misc.to(DEVICE)
                    fvc, percent, weeks = fvc.to(DEVICE), percent.to(DEVICE), weeks.to(DEVICE)
                    pred = model(scans, misc, fvc, percent,weeks)
                    
                    #Deprocessing
                    mean = unscale(pred[:, :-1, 0])
                    std = pred[:, :-1, 1]*100    
                    goal = FVC[:,ranger[1:]]
                    goal = unscale(goal).to(DEVICE)

                    loss = tools.ode_laplace_log_likelihood(goal, mean, std)
                    loss_test += loss
                    
            loss_train = loss_train/len(training_generator)
            loss_test = loss_test/len(testing_generator)
            print(f'| Epoch: {epoch+1} | Train Loss: {loss_train:.3f} | Test. Loss: {loss_test:.3f} |')
            histo[epoch, 0] = loss_train
            histo[epoch, 1] = loss_test
        torch.save(histo, f"{PATH_DATA}/histo-fold/histo-fold-{k}.pt")
        
        CHECKPOINT = {'model': model,
                  'state_dict': model.state_dict(),
                  'optimiser' : optimiser.state_dict()}
        torch.save(CHECKPOINT, f'../data/model/model-{k}.pth')
