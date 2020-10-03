"""Script to train """

import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils import data
from ODE_network import ODE_Network
from utils import train_test_indices, total_loss
from dataset import Dataset
from pickle import load
from os import remove
from glob import glob

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
PATH_DATA = '../data/'
NB_FOLDS = 1
LEARNING_RATE = 0.0001
NUM_EPOCHS = 40


if __name__ == "__main__":

    unscale = lambda x: x*(MAXI_FVC-MINI_FVC) + MINI_FVC
    
    for f in glob(f"{PATH_DATA}/histo-fold/histo-fold-*.pt"): #Removes existing histo-fold-X.pt
        remove(f)
        
    FOLD_LABELS = np.load("./4-folds-split.npy")
    best_test_loss = 1000
    
    for k in range(NB_FOLDS):
        print(f"Starting Fold {k}")
        histo = torch.zeros((NUM_EPOCHS, 2))
        torch.cuda.empty_cache()
        indices_train, indices_test = train_test_indices(FOLD_LABELS, k)

        model = ODE_Network(1, 10, (256, 256, 32), 16, 32, 3, 64)
        model.to(DEVICE)
        optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-8)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min')
        
        #####################
        # Data loading
        #####################
    
        training_set = Dataset(indices_train)
        training_generator = data.DataLoader(training_set, batch_size=1, shuffle=True)

        testing_set = Dataset(indices_test)
        testing_generator = data.DataLoader(testing_set, batch_size=1, shuffle=False)
        
        with open(f'{PATH_DATA}model/minmax.pickle', 'rb') as minmax_file:
            dict_extremum = load(minmax_file)
        
        MINI_FVC = dict_extremum['FVC']["min"]
        MAXI_FVC = dict_extremum['FVC']["max"]
    
        for epoch in range(NUM_EPOCHS):
            start_time = time.time()
            loss_train, loss_test = 0, 0
            model.train()

            #TRAINING    
            for scans, misc, FVC, percent, weeks, ranger in training_generator:

                ranger = np.where(ranger != 0)[1]
                misc = misc[:,ranger[0],:].squeeze(1) #Dépend du m
                fvc = FVC[:,ranger[0]]
                std = torch.ones_like(fvc)*.7
                fvc = torch.cat((fvc, std),1)
                percent = percent[:,ranger[0]]
                weeks = weeks[:,ranger]
                weeks = weeks - weeks[:,0]
                scans, misc = scans.to(DEVICE), misc.to(DEVICE)
                fvc, percent, weeks = fvc.to(DEVICE), percent.to(DEVICE), weeks.to(DEVICE)
                
                # Clear stored gradient
                optimiser.zero_grad()
                pred = model(scans, misc, fvc, percent, weeks)
                
                #Deprocessing
                mean = unscale(pred[:, :, 0])
                std = pred[:, :, 1]*500 
                goal = FVC[:,ranger]
                goal = unscale(goal).to(DEVICE)
                mean = mean  + (goal[:,0]- mean[:,0])
                loss = total_loss(goal, mean, std, epoch, -1)
                loss_train += loss
                loss.backward() # Gradient Computation
                optimiser.step() # Update parameters

            #VALIDATION
            with torch.no_grad():
                model.eval()
                for scans, misc, FVC, percent, weeks, ranger in testing_generator:
                    ranger = np.where(ranger != 0)[1]
                    misc = misc[:,ranger[0],:].squeeze(1) #Dépend du m
                    fvc = FVC[:,ranger[0]]
                    std = torch.ones_like(fvc)*.7
                    fvc = torch.cat((fvc, std),1)
                    percent = percent[:,ranger[0]]
                    weeks = weeks[:,ranger]
                    weeks = weeks - weeks[:,0]
                    scans, misc = scans.to(DEVICE), misc.to(DEVICE)
                    fvc, percent, weeks = fvc.to(DEVICE), percent.to(DEVICE), weeks.to(DEVICE)
                    pred = model(scans, misc, fvc, percent, weeks)
                    
                    #Deprocessing
                    mean = unscale(pred[:, :, 0])
                    std = pred[:, :, 1]*500    
                    goal = FVC[:,ranger]
                    goal = unscale(goal).to(DEVICE)
                    mean = mean  + (goal[:,0]- mean[:,0])
                    loss = total_loss(goal, mean, std, epoch, -1)
                    loss_test += loss
                    
                    
            loss_train = loss_train/len(training_generator)
            loss_test = loss_test/len(testing_generator)
            print(f'| Epoch: {epoch+1} | Train Loss: {loss_train:.3f} | Test. Loss: {loss_test:.3f} |')
            histo[epoch, 0] = loss_train
            histo[epoch, 1] = loss_test
            scheduler.step(loss_test)
            
            CHECKPOINT = {'model': model,
                  'state_dict': model.state_dict(),
                  'optimiser' : optimiser.state_dict()}
                    
            if loss_test < best_test_loss:
                best_test_loss = loss_test
                torch.save(CHECKPOINT['model'], '../data/model/model_6.pth')
                torch.save(CHECKPOINT['state_dict'], '../data/model/state_6.pth')
                DATA_SAVE = {'weeks': weeks, 'fvc': fvc, 'misc': misc, 'goal': goal, 'mean': mean, 'std': std}
                torch.save(DATA_SAVE, f"{PATH_DATA}/saved_data/data_save.pt")        
                
        torch.save(histo, f"{PATH_DATA}/histo-fold/histo-fold-{k}.pt")
        torch.save(CHECKPOINT['model'], '../data/model/model_final.pth')
        torch.save(CHECKPOINT['state_dict'], '../data/model/state_final.pth')