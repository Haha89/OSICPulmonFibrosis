"""Script to train """


import numpy as np
import torch
import torch.optim as optim
from torch.utils import data
from ODE_network import ODE_Network
from utils import ode_laplace_log_likelihood
from dataset import Dataset
from pickle import load
from os import remove
from glob import glob

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
PATH_DATA = '../data/'
LEARNING_RATE = 0.0001
NUM_EPOCHS = 50


if __name__ == "__main__":

    unscale = lambda x: x*(MAXI_FVC-MINI_FVC) + MINI_FVC
    
    for f in glob(f"{PATH_DATA}/histo-fold/histo-fold-*.pt"): #Removes existing histo-fold-X.pt
        remove(f)
        
    histo = torch.zeros((NUM_EPOCHS, 2))
    torch.cuda.empty_cache()

    model = ODE_Network(1, 10, (256, 256, 32), 16, 32, 3, 64)
    model.to(DEVICE)
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min')
    
    #####################
    # Data loading
    #####################
    training_set = Dataset(np.arange(176))
    training_generator = data.DataLoader(training_set, batch_size=1, shuffle=True)

    
    with open(f'{PATH_DATA}model/minmax.pickle', 'rb') as minmax_file:
        dict_extremum = load(minmax_file)
    
    MINI_FVC = dict_extremum['FVC']["min"]
    MAXI_FVC = dict_extremum['FVC']["max"]

    for epoch in range(NUM_EPOCHS):

        loss_train = 0
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
            loss = ode_laplace_log_likelihood(goal, mean, std, epoch, 25)
            loss_train += loss
            loss.backward() # Gradient Computation
            optimiser.step() # Update parameters

        loss_train = loss_train/len(training_generator)
        print(f'| Epoch: {epoch+1} | Train Loss: {loss_train:.3f} |')
        histo[epoch, 0] = loss_train

        
    DATA_SAVE = {'weeks': weeks, 'fvc': fvc, 'misc': misc, 'goal': goal, 'mean': mean, 'std': std}
    torch.save(DATA_SAVE, f"{PATH_DATA}/saved_data/data_save.pt")        
    torch.save(histo, f"{PATH_DATA}/histo-fold/histo.pt")
    
    CHECKPOINT = {'model': model,
              'state_dict': model.state_dict(),
              'optimiser' : optimiser.state_dict()}

    torch.save(CHECKPOINT['model'], '../data/model/model_full_train_v2.pth')
    torch.save(CHECKPOINT['state_dict'], '../data/model/state_full_train_v2.pth')