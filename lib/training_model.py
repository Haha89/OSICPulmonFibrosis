# -*- coding: utf-8 -*-

"""Script to train """

import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils import data
from conv3dNetwork import Convolutionnal_Network
import tools
from dataset import Dataset

DEVICE = ("0" if torch.cuda.is_available() else "cpu")
"""
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICE"] = "0"
"""

PATH_DATA = '../data/'
NB_FOLDS = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 100

(MINI, MAXI) = np.load("minmax.npy")
FOLD_LABELS = np.load("4-folds-split.npy")

for k in range(NB_FOLDS):
    indices_train, indices_test = tools.train_test_indices(FOLD_LABELS, k)
    model = Convolutionnal_Network(1, 10, (128, 128, 128), 16, 64, 4, 64)
    model.to(DEVICE)
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-5)

    #####################
    # Train model
    #####################

    training_set = Dataset(PATH_DATA, indices_train)
    training_generator = data.DataLoader(training_set, batch_size=4, shuffle=True)

    testing_set = Dataset(PATH_DATA, indices_test)
    testing_generator = data.DataLoader(testing_set, batch_size=1, shuffle=False)

    histo = torch.zeros((NUM_EPOCHS, 2))

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        loss_train, loss_test = 0, 0
        model.train()

        for scans, misc, FVC, percent in training_generator:

            scans, misc = scans.to(DEVICE), misc.to(DEVICE)
            FVC, percent = FVC.to(DEVICE), percent.to(DEVICE)

            # Clear stored gradient
            optimiser.zero_grad()
            pred = model(scans, misc, FVC, percent)

            #Deprocessing
            mean = pred[:, :-1, 0]*(MAXI-MINI) + MINI
            std = pred[:, :-1, 1]*(MAXI-MINI) + MINI
            goal = FVC[1:]*(MAXI-MINI) + MINI

            loss = tools.laplace_log_likelihood(goal, mean, std)
            loss_train += loss

            loss.backward() # Gradient Computation
            optimiser.step() # Update parameters

        #Validation
        with torch.no_grad():
            model.eval()
            for scans, misc, FVC, percent in testing_generator:
                scans, misc = scans.to(DEVICE), misc.to(DEVICE)
                FVC, percent = FVC.to(DEVICE), percent.to(DEVICE)
                pred = model(scans, misc, FVC, percent)

                mean = pred[:, :-1, 0]*(MAXI-MINI) + MINI
                std = pred[:, :-1, 1]*(MAXI-MINI) + MINI
                goal = FVC[1:]*(MAXI-MINI) + MINI

                loss = tools.laplace_log_likelihood(goal, mean, std)
                loss_test += loss

        loss_train = loss_train/len(training_generator)
        loss_test = loss_test/len(testing_generator)
        print(f'| Epoch: {epoch+1} | Train Loss: {loss_train:.3f} | Test. Loss: {loss_test:.3f} |')
        histo[epoch, 0] = loss_train
        histo[epoch, 1] = loss_test
    torch.save(histo, f"../data/histo-fold/histo-fold-{k}.pt")
