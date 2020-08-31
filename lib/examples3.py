# -*- coding: utf-8 -*-

"""Visualization of train and test loss"""
import torch
import matplotlib.pyplot as plt
from os import listdir

PATH_DATA = '../data/'
PATH_HISTO = PATH_DATA + "/histo-fold/"

files = listdir(PATH_HISTO)

if len(files)>1:
    fig, axs = plt.subplots(len(files), 1, constrained_layout=True)
    for i, file in enumerate(files):
        with open(PATH_HISTO + file, "rb") as data:
            histo = torch.load(data)
        axs[i].plot(histo[:, 0].detach(), c='r', label="train")
        axs[i].plot(histo[:, 1].detach(), c='b', label="test")
        
        if i > 0:
            axs[i].set_title(f'Fold {i}')
            plt.setp(axs[i].get_xticklabels(), visible=False)
else:
    with open(PATH_HISTO + files[0], "rb") as data:
        histo = torch.load(data)
    fig, axs = plt.subplots()   
    plt.plot(histo[:, 0].detach(), c='r', label="train")
    plt.plot(histo[:, 1].detach(), c='b', label="test")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.show()
