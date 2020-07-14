# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:53:53 2020

@author: Benjamin
"""

import numpy as np
import os 
import random
import time
from os import scandir
from tools import *

path = '..'

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, path, indices):
        'Initialization'
        self.indices = indices
        self.list_of_ids = np.array(os.listdir(path))[self.indices]
        
        
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.indices)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        scan = get_3d_scan(self.list_of_ids[index], True)
        data, FVC = ###Waiting for loading function

        return (scan,data,FVC)


#Fonction qui permet de labeliser nos entrÃ©es de 0 Ã  nb_folds pour k-fold cross validation  
def make_folds(path, nb_folds):
    batch_size = len(os.listdir(path + 'train/'))
    subfolders = [i for i in range(batch_size)]
      #Chaque donnÃ©e Ã  un label
    fold_label = np.zeros(batch_size, dtype=int)    
    items_per_fold = batch_size//nb_folds
    for fold in range(nb_folds-1):
        sample = random.sample(subfolders,items_per_fold)
        #On les assigne au label considÃ©rÃ©     
        for smpl in sample :
            fold_label[smpl] = fold + 1
            #On retire les indices tirÃ©s de la liste
        subfolders = [index for index in subfolders if fold_label[index] == 0]
    #Comme la division nombre d'Ã©lements/nb_folds ne tombe pas forcÃ©ment juste, on assigne tous les derniers Ã©lements au label restant
    for smpl in subfolders :
        fold_label[smpl] = nb_folds
    # Pour une input de taille batch x taille seq max x nb_features, fold_label est de taille batch
    # Et fold_label[k] contient le label appartenant Ã  [0, nb_fold-A] correspondant Ã  input[k,:,:]
    return(fold_label-1)

#Pour un fold donnÃ©, crÃ©e le testing set et training set en fonction de fold_label  
def train_test_valid_split(fold_label, nb_fold):
    
    indices_train = np.where(fold_label != nb_fold)[0]
    test_and_val = np.where(fold_label == nb_fold)[0]
    
    indices_test = test_and_val[:(len(test_and_val)//2)]
    indices_valid = test_and_val[(len(test_and_val)//2):]
    return(indices_train, indices_test, indices_valid)
















