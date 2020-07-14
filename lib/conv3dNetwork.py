#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:50:48 2019

@author: brou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

device = 'cuda:0'

"""
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICE"] = "0"
device = ("0" if torch.cuda.is_available() else "cpu" )
"""
# Dossier oÃ¹ se trouvent les fichiers + j'ai 2 dossiers Pred_lstm et Pred_auxiliary pour stocker les prÃ©dictions

class Convolutionnal_Network(nn.Module):

    def __init__(self, nb_features_in, nb_features_out, shape, multiplicator, hidden_dim_linear):
        super(Convolutionnal_Network, self).__init__()
        
        self.shape = shape
        self.input_dim = nb_features_in
        self.multiplicator = multiplicator
        self.hidden_dim_linear = hidden_dim_linear
        self.output_dim = nb_features_out
        
        # Define the 3D convolutionnal layers 512x512
        self.Conv01 =  nn.Conv3d(self.input_dim,self.input_dim*multiplicator, kernel_size=(3,3,3), padding=(1,1,1))
        self.Conv02 =  nn.Conv3d(self.input_dim*multiplicator,self.input_dim*multiplicator, kernel_size=(3,3,3), padding=(1,1,1))
        
        self.reduce0  = nn.MaxPool3d(kernel_size=(8, 16, 16))
        self.pool0  = nn.MaxPool3d(kernel_size=(1, 2, 2))

        # Define the 3D convolutionnal layers 216x216
        self.Conv11 =  nn.Conv3d(self.input_dim*multiplicator,self.input_dim*multiplicator*2, kernel_size=(3,3,3), padding=(1,1,1))
        self.Conv12 =  nn.Conv3d(self.input_dim*multiplicator*2,self.input_dim*multiplicator*2, kernel_size=(3,3,3), padding=(1,1,1))

        self.reduce1  = nn.MaxPool3d(kernel_size=(8, 8, 8))
        self.pool1  = nn.MaxPool3d(kernel_size=(2, 2, 2))


        # Define the 3D convolutionnal layers 128x128
        self.Conv21 =  nn.Conv3d(self.input_dim*multiplicator*2,self.input_dim*multiplicator*4, kernel_size=(3,3,3), padding=(1,1,1))
        self.Conv22 =  nn.Conv3d(self.input_dim*multiplicator*4,self.input_dim*multiplicator*4, kernel_size=(3,3,3), padding=(1,1,1))

        self.reduce2  = nn.MaxPool3d(kernel_size=(4, 4, 4))
        self.pool2  = nn.MaxPool3d(kernel_size=(2, 2, 2))


        # Define the 3D convolutionnal layers 64x64
        self.Conv31 =  nn.Conv3d(self.input_dim*multiplicator*4,self.input_dim*multiplicator*8, kernel_size=(3,3,3), padding=(1,1,1))
        self.Conv32 =  nn.Conv3d(self.input_dim*multiplicator*8,self.input_dim*multiplicator*8, kernel_size=(3,3,3), padding=(1,1,1))

        self.reduce3  = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.pool3  = nn.MaxPool3d(kernel_size=(2, 2, 2))


        # Define the 3D convolutionnal layers 32x32
        self.Conv41 =  nn.Conv3d(self.input_dim*multiplicator*8,self.input_dim*multiplicator*16, kernel_size=(3,3,3), padding=(1,1,1))
        self.Conv42 =  nn.Conv3d(self.input_dim*multiplicator*16,self.input_dim*multiplicator*16, kernel_size=(3,3,3), padding=(1,1,1))


       # Post_processing 

        self.postpross1 = nn.Linear(self.input_dim*multiplicator*(16+8+4+2+1)*self.shape[0]*self*shape[1]*self.shape[2]//(16*16*8),self.hidden_dim_linear*2)
        self.postpross2 = nn.Linear(self.hidden_dim_linear*2,self.hidden_dim_linear)

        

        self.out = nn.Linear(self.hidden_dim_linear,self.nb_features_out)

        
    def forward(self, x):
        
        batch_size, depth, width, height = x.shape
        

        
        x = F.relu(self.Conv01(x))
        x = F.relu(self.Conv02(x))
 
        interm0 = self.reduce0(x)
        x = self.pool0(x)


        x = F.relu(self.Conv11(x))
        x = F.relu(self.Conv12(x))
 
        interm1 = self.reduce1(x)
        x = self.pool1(x)
        
        
        x = F.relu(self.Conv21(x))
        x = F.relu(self.Conv22(x))
 
        interm2 = self.reduce2(x)
        x = self.pool2(x)
        
        x = F.relu(self.Conv31(x))
        x = F.relu(self.Conv32(x))

        interm3 = self.reduce2(x)
        x = self.pool3(x)
        
        x = F.relu(self.Conv41(x))
        x = F.relu(self.Conv42(x))
        
        x = torch.cat((x,interm0,interm1,interm2, interm3)) 
        batch_size, depth, width, height = x.shape
        
        x = x.view(batch_size,-1)
        
        x = F.relu(self.postpross1(x))
        x = F.relu(self.postpross2(x))

        outputs = self.out(x)
        return outputs
    
    
    def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence):
        """
        Calculates the modified Laplace Log Likelihood score for this competition.
        """
        std_min = torch.tensor([70]).cuda()
        delta_max = torch.tensor([1000]).cuda()
        
        std_clipped = torch.max(confidence, std_min)
        delta = torch.min(torch.abs(actual_fvc - predicted_fvc), delta_max)
        
        metric = - sqrt(2) * delta / std_clipped - torch.log(sqrt(2) * std_clipped)
    
        return -torch.mean(metric)

    
   
    







