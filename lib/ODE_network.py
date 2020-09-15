# -*- coding: utf-8 -*-

"""Contains the definition of the class Convolutionnal_Network"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint


class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, nhidden)
        self.fc4 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        out = self.elu(out)
        out = self.fc4(out)
        return out


class ODE_Network(nn.Module):
    """Definition of our network"""

    def __init__(self, nb_features_in, nb_features_out,
                 shape, multiplicator,
                 hidden_dim_linear, misc_dim, lstm_size):
        """Definition of the init function"""
        super(ODE_Network, self).__init__()

        self.shape = shape
        self.input_dim = nb_features_in
        self.multiplicator = multiplicator
        self.hidden_dim_linear = hidden_dim_linear
        self.output_dim = nb_features_out
        self.misc_dim = misc_dim
        self.hidden_lstm = lstm_size
        self.dropout = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.05)
        self.func = LatentODEfunc(self.output_dim,self.output_dim*4)
        
        # Define the 3D convolutionnal layers 512x512
        self.Conv01 = nn.Conv3d(self.input_dim,
                                self.input_dim*self.multiplicator,
                                kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.Conv02 = nn.Conv3d(self.input_dim*self.multiplicator,
                                self.input_dim*self.multiplicator,
                                kernel_size=(3, 3, 3), padding=(1, 1, 1))

        #self.bn01 = nn.BatchNorm3d(self.input_dim*self.multiplicator)
        #self.bn02 = nn.BatchNorm3d(self.input_dim*self.multiplicator)

        self.reduce0 = nn.MaxPool3d(kernel_size=(16, 16, 16))
        self.pool0 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # Define the 3D convolutionnal layers 216x216
        self.Conv11 = nn.Conv3d(self.input_dim*self.multiplicator,
                                self.input_dim*self.multiplicator*2,
                                kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.Conv12 = nn.Conv3d(self.input_dim*self.multiplicator*2,
                                self.input_dim*self.multiplicator*2,
                                kernel_size=(3, 3, 3), padding=(1, 1, 1))

        #self.bn11 = nn.BatchNorm3d(self.input_dim*self.multiplicator*2)
        #self.bn12 = nn.BatchNorm3d(self.input_dim*self.multiplicator*2)

        self.reduce1 = nn.MaxPool3d(kernel_size=(8, 8, 8))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))


        # Define the 3D convolutionnal layers 128x128
        self.Conv21 = nn.Conv3d(self.input_dim*self.multiplicator*2,
                                self.input_dim*self.multiplicator*4,
                                kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.Conv22 = nn.Conv3d(self.input_dim*self.multiplicator*4,
                                self.input_dim*self.multiplicator*4,
                                kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.reduce2 = nn.MaxPool3d(kernel_size=(4, 4, 4))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        #self.bn21 = nn.BatchNorm3d(self.input_dim*self.multiplicator*4)
        #self.bn22 = nn.BatchNorm3d(self.input_dim*self.multiplicator*4)

        # Define the 3D convolutionnal layers 64x64
        self.Conv31 = nn.Conv3d(self.input_dim*self.multiplicator*4,
                                self.input_dim*self.multiplicator*8,
                                kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.Conv32 = nn.Conv3d(self.input_dim*self.multiplicator*8,
                                self.input_dim*self.multiplicator*8,
                                kernel_size=(3, 3, 3), padding=(1, 1, 1))

        #self.bn31 = nn.BatchNorm3d(self.input_dim*self.multiplicator*8)
        #self.bn32 = nn.BatchNorm3d(self.input_dim*self.multiplicator*8)

        self.reduce3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))


        # Define the 3D convolutionnal layers 32x32
        self.Conv41 = nn.Conv3d(self.input_dim*self.multiplicator*8,
                                self.input_dim*self.multiplicator*16,
                                kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.Conv42 = nn.Conv3d(self.input_dim*self.multiplicator*16,
                                self.input_dim*self.multiplicator*16,
                                kernel_size=(3, 3, 3), padding=(1, 1, 1))

        #self.bn41 = nn.BatchNorm3d(self.input_dim*self.multiplicator*16)
        #self.bn42 = nn.BatchNorm3d(self.input_dim*self.multiplicator*16)


       # Post_processing
        input_dim_pp = self.input_dim*self.multiplicator*(16 + 8 + 4 + 2 + 1)*self.shape[0]*self.shape[1]*self.shape[2]//(16*16*16)
        self.postpross1 = nn.Linear(input_dim_pp, self.hidden_dim_linear)
        
        #self.postpross2 = nn.Linear(self.hidden_dim_linear*2, self.hidden_dim_linear)
        self.out = nn.Linear(self.hidden_dim_linear, self.output_dim)
        self.data_process1 = nn.Linear(self.output_dim + self.misc_dim + 3,
                                        self.output_dim)
        # self.data_process2 = nn.Linear(self.hidden_dim_linear,
        #                                 self.output_dim)
        # self.decode1 = nn.Linear(self.output_dim,
        #                                 self.hidden_dim_linear)
        self.decode2 = nn.Linear(self.output_dim, 2) 



    def forward(self, scans, misc, fvc, percent, weeks):
        """Forward function"""
        # batch_size, channels, depth, width, height = scans.shape
        
        x = F.relu(self.Conv01(scans))
        # x = self.dropout5(x) #ALEX
        x = F.relu(self.Conv02(x))
        # x = self.dropout5(x) #ALEX

        interm0 = self.reduce0(x)
        x = self.pool0(x)

        x = F.relu(self.Conv11(x))
        # x = self.dropout5(x) #ALEX
        x = F.relu(self.Conv12(x))
        # x = self.dropout5(x) #ALEX

        interm1 = self.reduce1(x)
        x = self.pool1(x)

        x = F.relu(self.Conv21(x))
        # x = self.dropout5(x) #ALEX
        x = F.relu(self.Conv22(x))
        # x = self.dropout5(x) #ALEX

        interm2 = self.reduce2(x)
        x = self.pool2(x)

        x = F.relu(self.Conv31(x))
        x = F.relu(self.Conv32(x))

        interm3 = self.reduce3(x)
        x = self.pool3(x)
        
        # x = self.dropout(x) #ALEX
        x = F.relu(self.Conv41(x))
        # x = self.dropout(x)
        x = F.relu(self.Conv42(x))
        x = torch.cat((x, interm0, interm1, interm2, interm3), dim=1)
        
        batch_size, nb_features, depth, width, height = x.shape
        x = x.view(batch_size, -1)
        # x = self.dropout(x)
        x = F.relu(self.postpross1(x))
        # x = self.dropout(x)
        #x = F.relu(self.postpross2(x))
        outputs_scan = self.out(x)
        
        evolution = torch.cat((outputs_scan, misc, fvc, percent), -1)
        # evolution = self.dropout(evolution)
        evolution = F.relu(self.data_process1(evolution))
        # evolution = self.dropout(evolution) 
        # evolution = F.relu(self.data_process2(evolution))
        latent = odeint(self.func,evolution, weeks.squeeze(0)).permute(1,0,2)
                
        # evolution = self.dropout(latent)
        # evolution = F.relu(self.decode1(evolution))
        # evolution = self.dropout(evolution)
        output = self.decode2(latent)
        
        # output[:,:,0] = nn.Sigmoid()(output[:,:,0])
        # output[:,:,1] = nn.Sigmoid()(output[:,:,1])
        # output = torch.sigmoid(output)
        return nn.Sigmoid()(output)
