# -*- coding: utf-8 -*-

"""Contains the definition of the class Convolutionnal_Network"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Convolutionnal_Network(nn.Module):
    """Definition of our network"""

    def __init__(self, nb_features_in, nb_features_out,
                 shape, multiplicator,
                 hidden_dim_linear, misc_dim, lstm_size):
        """Definition of the init function"""
        super(Convolutionnal_Network, self).__init__()

        self.shape = shape
        self.input_dim = nb_features_in
        self.multiplicator = multiplicator
        self.hidden_dim_linear = hidden_dim_linear
        self.output_dim = nb_features_out
        self.misc_dim = misc_dim
        self.hidden_lstm = lstm_size

        # Define the 3D convolutionnal layers 512x512
        self.Conv01 = nn.Conv3d(self.input_dim,
                                self.input_dim*self.multiplicator,
                                kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.Conv02 = nn.Conv3d(self.input_dim*self.multiplicator,
                                self.input_dim*self.multiplicator,
                                kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.bn01 = nn.BatchNorm3d(self.input_dim*self.multiplicator)
        self.bn02 = nn.BatchNorm3d(self.input_dim*self.multiplicator)

        self.reduce0 = nn.MaxPool3d(kernel_size=(16, 16, 16))
        self.pool0 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # Define the 3D convolutionnal layers 216x216
        self.Conv11 = nn.Conv3d(self.input_dim*self.multiplicator,
                                self.input_dim*self.multiplicator*2,
                                kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.Conv12 = nn.Conv3d(self.input_dim*self.multiplicator*2,
                                self.input_dim*self.multiplicator*2,
                                kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.bn11 = nn.BatchNorm3d(self.input_dim*self.multiplicator*2)
        self.bn12 = nn.BatchNorm3d(self.input_dim*self.multiplicator*2)

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

        self.bn21 = nn.BatchNorm3d(self.input_dim*self.multiplicator*4)
        self.bn22 = nn.BatchNorm3d(self.input_dim*self.multiplicator*4)

        # Define the 3D convolutionnal layers 64x64
        self.Conv31 = nn.Conv3d(self.input_dim*self.multiplicator*4,
                                self.input_dim*self.multiplicator*8,
                                kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.Conv32 = nn.Conv3d(self.input_dim*self.multiplicator*8,
                                self.input_dim*self.multiplicator*8,
                                kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.bn31 = nn.BatchNorm3d(self.input_dim*self.multiplicator*8)
        self.bn32 = nn.BatchNorm3d(self.input_dim*self.multiplicator*8)

        self.reduce3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))


        # Define the 3D convolutionnal layers 32x32
        self.Conv41 = nn.Conv3d(self.input_dim*self.multiplicator*8,
                                self.input_dim*self.multiplicator*16,
                                kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.Conv42 = nn.Conv3d(self.input_dim*self.multiplicator*16,
                                self.input_dim*self.multiplicator*16,
                                kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.bn41 = nn.BatchNorm3d(self.input_dim*self.multiplicator*16)
        self.bn42 = nn.BatchNorm3d(self.input_dim*self.multiplicator*16)


       # Post_processing
        input_dim_pp = self.input_dim*self.multiplicator*(16 + 8 + 4 + 2 + 1)*self.shape[0]*self.shape[1]*self.shape[2]//(16*16*16)
        self.postpross1 = nn.Linear(input_dim_pp, self.hidden_dim_linear*2)
        self.postpross2 = nn.Linear(self.hidden_dim_linear*2, self.hidden_dim_linear)

        self.out = nn.Linear(self.hidden_dim_linear, self.output_dim)

        self.data_process1 = nn.Linear(self.output_dim + self.misc_dim,
                                       self.hidden_dim_linear)
        self.data_process2 = nn.Linear(self.hidden_dim_linear,
                                       self.output_dim)

        self.evolution_process1 = nn.Linear(self.output_dim + 2,
                                            self.hidden_dim_linear)
        self.evolution_process2 = nn.Linear(self.hidden_dim_linear,
                                            self.output_dim)

        self.LSTM1 = nn.LSTM(self.output_dim, self.hidden_lstm,
                             num_layers=1, batch_first=True)
        self.LSTM2 = nn.LSTM(self.output_dim + self.hidden_lstm,
                             self.hidden_lstm, num_layers=1, batch_first=True)

        self.postprocess1 = nn.Linear(self.hidden_lstm, self.hidden_dim_linear)
        self.postprocess2 = nn.Linear(self.hidden_dim_linear, 2)


    def forward(self, scans, misc, fvc, percent):
        """Forward function"""
        # batch_size, channels, depth, width, height = scans.shape

        x = F.relu(self.bn01(self.Conv01(scans)))
        x = F.relu(self.bn02(self.Conv02(x)))

        interm0 = self.reduce0(x)
        x = self.pool0(x)

        x = F.relu(self.bn11(self.Conv11(x)))
        x = F.relu(self.bn12(self.Conv12(x)))

        interm1 = self.reduce1(x)
        x = self.pool1(x)

        x = F.relu(self.bn21(self.Conv21(x)))
        x = F.relu(self.bn22(self.Conv22(x)))

        interm2 = self.reduce2(x)
        x = self.pool2(x)

        x = F.relu(self.bn31(self.Conv31(x)))
        x = F.relu(self.bn32(self.Conv32(x)))

        interm3 = self.reduce3(x)
        x = self.pool3(x)

        x = F.relu(self.bn41(self.Conv41(x)))
        x = F.relu(self.bn42(self.Conv42(x)))
        x = torch.cat((x, interm0, interm1, interm2, interm3), dim=1)

        batch_size, nb_features, depth, width, height = x.shape

        x = x.view(batch_size, -1)

        x = F.relu(self.postpross1(x))
        x = F.relu(self.postpross2(x))

        outputs_scan = self.out(x)

        leng = fvc.shape[-1]
        scans_over_time = outputs_scan.unsqueeze(1)
        scans_over_time = scans_over_time.expand(batch_size, leng, self.output_dim)

        evolution = torch.cat((scans_over_time, misc), -1)
        evolution = F.relu(self.data_process1(evolution))
        evolution = F.relu(self.data_process2(evolution))

        fvc = fvc.view(batch_size, leng, 1)
        percent = percent.view(batch_size, leng, 1)

        evolution = torch.cat((evolution, fvc, percent), -1)
        evolution = F.relu(self.evolution_process1(evolution))
        evolution = F.relu(self.evolution_process2(evolution))

        out1, _ = self.LSTM1(evolution, None)
        evolution = torch.cat((evolution, out1), -1)
        evolution, _ = self.LSTM2(evolution, None)
        evolution = evolution.contiguous()

        evolution = F.relu(self.postprocess1(evolution))
        output = self.postprocess2(evolution)
        return output
