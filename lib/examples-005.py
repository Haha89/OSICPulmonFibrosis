# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:56:37 2020

@author: Alexandre
"""


import torch

checkpoint = torch.load('../data/model/model-0.pth')
torch.save(checkpoint['model'], '../data/model/model_6.pth')
torch.save(checkpoint['state_dict'], '../data/model/state_6.pth')