#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:57:38 2020

@author: brou
"""

import torch
from math import sqrt


def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence):
    """
    Calculates the modified Laplace Log Likelihood score for this competition.
    """
    std_min = torch.tensor([70]).cuda()
    delta_max = torch.tensor([1000]).cuda()
    
    std_clipped = torch.max(confidence, std_min)
    delta = torch.min(torch.abs(actual_fvc - predicted_fvc), delta_max)
    
    metric = - sqrt(2) * delta / std_clipped - torch.log(sqrt(2) * std_clipped)

    return - torch.mean(metric)