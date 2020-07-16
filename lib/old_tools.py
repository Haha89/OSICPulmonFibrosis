# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 12:23:55 2020

@author: Alexandre
"""
from os import listdir, path, scandir
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from random import choice
import cv2
import pandas as pd
from scipy.ndimage import zoom
import sys

PATH_DATA = "../data/"
PIXEL_SPACING = 0.8
THICKNESS = 1
SCAN_SIZE = [128, 128, 128] #z, x, y

def normalize_scan(scan, size_target=[128,128]):
    """Resize the scan and normalize it (values between 0 and 1).
    The output matrix is 128*128 with a pixel spacing of 0.4"""
    res = cv2.resize(scan.pixel_array, 
                     (int(scan.PixelSpacing[0]/PIXEL_SPACING*scan.Rows),
                      int(scan.PixelSpacing[1]/PIXEL_SPACING*scan.Columns)))
    x, y = np.shape(res)
    if (x < size_target[0]) or (y < size_target[1]): #Result smaller than 128*128
        pass
    else: #Picture too big, need to crop. Will keep only the center of picture
        y1, y2 = int(.5*(y - size_target[1])), int(.5*(y + size_target[1]))
        x1, x2 = int(.5*(x - size_target[0])), int(.5*(x + size_target[0]))
        res = res[y1:y2, x1:x2]
    return np.array((res - np.min(res))/(np.max(res) - np.min(res))) #Normalisation between [0,1]
