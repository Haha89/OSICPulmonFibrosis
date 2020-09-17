# -*- coding: utf-8 -*-

"""Display of the data post training"""
import torch
from os import listdir


PATH_DATA = '../data/'
PATH_HISTO = PATH_DATA + "/saved_data/"

with open(PATH_HISTO + listdir(PATH_HISTO)[0], "rb") as file:
    data = torch.load(file)
    print("Weeks")
    print(data['weeks'])
    print("FVC")
    print(data['fvc'])
    print("MISC")
    print(data['misc'])
    print("Goal")
    print(data['goal'])
    print("MEAN")
    print(data['mean'])
    print("STD")
    print(data['std'])
