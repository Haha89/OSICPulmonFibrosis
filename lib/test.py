# -*- coding: utf-8 -*-

# import tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH_DATA = "../data/"

train = pd.read_csv(PATH_DATA + 'train.csv')
test = pd.read_csv(PATH_DATA + 'train.csv')
df = pd.concat([train, test])

