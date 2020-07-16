# -*- coding: utf-8 -*-

# =============================================================================
# Preprocessing of Train and Test csv datasets
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tools

PATH_DATA = "../data/"

train = pd.read_csv(PATH_DATA + 'train.csv')

preproc = tools.preprocessing_data(train)

id = "ID00026637202179561894768"

other, fvc = tools.filter_data(preproc, id)

print(fvc.head())
print(other .head())