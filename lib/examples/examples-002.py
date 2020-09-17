# -*- coding: utf-8 -*-

"""Preprocessing of the Train dataset, display of content of one patient"""

import pandas as pd
import tools

PATH_DATA = "../data/"

TRAIN = pd.read_csv(PATH_DATA + 'train.csv')
PREPROC = tools.preprocessing_data(TRAIN)

ID_PATIENT = "ID00026637202179561894768"
OTHER, FVC, PERCENT = tools.filter_data(PREPROC, ID_PATIENT)

print(FVC)
print(OTHER)
print(PERCENT)
