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
test = pd.read_csv(PATH_DATA + 'train.csv')
df = pd.concat([train, test])
print(tools.preprocessing_data(df))