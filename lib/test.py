# -*- coding: utf-8 -*-

# import tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH_DATA = "../data/"

train = pd.read_csv(PATH_DATA + 'train.csv')
test = pd.read_csv(PATH_DATA + 'train.csv')
df = pd.concat([train, test])

#Preprocessing
print(df.head())

#Encode Sex and Smoker Status to one hot encoding
df = pd.get_dummies(df, columns=['Sex', 'SmokingStatus'])

# =============================================================================
# Transform Weeks, FVC, Percent, Age to be in [0, 1]
# =============================================================================
# for col in ["Weeks", "FVC", "Percent", "Age"]:
#     df[col] = (df[col] - df[col].min())/(df[col].max() - df[col].min())
#     df[col].plot(kind='hist')
#     plt.show()

# =============================================================================
# Transformation pour etre presque des lois normales TODO
# =============================================================================
from sklearn.preprocessing import PowerTransformer
yj = PowerTransformer(method='yeo-johnson')





print(df.head())  