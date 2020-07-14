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
#Transform Weeks, FVC, Percent, Age to a normal distribution
for col in ["Weeks", "FVC", "Percent", "Age"]:
    df[col] = (df[col] - df[col].min())/(df[col].max() - df[col].min())
    df[col].plot(kind='hist')
    plt.show()

#Encode Sex and Smoker Status to one hot encoding
df = pd.get_dummies(df, columns=['Sex', 'SmokingStatus'])
print(df.head())  