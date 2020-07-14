# -*- coding: utf-8 -*-

import pandas as  pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling as pdp
from os import scandir
import pydicom
import tools
from tqdm import tqdm

PATH_DATA = "../data/"

train = pd.read_csv(PATH_DATA + 'train.csv')
test = pd.read_csv(PATH_DATA + 'train.csv')
df = pd.concat([train, test])

#Generation of a HTML file with graphs and analysis
profile_train_df = pdp.ProfileReport(df, title="Pandas Profiling Report", explorative=True)
profile_train_df.to_file("OSIC-EDA.html")


sns.violinplot(x='SmokingStatus', y='Age', data=df, hue="Sex", palette='muted', split=True)
plt.title('Age Distributions Across Smoking Groups', size=16)
plt.legend(loc='lower left', ncol=2)
plt.show()

#Evolution of FVC in time for one patient
id_target = 'ID00007637202177411956430'
sns.pointplot(x='Weeks', y='FVC', data=df[df['Patient']==id_target])
plt.title(f"Evolution of FVC in time for {id_target}", size=16)
plt.show()

# =============================================================================
# Analysis on ct-scans
# =============================================================================

subfolders = [f.name for f in scandir(PATH_DATA + "train") if f.is_dir()]
nb_tranches = []
nb_rows, nb_col = [], []

for id_patient in tqdm(subfolders):
    scans = tools.get_scans_from_id(id_patient)
    nb_tranches.append(len(scans))
    for scan in scans:
        data = pydicom.dcmread(f"{tools.get_path_id(id_patient)}/{scan}")
        nb_rows.append(data.Rows)
        nb_col.append(data.Columns)
        

def CountFrequency(my_list): 
    """ Transfroms a list to a dictionnary where:
        - keys are list's values 
        - values are occurence in the list"""
    freq = {} 
    for item in my_list: 
        if item in freq: 
            freq[item] += 1
        else: 
            freq[item] = 1
    return freq
        
# =============================================================================
# Combien y'a t'il de tranches par scan ?
# =============================================================================
print("Details on number of slices")
rep_tranches = CountFrequency(nb_tranches)
print(rep_tranches)
plt.figure(figsize=(20,10))
plt.bar(rep_tranches.keys(), height=rep_tranches.values())
plt.title("Number of slices per Scan")
plt.show()

# =============================================================================
# Quels sont les differents formats de scan (taille matrice)
# =============================================================================
print("Details on number of rows")
rep_rows = CountFrequency(nb_rows)
print(rep_rows)
plt.figure(figsize=(20,10))
plt.bar(rep_rows.keys(), height=rep_rows.values())
plt.title("Repartition of ct-scans heights")
plt.show()


print("Details on number of columns")
rep_col = CountFrequency(nb_col)
print(rep_col)
plt.figure(figsize=(20,10))
plt.bar(rep_col.keys(), height=rep_col.values())
plt.title("Repartition of ct-scans widths")
plt.show()