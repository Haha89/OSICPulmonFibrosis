# -*- coding: utf-8 -*-

"""Data analysis of the CSV files provided,
Analysis of some caracteristics of the CT scans"""

from os import scandir
import pandas as  pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling as pdp
import pydicom
import tools


PATH_DATA = "../data/"

train = pd.read_csv(PATH_DATA + 'train.csv')
test = pd.read_csv(PATH_DATA + 'train.csv')
df = pd.concat([train, test])

# #Generation of a HTML file with graphs and analysis
profile_train_df = pdp.ProfileReport(df, title="Pandas Profiling Report", explorative=True)
profile_train_df.to_file("OSIC-EDA.html")

#Impact of smoking in the distributions of cases
sns.violinplot(x='SmokingStatus', y='Age', data=df, hue="Sex", palette='muted', split=True)
plt.title('Age Distributions Across Smoking Groups', size=16)
plt.legend(loc='lower left', ncol=2)
plt.show()

# #Evolution of FVC in time for one patient
id_target = 'ID00007637202177411956430'
sns.pointplot(x='Weeks', y='FVC', data=df[df['Patient'] == id_target])
plt.title(f"Evolution of FVC in time for {id_target}", size=16)
plt.show()

# =============================================================================
# Analysis on ct-scans
# =============================================================================

subfolders = [f.name for f in scandir(PATH_DATA + "train") if f.is_dir()]
nb_tranches, spacing = [], []
nb_rows, nb_col = [], []
resized_list = []

for id_patient in subfolders:
    scans = tools.get_scans_from_id(id_patient)
    nb_tranches.append(len(scans))
    if len(scans) > 100:
        print(id_patient)
    for scan in scans:
        data = pydicom.dcmread(f"{tools.get_path_id(id_patient)}/{scan}")
        nb_rows.append(data.Rows)
        nb_col.append(data.Columns)
        spacing.append(data.PixelSpacing)
        # resized = tools.normalize_scan(data)
        # resized_list.append(resized.shape)


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
plt.figure(figsize=(20, 10))
plt.bar(rep_tranches.keys(), height=rep_tranches.values())
plt.title("Number of slices per Scan")
plt.show()


# =============================================================================
# Quels sont les differents formats de scan (taille matrice)
# =============================================================================
print("Details on number of rows")
rep_rows = CountFrequency(nb_rows)
print(rep_rows)
plt.figure(figsize=(20, 10))
plt.bar(rep_rows.keys(), height=rep_rows.values())
plt.title("Repartition of ct-scans heights")
plt.show()


print("Details on number of columns")
rep_col = CountFrequency(nb_col)
print(rep_col)
plt.figure(figsize=(20, 10))
plt.bar(rep_col.keys(), height=rep_col.values())
plt.title("Repartition of ct-scans widths")
plt.show()

# =============================================================================
# Quels sont les pixel spacing
# =============================================================================
print("Details on spacings")

freq = {}
for item in spacing:
    key = f"{item[0]}"
    if key in freq:
        freq[key] += 1
    else:
        freq[key] = 1
print(freq)
