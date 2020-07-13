# -*- coding: utf-8 -*-

from os import scandir
import matplotlib.pyplot as plt
import pydicom
import tools


PATH_DATA = "../data/"

subfolders = [f.name for f in scandir(PATH_DATA + "train") if f.is_dir()]
nb_tranches = []
nb_rows, nb_col = [], []

for id_patient in subfolders:
    scans = tools.get_scans_from_id(id_patient)
    nb_tranches.append(len(scans))
    for scan in scans:
        data = pydicom.dcmread(f"{tools.get_path_id(id_patient)}/{scan}")
        nb_rows.append(data.Rows)
        nb_col.append(data.Columns)
        

def CountFrequency(my_list): 
    # Creating an empty dictionary  
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
plt.bar(rep_tranches.keys(), height=rep_tranches.values())

# =============================================================================
# Quels sont les differents formats de scan (taille matrice)
# =============================================================================
print("Details on number of rows")
print(CountFrequency(nb_rows))

print("Details on number of columns")
print(CountFrequency(nb_col))
