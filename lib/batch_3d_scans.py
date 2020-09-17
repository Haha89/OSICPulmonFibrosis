# -*- coding: utf-8 -*-
"""
This script opens each patient's CT scan, generates the 3d array (128*128*128)
and saves it in the folder ../data/scans/
"""

from os import listdir
import numpy as np
from scan_processing import process_3d_scan

PATH_DATA = "../data/"

if __name__ == "__main__": 
    
    patients_ids = listdir(PATH_DATA + "train/")
    for i, patient in enumerate(patients_ids): 
        try:
            data = process_3d_scan(id_patient=patient)
            with open(f'{PATH_DATA}scans/{patient}.npy', 'wb') as f:
                np.save(f, data)
            print(f"Index {i+1}/{len(patients_ids)}")  
        except:
            print(i+1, patient)
