# -*- coding: utf-8 -*-
"""
This script opens each patient's CT scan, generates the 3d array (128*128*128)
and saves it in the folder ../data/scans/
"""

from os import listdir
import numpy as np

import pydicom
from scipy.ndimage import zoom
from tools import get_path_id, get_scans_from_id, get_id_folders

PATH_DATA = "../data/"
PIXEL_SPACING = 0.8
SPACING_Z = 3
SCAN_SIZE = [128, 128, 128] #z, x, y


def create_3d_scan(id):
    """Return a 3d matrix of the different slices (ct scans) of a patient,
    the list of slice heights and widths"""
    path_data = get_path_id(id)
    filelist = get_scans_from_id(id)
    slice_agg, spacing, y_pos = [], 0., []
    try:
        for file in filelist:
            data = pydicom.dcmread(f"{path_data}/{file}")
            slice_agg.append(data.pixel_array)
            spacing = data.PixelSpacing
            y_pos.append(data.SliceLocation)
       
        space_z = abs(float(y_pos[12])-float(y_pos[11]))
        return np.array(slice_agg), spacing, space_z
    except :
        
        for file in filelist:
            data = pydicom.dcmread(f"{path_data}/{file}")
            slice_agg.append(data.pixel_array)
            spacing = data.PixelSpacing
        try:    
            return np.array(slice_agg), spacing, data.SpacingBetweenSlices
        except:
            return np.array(slice_agg), spacing, data.SliceThickness



def process_3d_scan(id_patient=None, indice=None):
    """Returns the 3d scan array of a patient,
    as a 128*128*128 array, values between 0 and 1"""

    if id_patient is None:
        id_patient = get_id_folders(indice)
    matrix, spacing, space_z = create_3d_scan(id_patient)

    #Resizing factors
    fx = spacing[0]/PIXEL_SPACING
    fy = spacing[1]/PIXEL_SPACING
    fz = space_z/SPACING_Z
    
    #cut 128/fx x128/fy x128/fz
    z, x, y = matrix.shape
    startx = max(0, (x - SCAN_SIZE[1]/fx)//2)
    starty = max(0, (y - SCAN_SIZE[2]/fy)//2)
    startz = max(0, (z - SCAN_SIZE[0]/fz)//2)

    resized_mat = matrix[int(startz):int(startz+SCAN_SIZE[0]//fz),
                         int(startx):int(startx+SCAN_SIZE[1]//fx),
                         int(starty):int(starty+SCAN_SIZE[2]//fy)]
    resized_mat = zoom(resized_mat, (fz, fx, fy))

    #Add padding based on size
    z, x, y = resized_mat.shape
    z1 = (SCAN_SIZE[0] - z)//2
    z2 = (SCAN_SIZE[0]- z + 1)//2
    y1 = (SCAN_SIZE[2] - y)//2
    y2 = (SCAN_SIZE[2] - y + 1)//2
    x1 = (SCAN_SIZE[1] - x)//2
    x2 = (SCAN_SIZE[1] - x + 1)//2

    processed_mat = np.pad(resized_mat, ((z1, z2), (x1, x2), (y1, y2)), 'constant')
    min_matrix = np.min(processed_mat) #Normalization
    return (processed_mat - min_matrix)/(np.max(processed_mat) - min_matrix)


FOLDERS = listdir(PATH_DATA + "train/")
for i, patient in enumerate(["ID00128637202219474716089", "ID00132637202222178761324"]): #enumerate(FOLDERS):
    try:
        with open(f'{PATH_DATA}scans/{patient}.npy', 'wb') as f:
            np.save(f, process_3d_scan(id_patient=patient))
    except:
        print(patient)

