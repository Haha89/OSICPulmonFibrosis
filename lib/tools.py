# -*- coding: utf-8 -*-

from os import listdir, path, scandir
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from random import choice
import cv2

PATH_DATA = "../data/"

def get_path_id(id):
    """Returns the path of the folder containing the patient ID CT scans.
    Notifies if the path is not found"""
    path_folder = PATH_DATA + "train/" + id
    if path.isdir(path_folder):
        return path_folder
    print(f"Could not find the folder for patient: {id}")

    
def get_scans_from_id(id):
    """Returns an ordered list of CT scans filenames from the client id"""
    path_folder = get_path_id(id)
    if path_folder:
        return sorted(listdir(path_folder), key=lambda f : int(f.split(".")[0]))


def get_3d_scan(id, normalized=True):
    """Return a 3d matrix of the different slices (ct scans) of a patient, 
    the list of slice heights and widths"""
    path_data = get_path_id(id)
    filelist = get_scans_from_id(id)
    slice_agg, heights, widths = [], [], []
    try:
        for file in filelist:
            data = pydicom.dcmread(f"{path_data}/{file}")
            if normalized:
                slice_agg.append(normalize_scan(data.pixel_array))
            else:
                slice_agg.append(data.pixel_array)
            heights.append(float(data.SliceLocation))
            widths.append(float(data.SliceThickness))
    
        return np.array(slice_agg), heights, widths
    except:
        print(f"Error in the creation of the scan for {id}")
    
    
def multi_slice_viewer(matrix):
    """Visualization of the matrix slice by slice.
    Allegrement Stolen online"""
    
    def process_key(event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])
        fig.canvas.draw()
    
    fig, ax = plt.subplots()
    ax.volume = matrix
    ax.index = matrix.shape[0] // 2
    ax.imshow(matrix[ax.index], cmap=plt.cm.bone)
    fig.canvas.mpl_connect('scroll_event', process_key)
    plt.show()
    
    
def get_random_scan():
    """Returns a random scan contained in the train data set"""
    subfolders = [ f.name for f in scandir(PATH_DATA + "train") if f.is_dir() ]
    random_id = choice(subfolders) #Random element from list
    random_scan = choice(get_scans_from_id(random_id))
    data = pydicom.dcmread(f"{get_path_id(random_id)}/{random_scan}")
    try:
        print(f"Random scan for {random_id}, file {random_scan}")
        return data.pixel_array, float(data.SliceLocation), float(data.SliceThickness)
    except:
        print(f"Error during the random scan for {random_id}, file {random_scan}")
    
    
def get_specific_scan(id, scan_number):
    """Returns the data of a specific patient, specific scan"""
    return pydicom.dcmread(f"{get_path_id(id)}/{scan_number}.dcm")

def normalize_scan(scan, size=(512,512)):
    """Resize the scan and normalize it (values between 0 and 1)"""
    res = cv2.resize(scan, dsize=size)
    min_array = np.min(res)
    return (res - min_array)/(np.max(res) - min_array)  
    