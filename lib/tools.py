# -*- coding: utf-8 -*-

from os import listdir, path, scandir
import torch
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from random import choice
import pandas as pd
import sys

PATH_DATA = "../data/"
PIXEL_SPACING = 0.8
THICKNESS = 1
SCAN_SIZE = [128, 128, 128] #z, x, y

def get_id_folders(indice):
    try:
        return listdir(PATH_DATA + "train/")[indice]
    except:
        print(f"Error for indice {indice}")
        return listdir(PATH_DATA + "train/")[0]
    
def get_path_id(id):
    """Returns the path of the folder containing the patient ID CT scans.
    Notifies if the path is not found"""
    path_folder = f"{PATH_DATA}train/{id}"
    if path.isdir(path_folder):
        return path_folder
    print(f"Could not find the folder for patient: {id}")

    
def get_scans_from_id(id):
    """Returns an ordered list of CT scans filenames from the client id"""
    path_folder = get_path_id(id)
    if path_folder:
        return sorted(listdir(path_folder), key=lambda f : int(f.split(".")[0]))

def crop_slice(s):

    """
    Crop frames from slices

    Parameters
    ----------
    s : numpy array, shape = (Rows, Columns)
    numpy array of slices with frame

    Returns
    -------
    s_cropped : numpy array, shape = (Rows - All Zero Rows, Columns - All Zero Columns)
    numpy array after the all zero rows and columns are dropped
    """

    s_cropped = s[~np.all(s == 0, axis=1)]
    s_cropped = s_cropped[:, ~np.all(s_cropped == 0, axis=0)]
    return s_cropped


def multi_slice_viewer(matrix_3d):
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
    ax.volume = matrix_3d
    ax.index = matrix_3d.shape[0] // 2
    ax.imshow(matrix_3d[ax.index], cmap=plt.cm.bone)
    fig.canvas.mpl_connect('scroll_event', process_key)
    plt.show()


def get_random_scan():
    """Returns a random scan contained in the train data set"""
    subfolders = [ f.name for f in scandir(PATH_DATA + "train") if f.is_dir()]
    random_id = choice(subfolders) #Random element from list of id
    random_scan = choice(get_scans_from_id(random_id)) #Random scan in the folder
    try:
        print(f"Random scan for {random_id}, file {random_scan}")
        return pydicom.dcmread(f"{get_path_id(random_id)}/{random_scan}")
    except:
        print(f"Error during the random scan for {random_id}, file {random_scan}")
    
    
def get_specific_scan(id, scan_number):
    """Returns the data of a specific patient, specific scan"""
    return pydicom.dcmread(f"{get_path_id(id)}/{scan_number}.dcm")

def unormalize_fvc(data):
    return(data["FVC"].min(),data["FVC"].max())

def preprocessing_data(data):
    #Creation of one hot encoder, normalisation between [0,1]
    data = pd.get_dummies(data, columns=['Sex', 'SmokingStatus'])
    # Transform Weeks, FVC, Percent, Age to be in [0, 1]
    for col in ["Weeks", "FVC", "Age"]:
        data[col] = (data[col] - data[col].min())/(data[col].max() - data[col].min())
    data["Percent"] = data["Percent"]/100.
    return data

def filter_data(data, id_patient=None, indice=None):
    if id_patient is None:
        id_patient = get_id_folders(indice)
    """Returns the data only for the id_patient"""
    filtered_data = data[data.Patient == id_patient]
    fvc = torch.tensor(filtered_data.FVC.values)
    percent = torch.tensor(filtered_data.Percent.values)

    misc = torch.zeros((len(fvc), 4))
    misc[:,0] = torch.tensor(filtered_data.Weeks.values)
    misc[:,1] = torch.tensor(filtered_data.Age.values)
    misc[:,2] = torch.tensor(filtered_data.Sex_Male.values)
    misc[:,3] = torch.tensor(0.5*np.array(filtered_data['SmokingStatus_Currently smokes']) +\
        np.array(filtered_data['SmokingStatus_Ex-smoker']))

    return misc, fvc, percent


def get_data(path): 
    raw_data = pd.read_csv(path + 'train.csv')
    #mini, maxi = unormalize_fvc(raw_data)
    #np.save("minmax",np.array([mini,maxi]))
    return (preprocessing_data(raw_data))