# -*- coding: utf-8 -*-

from os import listdir, path, scandir
import torch
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from random import choice
import pandas as pd
from scipy.ndimage import zoom
import sys

PATH_DATA = "C:/Users/Benjamin/Desktop/Kaggle/osic-pulmonary-fibrosis-progression/"
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


def get_3d_scan(id):
    """Return a 3d matrix of the different slices (ct scans) of a patient, 
    the list of slice heights and widths"""
    path_data = get_path_id(id)
    filelist = get_scans_from_id(id)
    slice_agg, spacing, thickness = [], 0., 0.
    try:
        for file in filelist:
            data = pydicom.dcmread(f"{path_data}/{file}")
            slice_agg.append(data.pixel_array)
            spacing = data.PixelSpacing
            thickness = float(data.SliceThickness)
        return np.array(slice_agg), spacing, thickness
    except:
        print(f"Error when creating 3d scan of patient {id}")
        print(sys.exc_info()[0])


def get_data_patient(id_patient=None, indice=None):
    """Returns the 3d scan array of a patient,
    as a 128*128*128 array, values between 0 and 1"""
    
    if id_patient is None:
        id_patient = get_id_folders(indice)
    matrix, spacing, thickness = get_3d_scan(id_patient) 
    nb_slice, nb_row, nb_col = np.shape(matrix)
    
    #Resizing factors
    fx = spacing[0]/PIXEL_SPACING
    fy = spacing[1]/PIXEL_SPACING
    fz = thickness/THICKNESS
    
    #cut 128/fx x128/fy x128/fz 
    z,x,y = matrix.shape
    startx = max(0, (x - SCAN_SIZE[1]/fx)//2)
    starty = max(0, (y - SCAN_SIZE[2]/fy)//2)
    startz = max(0, (z - SCAN_SIZE[0]/fz)//2)

    resized_mat = matrix[int(startz):int(startz+SCAN_SIZE[0]//fz),
                         int(startx):int(startx+SCAN_SIZE[1]//fx),
                         int(starty):int(starty+SCAN_SIZE[2]//fy)]
    resized_mat = zoom(resized_mat, (fz, fx, fy))
        
    # #Add padding based on size
    z,x,y = resized_mat.shape
    z1 = (SCAN_SIZE[0] - z)//2
    z2 = (SCAN_SIZE[0]- z + 1)//2
    y1 = (SCAN_SIZE[2] - y)//2
    y2 = (SCAN_SIZE[2] - y + 1)//2
    x1 = (SCAN_SIZE[1] - x)//2
    x2 = (SCAN_SIZE[1] - x + 1)//2
    
    processed_mat=np.pad(resized_mat, ((z1, z2), (x1, x2), (y1, y2)), 'constant') #Always size SCAN_SIZE    
    min_matrix = np.min(processed_mat) #Normalization
    return (processed_mat-min_matrix)/(np.max(processed_mat)-min_matrix)
    

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
    # Transformation pour etre des lois normales TODO and TRY
    # from sklearn.preprocessing import PowerTransformer
    # yj = PowerTransformer(method='yeo-johnson')
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
    normalized = preprocessing_data(raw_data)
    return(normalized)