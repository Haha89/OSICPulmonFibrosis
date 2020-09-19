# -*- coding: utf-8 -*-

"""Set of many functions used throughout the different scripts"""

from os import listdir, path
from random import sample
from math import sqrt
import torch
import numpy as np
import pandas as pd
from pickle import dump, load

PATH_DATA =  "../data/" #"../input/osic-pulmonary-fibrosis-progression/" #
PIXEL_SPACING = 0.8
THICKNESS = 1
SCAN_SIZE = [128, 128, 128]
OFFSET_WEEKS = 5
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
MAP_SMOKE = {"Ex-smoker":.5, "Currently smokes":1, "Never smoked":0}

def get_id_folders(indice, train=True, path_folder=PATH_DATA):
    """Return the ID of the patient for a specific index."""
    folder = "train/" if train else "test/"
    try:
        return listdir(path_folder + folder)[indice]
    except:
        print(f"Error for indice {indice}")
        return listdir(path_folder + folder)[0]


def get_path_id(id_patient, train=True, path_folder=PATH_DATA):
    """Returns the path of the folder containing the patient ID CT scans.
    Notifies if the path is not found."""
    folder = "train/" if train else "test/"
    path_folder_ = path_folder + folder + id_patient
    if path.isdir(path_folder_):
        return path_folder_
    print(f"Could not find the folder for patient: {id_patient}")
    return None


def get_scans_from_id(id_patient, train=True):
    """Returns an ordered list of CT scans filenames from the id_patient."""
    path_folder = get_path_id(id_patient, train)
    if path_folder:
        return sorted(listdir(path_folder), key=lambda f: int(f.split(".")[0]))
    return []


def get_3d_scan(id_patient, path_folder=PATH_DATA):
    """Load and returns the 3d array of id_patient."""
    return np.load(f"{path_folder}scans/{id_patient}.npy", allow_pickle=True)


def unormalize_fvc(data):
    """Return the min and max FVC from dataset."""
    return(data["FVC"].min(), data["FVC"].max())


def preprocessing_data(data, train=True):
    """Preprocess the csv file, add one hot encoder and normalization between [0,1]."""
    
    data["Percent"] = data["Percent"]/100.
    data["SmokeNum"] = data['SmokingStatus'].map(MAP_SMOKE)
    data["Sex_Male"] = data['Sex'].map({"Male": 1, "Female": 0})
    if train:
        #Creation of dict containing min, max, mean, std of columns
        dict_postpro = {}
        for col in ["FVC", "Age"]:
            dict_postpro[col] = {"min": data[col].min(), "max": data[col].max()}
            data[col] = (data[col] - data[col].min())/(data[col].max() - data[col].min())
        
        with open('minmax.pickle', 'wb') as file_save:
            dump(dict_postpro, file_save) #, protocol=pickle.HIGHEST_PROTOCOL
            
    else: #Testing, loads the data from the existing pickle file and normalizes FVC, Age
        with open('minmax.pickle', 'rb') as file_save:
            dictio = load(file_save)

        for col in ["FVC", "Age"]:
            data[col] = (data[col] - dictio[col]["min"])/(dictio[col]["max"] - dictio[col]["min"])
    return data


def filter_data(data, id_patient=None, indice=None, path_folder=PATH_DATA):
    """Return the data only for the id_patient."""
    if id_patient is None:
        id_patient = get_id_folders(indice, path_folder=path_folder)
        
    filtered_data = data[data.Patient == id_patient]
    week_val = filtered_data.Weeks.values
    
    fvc = torch.zeros((140, 1)) #Avant (140)
    percent = torch.zeros((140, 1)) #Avant (140)
    weeks = torch.zeros((140))
    misc = torch.zeros((140, 3))
    ranger = torch.zeros((140))
    for i, week in enumerate(week_val):
        fvc[week + OFFSET_WEEKS] = filtered_data.FVC.values[i]
        percent[week + OFFSET_WEEKS] = filtered_data.Percent.values[i]
        weeks[week + OFFSET_WEEKS] = week # + OFFSET_WEEKS Alex 19/9
        ranger[week + OFFSET_WEEKS] = 1
    misc[:, 0] = torch.tensor(filtered_data.Age.values)[0]
    misc[:, 1] = torch.tensor(filtered_data.Sex_Male.values)[0]
    misc[:, 2] = torch.tensor(filtered_data.SmokeNum.values)[0]
    return misc, fvc, percent, weeks, ranger


def get_data(train=True, path_folder=PATH_DATA):
    """Return the content proprocessed on the train.csv file (containing patient data)."""
    file = "train" if train else "test"
    raw_data = pd.read_csv(f"{path_folder}{file}.csv")
    return preprocessing_data(raw_data)


def make_folds(nb_folds, path_folder = PATH_DATA):
    """Fonction qui permet de labeliser nos entrees de 0 a
    nb_folds pour k-fold cross validation """

    batch_size = len(listdir(path_folder + 'train/'))
    subfolders = [i for i in range(batch_size)]
    #Chaque donnÃ©e Ã  un label
    fold_label = np.zeros(batch_size, dtype=int)
    items_per_fold = batch_size//nb_folds
    for fold in range(nb_folds-1):
        sample_id = sample(subfolders, items_per_fold)
        #On les assigne au label considÃ©rÃ©
        for smpl in sample_id:
            fold_label[smpl] = fold + 1
            #On retire les indices tirÃ©s de la liste
        subfolders = [index for index in subfolders if fold_label[index] == 0]
    #Comme la division nombre d'Ã©lements/nb_folds ne tombe pas forcÃ©ment juste,
    #on assigne tous les derniers Ã©lements au label restant
    for smpl in subfolders:
        fold_label[smpl] = nb_folds
    # Pour une input de taille batch x taille seq max x nb_features, fold_label est de taille batch
    # Et fold_label[k] contient le label appartenant Ã  [0, nb_fold-A] correspondant Ã  input[k,:,:]
    return fold_label - 1


def train_test_indices(fold_label, nb_fold):
    """Pour un fold donne, cree le testing set et training
    set en fonction de fold_label """
    indices_train_1 = np.where((fold_label != nb_fold))[0]
    indices_train = np.array([x for x in indices_train_1 if fold_label[x] <5])
    
    indices_test_1 = np.where((fold_label == nb_fold))[0]
    indices_test = np.array([x for x in indices_test_1 if fold_label[x] <5])
    return (indices_train, indices_test)


def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, mask):
    """
    Calculates the modified Laplace Log Likelihood score for this competition.
    """
    std_min = torch.tensor([70.]).cuda()
    delta_max = torch.tensor([1000.]).cuda()
    std_clipped = torch.max(confidence, std_min)
    delta = torch.min(torch.abs(actual_fvc - predicted_fvc), delta_max)
    metric = (- sqrt(2) * delta / std_clipped - torch.log(sqrt(2) * std_clipped))*mask
    metric = metric.sum()/mask.sum()
    return -metric


def ode_laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, epoch, epoch_max):
    """
    Calculates the modified Laplace Log Likelihood score for this competition.
    """
    if epoch > epoch_max : 
        std_clipped = torch.max(confidence, torch.tensor([70.]).to(DEVICE))
        delta = torch.min(torch.abs(actual_fvc - predicted_fvc), torch.tensor([1000.]).to(DEVICE))
    else :
        std_clipped = torch.abs(confidence)
        delta = torch.abs(actual_fvc - predicted_fvc)
        
    metric = (- sqrt(2) * delta / std_clipped - torch.log(sqrt(2) * std_clipped))
    return -metric.mean()
