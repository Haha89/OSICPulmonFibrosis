# -*- coding: utf-8 -*-

"""
This case is useful for real data. Good examples are PhysioNet class (—dataset physionet)
 and PersonActivity class (--dataset activity).

To use Latent ODE on your dataset, I recommend formatting your data as a list of records,
 as described below. You can use the collate function for DataLoader from here

Data format
Since each time series has different length, we represent the dataset as a list of 
records: [record1, record2, record3, …]. Each record represents one time series 
(e.g. one patient in Physionet).

Each record has the following format:
(record_id, observation_times, values, mask, labels)

record_id: an id of this string
observation_times: a 1-dimensional numpy array containing T time values of
values: a (T, D) numpy array containing observed D-dimensional values at T time points
mask: a (T, D) tensor containing 1 where values were observed and 0 otherwise. 
        Useful if different dimensions are observed at different times. 
        If all dimensions are observed at the same time, fill the mask with ones.
labels: a list of labels for the current patient, if labels are available. Otherwise None.


Pipeline of the Physionet dataset
To use it on your dataset, you need to only replace step 1 to produce the list of records.

Physionet class loads the dataset from files (each patient is stored in its own file) and outputs the list of records (format described above) like so

Physionet class is called in lib/parse_dataset.py to get a list of records. List of records is then split into train/test here

Dataloader takes the list of records and collates them into batches like so Function that collates records into batches is here

During the training, the model calls data loader to get a new batch.
"""

from tools import get_data
import numpy as np

data = get_data()
data.Weeks -= data.Weeks.min()
data.Weeks = data.Weeks.astype(int)
observation_times = np.arange(0, 138, 1)
labels = [None]*len(observation_times)

def get_record(patient : str, data):
    sub_data = data[data.Patient == patient]
    values = np.zeros((1, len(observation_times)))
    mask = np.zeros((1, len(observation_times)))
    fvcs = sub_data.FVC.values
    
    for i, el in enumerate(sub_data.Weeks.values):
        mask[0, el] = 1  
        values[0, el] = fvcs[i]  
    return (patient, observation_times, values, mask, labels)
    
print(get_record("ID00007637202177411956430", data))
