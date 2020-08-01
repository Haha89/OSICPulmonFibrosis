# -*- coding: utf-8 -*-

import tools
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# Display the list of CT scans of a patient
# =============================================================================

id = "ID00026637202179561894768"
scan_filenames = tools.get_scans_from_id(id)
print(f"List of CT scans found for id : {id}")
print(scan_filenames)

# =============================================================================
# Display the content of one dcm file
# =============================================================================

path_data = tools.get_path_id(id)
dataset = pydicom.dcmread(f"{path_data}/{scan_filenames[0]}")
print("Content of the DCM file")
print(dataset)

# =============================================================================
# Get the numpy 3d array already preprocessed
# =============================================================================
matrix = tools.get_3d_scan(id)

# =============================================================================
# Annimation of the 3d matrix slice by slice (run from console)
# =============================================================================
print("Scroll to animate")
tools.multi_slice_viewer(matrix)

# =============================================================================
# Get specific scan
# =============================================================================
data = tools.get_specific_scan("ID00086637202203494931510", 4)
plt.imshow(data.pixel_array)
