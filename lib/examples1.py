# -*- coding: utf-8 -*-

import tools
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Display the list of CT scans of a patient
# =============================================================================

id = "ID00007637202177411956430"
# scan_filenames = tools.get_scans_from_id(id)
# print(f"List of CT scans found for id : {id}")
# print(scan_filenames)

# =============================================================================
# Display the content of one dcm file
# =============================================================================

# path_data = tools.get_path_id(id)
# dataset = pydicom.dcmread(f"{path_data}/{scan_filenames[0]}")
# print("Content of the DCM file")
# print(dataset)

# =============================================================================
# Aggregates the CT scans into a numpy 3d array
# =============================================================================
# matrix, heights, widths = tools.get_3d_scan(id, normalized=False)

# =============================================================================
# Annimation of the 3d matrix slice by slice
# =============================================================================
# print("Scroll to animate")
# tools.multi_slice_viewer(matrix)
# print(heights)
# print(widths)

# =============================================================================
# 
# =============================================================================

data = tools.get_specific_scan("ID00086637202203494931510", 4)
print(data)
plt.imshow(data.pixel_array)

# =============================================================================
# Get random ct scan
# =============================================================================
 
pixels, spacing, slice_thick = tools.get_random_scan()

# print("Before normalization")
# print(f"Shape: {np.shape(pixels)}, max {np.max(pixels)}, min {np.min(pixels)}")

# import matplotlib.pyplot as plt
# normalized = tools.normalize_scan(pixels)

# print("After normalization")
# print(f"Shape: {np.shape(normalized)}, max {np.max(normalized)}, min {np.min(normalized)}")

# plt.imshow(normalized, cmap=plt.cm.bone)

