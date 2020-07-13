# -*- coding: utf-8 -*-

import tools
import pydicom

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
matrix, heights, widths = tools.get_3d_scan(id)

# =============================================================================
# Annimation of the 3d matrix slice by slice
# =============================================================================
print("Scroll to animate")
tools.multi_slice_viewer(matrix)
print(heights)
print(widths)


# =============================================================================
# Get random ct scan
# =============================================================================



# for i in range(500):
#     arr, he, wid = tools.get_random_scan()
    
    
# print(tools.get_specific_scan("ID00128637202219474716089", 9))