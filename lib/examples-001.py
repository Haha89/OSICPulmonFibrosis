# -*- coding: utf-8 -*-

"""Examples to show what tools can do"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pydicom
import tools
# print(tools)

# =============================================================================
# Display the list of CT scans of a patient
# =============================================================================

ID_PATIENT = "ID00423637202312137826377"
scan_filenames = tools.get_scans_from_id(ID_PATIENT)
# print(f"List of CT scans found for id : {id}")
# print(scan_filenames)

# # =============================================================================
# # Display the content of one dcm file
# # =============================================================================

# PATH_DATA = tools.get_path_id(ID_PATIENT)
# DATA_FILE = pydicom.dcmread(f"{PATH_DATA}/{scan_filenames[0]}")
# print("Content of the DCM file")
# print(DATA_FILE)

# =============================================================================
# # Get the numpy 3d array already preprocessed
# # =============================================================================
MATRIX = tools.get_3d_scan(ID_PATIENT)

# # =============================================================================
# # Annimation of the 3d matrix slice by slice (run from console)
# # =============================================================================
print("Scroll to animate")
tools.multi_slice_viewer(MATRIX)
