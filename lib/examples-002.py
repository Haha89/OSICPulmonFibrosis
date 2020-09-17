# -*- coding: utf-8 -*-

"""Preprocessing of the Train dataset, display of content of one patient"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_data, filter_data

PREPROC = get_data()
ID_PATIENT = "ID00026637202179561894768"
OTHER, FVC, PERCENT, WEEKS = filter_data(PREPROC, ID_PATIENT)

print(FVC)
print(OTHER)
print(PERCENT)
print(WEEKS)
