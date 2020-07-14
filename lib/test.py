# -*- coding: utf-8 -*-

import tools
import numpy as np

pixels, slice_location, slice_thick = tools.get_random_scan()

print("Before normalization")
print(f"Shape: {np.shape(pixels)}, max {np.max(pixels)}, min {np.min(pixels)}")

import matplotlib.pyplot as plt
normalized = tools.normalize_scan(pixels)

print("After normalization")
print(f"Shape: {np.shape(normalized)}, max {np.max(normalized)}, min {np.min(normalized)}")

plt.imshow(normalized, cmap=plt.cm.bone)
