# -*- coding: utf-8 -*-

"""Set of many functions used throughout the different scripts"""

import numpy as np
import matplotlib.pyplot as plt
from pydicom import dcmread
from scipy.ndimage import zoom
from utils import get_path_id, get_scans_from_id
import scipy.ndimage as ndimage
from skimage import measure, segmentation

PATH_DATA = "../data/"
PIXEL_SPACING = 1
SPACING_Z = 3
SCAN_SIZE = [32, 256, 256] #z, x, y
clip_bounds = (-1000, 200)


def create_3d_scan(id_patient, train=True):
    """Return a 3d matrix of the different slices (ct scans) of a patient,
    the list of slice heights and widths. 
    Heterogeneity in the dataset requires to return 
     - the distance between SliceLocation when available;
     - otherwise SpacingBetweenSlices,
     - otherwise the SliceThickness"""
    
    path_data = get_path_id(id_patient, train)
    filelist = get_scans_from_id(id_patient, train)
    if len(filelist)==0: print("No scans for ", id_patient)
    slice_agg, spacing, y_pos = [], 0., []
    
    for file in filelist:
        data = dcmread(f"{path_data}/{file}")
        slice_agg.append(data.pixel_array)
        spacing = data.PixelSpacing
        if "SliceLocation" in data:
            y_pos.append(data.SliceLocation)
            
    if len(y_pos)>1:   
        space_z = abs(float(y_pos[1])-float(y_pos[0]))
        return np.array(slice_agg), spacing, 1 if space_z == 0 else space_z, data
    
    else:
        if "SpacingBetweenSlices" in data:
            return np.array(slice_agg), spacing, data.SpacingBetweenSlices, data
        else:
            return np.array(slice_agg), spacing, data.SliceThickness, data


def rescale(sample, bounds=(-1000, -200)):
    image, data = sample['image'], sample['metadata']
    

    if data.PatientID == "ID00132637202222178761324":
        intercept = 2048
    elif data.PatientID == "ID00128637202219474716089":
        intercept = 1024  
    else:
        if "RescaleIntercept" in data:
            intercept = data.RescaleIntercept
        else:
            if data.Manufacturer == "TOSHIBA":
                intercept = 2048
            else:
                intercept = 1024
            
    slope = data.RescaleSlope
    image = (image * slope + intercept).astype(np.int16)
    image[image < min(bounds)] = min(bounds)
    image[image > max(bounds)] = max(bounds)

    return {'image': image, 'metadata': data}

def bounding_box(img3d: np.array):
    mid_img = img3d[int(img3d.shape[0] / 2)]
    same_first_row = (mid_img[0, :] == mid_img[0, 0]).all()
    same_first_col = (mid_img[:, 0] == mid_img[0, 0]).all()
    return same_first_col and same_first_row


def crop(sample):
    image, data = sample['image'], sample['metadata']
    if not bounding_box(image):
        return sample

    mid_img = image[int(image.shape[0] / 2)]
    r_min, r_max = None, None
    c_min, c_max = None, None
    for row in range(mid_img.shape[0]):
        if not (mid_img[row, :] == mid_img[0, 0]).all() and r_min is None:
            r_min = row
        if (mid_img[row, :] == mid_img[0, 0]).all() and r_max is None \
                and r_min is not None:
            r_max = row
            break

    for col in range(mid_img.shape[1]):
        if not (mid_img[:, col] == mid_img[0, 0]).all() and c_min is None:
            c_min = col
        if (mid_img[:, col] == mid_img[0, 0]).all() and c_max is None \
                and c_min is not None:
            c_max = col
            break

    image = image[:, r_min:r_max, c_min:c_max]
    return {'image': image, 'metadata': data}
  
 
def analyse(sample, min_hu, iterations):
    threshold = -900 if sample['metadata'].PatientID == "ID00229637202260254240583" else -700
    stack = [seperate_lungs(scan, min_hu, iterations, threshold) for scan in sample['image']]
    return {'image': np.stack(stack), 'metadata': sample['metadata']}


def seperate_lungs(image, min_hu, iterations, threshold =-700):
    """
    Segments lungs using various techniques.
    
    Parameters: image (Scan image), iterations (more iterations, more accurate mask)
    
    Returns: 
        - Segmented Lung
    """
    
    h, w = image.shape[0], image.shape[1]
    
    marker_internal, marker_external, marker_watershed = generate_markers(image, threshold)

    # Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    
    if np.max(sobel_gradient) == 0:
        return image
        
    sobel_gradient *= 255.0 / np.max(sobel_gradient)
    watershed = segmentation.watershed(sobel_gradient, marker_watershed)
    outline = ndimage.morphological_gradient(watershed, size=(3,3))
    outline = outline.astype(bool)

    # Structuring element used for the filter
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, iterations)

    # Perform Black Top-hat filter
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)
    lungfilter = np.bitwise_or(marker_internal, outline)
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)
    segmented = np.where(lungfilter == 1, image, min_hu * np.ones((h, w)))
    return segmented

    
def generate_markers(image, threshold):
    h, w = image.shape[0], image.shape[1]

    marker_internal = image < threshold
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)

    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()

    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    marker_internal_labels[coordinates[0], coordinates[1]] = 0

    marker_internal = marker_internal_labels > 0

    # Creation of the External Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=20)
    marker_external = external_b ^ external_a

    # Creation of the Watershed Marker
    marker_watershed = np.zeros((h, w), dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128
    return marker_internal, marker_external, marker_watershed    


def resize(matrix, spacing, space_z):
    #Resizing factors
    fx = spacing[0]/PIXEL_SPACING
    fy = spacing[1]/PIXEL_SPACING
    fz = space_z/SPACING_Z
    
    #cut 128/fx x128/fy x128/fz
    z, x, y = matrix.shape
    startx = max(0, (x - SCAN_SIZE[1]/fx)//2)
    starty = max(0, (y - SCAN_SIZE[2]/fy)//2)
    startz = max(0, (z - SCAN_SIZE[0]/fz)//2)

    resized_mat = matrix[int(startz):int(startz+SCAN_SIZE[0]//fz),
                         int(startx):int(startx+SCAN_SIZE[1]//fx),
                         int(starty):int(starty+SCAN_SIZE[2]//fy)]
    resized_mat = zoom(resized_mat, (fz, fx, fy))
    
    #Add padding based on size
    z, x, y = resized_mat.shape
    z1 = (SCAN_SIZE[0] - z)//2
    z2 = (SCAN_SIZE[0]- z + 1)//2
    y1 = (SCAN_SIZE[2] - y)//2
    y2 = (SCAN_SIZE[2] - y + 1)//2
    x1 = (SCAN_SIZE[1] - x)//2
    x2 = (SCAN_SIZE[1] - x + 1)//2

    processed_mat = np.pad(resized_mat, ((z1, z2), (x1, x2), (y1, y2)), 'constant')
    return processed_mat


def process_3d_scan(id_patient, train=True):
    """Returns the 3d scan array of a patient,
    as a 32*256*256 array, values between 0 and 1"""
    
    matrix, spacing, space_z, metadata = create_3d_scan(id_patient, train)
    sample = {'image': matrix, 'metadata': metadata} 
    sample = crop(sample)
    
    sample['image'] = resize(sample['image'], spacing, space_z)
    sample_watershed = rescale(sample, bounds=clip_bounds)
    
    # multi_slice_viewer(sample_watershed['image'])
    processed_mat = analyse(sample_watershed, min(clip_bounds), 1)['image']
    
    if np.min(processed_mat) - np.max(processed_mat) == 0: #Empty mask, return not modyfied
        min_matrix = np.min(sample['image']) #Normalization
        sample['image'] = (sample['image'] - min_matrix)/(np.max(sample['image']) - min_matrix)
        return sample['image']
    else:
        min_matrix = np.min(processed_mat) #Normalization
        processed_mat = (processed_mat - min_matrix)/(np.max(processed_mat) - min_matrix)
        return processed_mat
   
    
    
def multi_slice_viewer(matrix_3d, title=None):
    """Visualization of the matrix slice by slice.Allegrement Stolen online."""

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
    if title: plt.title(title)
    plt.show()
    
def crop_slice(s):
    """Crop frames from slices, borders where only 0."""
    s_cropped = s[~np.all(s == 0, axis=1)]
    s_cropped = s_cropped[:, ~np.all(s_cropped == 0, axis=0)]
    return s_cropped