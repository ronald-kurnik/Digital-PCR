import os
import cv2
import numpy as np
from skimage import morphology, measure
from scipy import ndimage
import matplotlib.pyplot as plt

# Step 1: Import the image (grayscale)
im = cv2.imread("dPCR.tiff", cv2.IMREAD_GRAYSCALE)

# Step 2: Binarize and remove small components 
# Binarize using Otsu's thresholding
_, im2 = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Remove small objects (default is 20 pixels in Mathematica; adjust if needed)
im2 = morphology.remove_small_objects(im2.astype(bool), min_size=10)

# Step 3: Select components with area between 10 and 50, and not touching the border
# Label connected components
labeled, num_features = ndimage.label(im2)
props = measure.regionprops(labeled)

# Filter components: 10 < area < 50 and not touching border
def touches_border(prop):
    bbox = prop.bbox  # (min_row, min_col, max_row, max_col)
    return (bbox[0] == 0 or bbox[1] == 0 or 
            bbox[2] == im.shape[0] or bbox[3] == im.shape[1])

valid_mask = np.zeros_like(labeled, dtype=bool)
for prop in props:
    if 10 < prop.area < 50 and not touches_border(prop):
        valid_mask[labeled == prop.label] = True

im3 = valid_mask

# Step 4 & 5: Count components and get the label of the last one (by sorted order)
# MorphologicalComponents[im3] -> labeling of binary image

labeled_final = measure.label(im3)
counts = np.bincount(labeled_final.ravel())[1:]  # exclude background
cts = len(counts)  # This gives the count of components

print(f"Number of selected components: {cts}")

plt.imshow(im2, cmap='gray')
plt.axis('off')
plt.title("Number of Selected Components = " + str(cts))
plt.show()