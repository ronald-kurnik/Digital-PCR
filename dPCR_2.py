import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, feature, measure
import pandas as pd

# Load grayscale image of one well
img = io.imread("dPCR.tiff")

# Detect droplets 
peaks = feature.peak_local_max(img, min_distance=10, threshold_abs=2000)

droplet_coords = peaks[:, [1, 0]]  # (x,y)

# Extract mean intensity in 9x9 ROI around each peak
amplitudes = []
for x, y in droplet_coords:
    roi = img[max(0,y-5):y+6, max(0,x-5):x+6]
    amplitudes.append(np.mean(roi))

# Subtract global background (optional)
bg = np.percentile(img, 5)
amplitudes = np.array(amplitudes) - bg
cnts = len(droplet_coords)

plt.figure(figsize=(8,5))
plt.scatter(np.arange(1,cnts+1),amplitudes, s=2, c='k')
plt.grid('visible')
plt.xlabel('Number of Cells')
plt.ylabel('Mean Fluorescence Amplitude')
plt.title('dPCR Amplitude Distribution')
plt.show()

# Save data
df = pd.DataFrame({'Amplitude': amplitudes})
df.to_csv('droplet_amplitudes.csv', index=False)
