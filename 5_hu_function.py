from ct_scan import *
from ct_calibrate import *
from ct_lib import *
from ramp_filter import *
from back_project import *
from hu import *
import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from scan_and_reconstruct import *
from create_dicom import *
import matplotlib.pyplot as plt
from ramp_filter import *
from back_project import *
from hu import *
from attenuate import *

alpha=0.001
mas=10000
scale = 0.01

# create object instances
material = Material()
source = Source()

phantom = ct_phantom(material.name, 256, 1) #creates 256x256 image
photons = source.photon('100kVp, 2mm Al')
photons_total = photons * mas * (scale**2)

def hu(photons_total, material, reconstruction, scale):
    n = 256
    scale = 0.01  # cm per pixel
    names = ['Air', 'Adipose', 'Soft Tissue', 'Breast Tissue', 'Water']
    idx_water = names.index('Water')


    # water phantom
    phantom = np.full((n, n), idx_water, dtype=int)
    sinogram = ct_scan(photons_total, material, phantom, scale=scale, angles=256) # Generate sinogram of water
    attenuation_sinogram = ct_calibrate(photons_total, material, sinogram, scale)

    # μ of water is mean of uniform region (use center strip)
    mu_water = np.mean(attenuation_sinogram[:, n//2])
    print(f"μ_water ≈ {mu_water:.3f} cm⁻¹")

    hu_image = 1000 * (reconstruction - mu_water) / mu_water
    return hu_image


###################
sinogram = ct_scan(photons_total, material, phantom, scale=0.01, angles=256)

# Calibrate sinogram to attenuation values
calibrated = ct_calibrate(photons_total, material, sinogram, scale)

# Filter the calibrated sinogram with Ram-Lak filter (raised cosine alpha)
filtered = ramp_filter(calibrated, scale, alpha)

# Reconstruct image by back-projecting the filtered sinogram
reconstruction = back_project(filtered, skip=1)

hu_image = hu(photons_total, material, reconstruction, scale=0.01)
# clip for DICOM-storage
#hu_image = np.clip(hu_image, -1024, 3072)


plt.imshow(hu_image, cmap='gray', interpolation='nearest')
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()


#####################
n = 256
scale = 0.01  # cm per pixel
names = ['Air', 'Adipose', 'Soft Tissue', 'Breast Tissue', 'Water']
idx_water = names.index('Water')
idx_air = names.index('Air')

# water phantom
phantom = np.full((n, n), idx_water)
sinogram = ct_scan(photons_total, material, phantom, scale=scale, angles=256) # Generate sinogram of water
attenuation_sinogram_water = ct_calibrate(photons_total, material, sinogram, scale)

# air phantom
phantom = np.full((n, n), idx_air)
sinogram = ct_scan(photons_total, material, phantom, scale=scale, angles=256) # Generate sinogram of air
attenuation_sinogram_air = ct_calibrate(photons_total, material, sinogram, scale)

# μ of water is mean of uniform region (use center strip)
mu_water = np.mean(attenuation_sinogram_water[:, n//2])
mu_air = np.mean(attenuation_sinogram_air[:, n//2])
print(f"μ_water = {mu_water:.3f} cm⁻¹")
print(f"μ_air = {mu_air:.3f} cm⁻¹")



#====================================

plt.imshow(hu_image, cmap='gray')
plt.colorbar(label='HU')

# Annotate HU in a grid pattern
step = 32  # adjust to control density of labels
for y in range(0, hu_image.shape[0], step):
    for x in range(0, hu_image.shape[1], step):
        val = hu_image[y, x]
        plt.text(x, y, f"{int(val)}", color='red', fontsize=6, ha='center', va='center')

plt.title('HU Image with value labels')
plt.gca().invert_yaxis()
plt.show()