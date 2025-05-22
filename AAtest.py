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

phantom = ct_phantom(material.name, 256, 3) #creates 256x256 image
photons = source.photon('100kVp, 2mm Al')
photons_total = photons * mas * (scale**2)

def hu(p, material, reconstruction, scale):
    n = max(p.shape)
    depth = np.zeros((len(material.coeffs), n))

    coeff_water = material.coeff('Water')
    coeff_air = material.coeff('Air')

    I_water = attenuate(p, coeff_water, depth)
    I_air = attenuate(p, coeff_air, depth)

    sinogram_water = np.sum(I_water, axis=0).reshape(1, 1)
    sinogram_air = np.sum(I_air, axis=0).reshape(1, 1)

    mu_water = ct_calibrate(p, material, sinogram_water, scale)[0, 0]
    mu_air = ct_calibrate(p, material, sinogram_air, scale)[0, 0]

    denominator = mu_water - mu_air
    if abs(denominator) < 1e-8:
        denominator = 1e-8

    hu_image = 1000 * (reconstruction - mu_air) / denominator

    return hu_image

###################
filtered = ramp_filter(phantom, scale, alpha=0.001)

sinogram = ct_scan(photons_total, material, phantom, scale=0.01, angles=190)

# Calibrate sinogram to attenuation values
calibrated = ct_calibrate(photons_total, material, sinogram, scale)

# Filter the calibrated sinogram with Ram-Lak filter (raised cosine alpha)
filtered = ramp_filter(calibrated, scale, alpha)

# Reconstruct image by back-projecting the filtered sinogram
reconstruction = back_project(filtered, skip=1)


hu_image = hu(photons_total, material, reconstruction, scale)
plt.imshow(hu_image, cmap='gray', interpolation='nearest')
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()
