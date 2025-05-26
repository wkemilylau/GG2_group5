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
from scan_and_reconstruct import *

alpha=0.001
mas=10000
scale = 0.01

# create object instances
material = Material()
source = Source()

phantom = ct_phantom(material.name, 256, 3) #creates 256x256 image
photons = source.photon('100kVp, 2mm Al')
photons_total = photons * mas * (scale**2)

###################
sinogram = ct_scan(photons_total, material, phantom, scale=0.01, angles=256)

# Calibrate sinogram to attenuation values
calibrated = ct_calibrate(photons_total, material, sinogram, scale)

# Filter the calibrated sinogram with Ram-Lak filter (raised cosine alpha)
filtered = ramp_filter(calibrated, scale, alpha)

# Reconstruct image by back-projecting the filtered sinogram
reconstruction = back_project(filtered, skip=1)

#===============================

p= ct_phantom(material.name, 256, 3, metal=None) #metal=None default goes to 'Soft Tissue'
s = fake_source(material.mev, 120, method='ideal') #ideal source, (len=200), all zero excpet final energy
y = scan_and_reconstruct(s, material, p, 0.01, 256) #256x256


plt.imshow(y, cmap='gray', interpolation='nearest')
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()


'''
print(reconstruction)
plt.imshow(reconstruction, cmap='gray', interpolation='nearest')
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()
'''

y = scan_and_reconstruct(s, material, p, 0.1, 256) #256x256
