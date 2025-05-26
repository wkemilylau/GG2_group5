import matplotlib.pyplot as plt
import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from create_dicom import *
from ct_detect import *

# create object instances
material = Material()
source = Source()

# Create a phantom
phantom = ct_phantom(material.name, 256, 3) #creates 256x256 image
#print(material.name)
#draw(phantom)

# Generate source photons
photons = source.photon('100kVp, 2mm Al')

# Generate sinogram using ct_scan
sinogram = ct_scan(photons, material, phantom, scale=0.01, angles=402)
#print(sinogram.shape) = (angles, detector pixels)


'''
#residual energy sinogram, ct_scan with different interpolation techniques
interpolations = ['nearest', 'bilinear', 'bicubic']
for interp in interpolations:
    plt.figure()
    plt.imshow(sinogram, cmap='gray', interpolation=interp)
    plt.gca().invert_yaxis()
    plt.xlabel("Detector pixel locations")
    plt.ylabel("Projection angle")
    plt.show()
'''
#========================
'''
#residual energy sinogram, for different scanning angles
angle_list = [50,190,500]

# Plot sinograms for each angle setting
for angles in angle_list:
    sinogram = ct_scan(photons, material, phantom, scale=0.01, angles=angles)

    plt.figure()
    plt.imshow(sinogram, cmap='gray', interpolation='bilinear')
    plt.gca().invert_yaxis()
    plt.xlabel("Detector pixel locations")
    plt.ylabel("Projection angle")
    plt.show()
'''

#========================================

#Figure
p = ct_phantom(material.name, 256, 3)
s = source.photon('100kVp, 3mm Al')
y = scan_and_reconstruct(s, material, p, 0.01, 300)

# Save results
save_draw(y, 'results', 'test_1_image')
save_draw(p, 'results', 'test_1_phantom')

#========================
'''
#residual energy sinogram
plt.imshow(sinogram, cmap='gray', interpolation='nearest')
plt.gca().invert_yaxis()
#plt.title("Residual energy sinogram")
plt.xlabel("Detector pixel locations")
plt.ylabel("Projection angle")
plt.colorbar()
plt.show()
'''

#======================
#Attenuation sinogram
scale = 0.1  # cm per pixel
attenuation_sinogram = ct_calibrate(photons, material, sinogram, scale)
plt.imshow(attenuation_sinogram, cmap='gray', interpolation='nearest')
plt.gca().invert_yaxis()
#plt.title("Attenuation sinogram")
plt.xlabel("Detector pixel locations")
plt.ylabel("Projection angle")
plt.colorbar()
plt.show()
