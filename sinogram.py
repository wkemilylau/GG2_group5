import matplotlib.pyplot as plt
import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from create_dicom import *

# create object instances
material = Material()
source = Source()

# Create a phantom
phantom = ct_phantom(material.name, 256, 1) #creates 256x256 image
#print(material.name)
draw(phantom)

# Generate source photons
photons = source.photon('100kVp, 2mm Al')

# Generate sinogram using ct_scan
sinogram = ct_scan(photons, material, phantom, scale=0.01, angles=60)
#print(sinogram.shape) = (angles, detector pixels)


plt.imshow(sinogram, cmap='gray', interpolation='nearest')
plt.gca().invert_yaxis()
plt.title("Sinogram")
plt.xlabel("Detector pixel locations")
plt.ylabel("Projection angle")
plt.colorbar()
plt.show()