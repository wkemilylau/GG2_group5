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

p = source.photon('100kVp, 2mm Al')
coeffs = material.coeff('Water')
depth = np.arange(0, 10.1, 0.1)
mas = 1
y = ct_detect(p, coeffs, depth, mas)

plt.plot(np.log(y))
plt.xlabel('Depth (cm)')
plt.ylabel('Log detector signal (residual energy)')
plt.grid(True)
plt.show()