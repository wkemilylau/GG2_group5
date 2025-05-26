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

p= ct_phantom(material.name, 256, 3, metal=None) #metal=None default goes to 'Soft Tissue'
photons = source.photon('100kVp, 2mm Al')
s = photons * mas * (scale**2)
y = scan_and_reconstruct(s, material, p, 0.01, 256) #256x256


plt.imshow(y, cmap='gray', interpolation='bilinear', vmin=-1200)
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()

plt.hist(y, bins=100)
plt.yscale('log')
plt.savefig('./results/hist_plot_trial.png')
plt.close()
