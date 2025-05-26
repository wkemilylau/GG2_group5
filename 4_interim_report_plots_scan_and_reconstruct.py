import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
#from scan_and_reconstruct import *
from create_dicom import *
from attenuate import *

from ct_scan import *
from ct_calibrate import *
from ct_lib import *
from ramp_filter import *
from back_project import *
from hu import *

def scan_and_reconstruct_no_filter(photons, material, phantom, scale, angles, mas=10000, alpha=0.001):
    # Convert photons per (mas,cm^2) to actual photons count (scale by mas and pixel size)
    photons_total = photons * mas * (scale**2)
    sinogram = ct_scan(photons_total, material, phantom, scale, angles)
    calibrated = ct_calibrate(photons_total, material, sinogram, scale)

    # Reconstruct image by back-projecting the filtered sinogram
    reconstruction = back_project(calibrated, skip=1)

    return reconstruction


def scan_and_reconstruct_with_filter(photons, material, phantom, scale, angles, mas=10000, alpha=0.001):
    # Convert photons per (mas,cm^2) to actual photons count (scale by mas and pixel size)
    photons_total = photons * mas * (scale**2)
    sinogram = ct_scan(photons_total, material, phantom, scale, angles)
    calibrated = ct_calibrate(photons_total, material, sinogram, scale)

    filtered = ramp_filter(calibrated, scale, alpha)

    # Reconstruct image by back-projecting the filtered sinogram
    reconstruction = back_project(filtered, skip=1)

    return reconstruction


def save_image(img, folder, filename, title="Reconstructed Image"):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray', vmin=0.05)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{filename}.png"))
    plt.close()



# create object instances
material = Material()
source = Source()

p= ct_phantom(material.name, 256, 3, metal=None) #metal=None default goes to 'Soft Tissue'
save_draw(p, 'results', 'test_phantom')

s = fake_source(material.mev, 120, method='ideal') #ideal source, (len=200), all zero excpet final energy
y_no = scan_and_reconstruct_no_filter(s, material, p, 0.01, 256) #256x256
y_with = scan_and_reconstruct_with_filter(s, material, p, 0.01, 256) #256x256

save_plot(y_no[128,:], 'results', 'no_filter')
save_plot(y_with[128,:], 'results', 'with_filter')

# Save full 2D reconstructed images
save_image(y_no, 'results', 'recon_no_filter')
save_image(y_with, 'results', 'recon_with_filter')

plt.hist(y_with.flatten(), bins=100)
plt.yscale('log')
plt.savefig('./results/hist_plot_trial.png')
plt.close()


#==================
'''
import numpy as np
import matplotlib.pyplot as plt
import ramp_filter
import ct_lib

impulse = np.zeros((256, 256))
impulse[128][128] = 1.0

# Filter responses for different alpha values
filtered_response_1 = ramp_filter.ramp_filter(impulse, 0.1, 0.001)[128]
filtered_response_3 = ramp_filter.ramp_filter(impulse, 0.1, 1.0)[128]
filtered_response_4 = ramp_filter.ramp_filter(impulse, 0.1, 10.0)[128]

# Plotting
plt.plot(filtered_response_1, label='α = 0.001', linewidth=1)
plt.plot(filtered_response_3, label='α = 1.0', linewidth=1)
plt.plot(filtered_response_4, label='α = 10.0', linewidth=1)

# Add axis labels
plt.xlabel('Detector index (x)')
plt.ylabel('Filtered impulse response')

#plt.title('Ramp Filter Impulse Response (Row 128)')
plt.xlim([113, 143])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''

#########
