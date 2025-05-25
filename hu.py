import numpy as np
from attenuate import *
from ct_calibrate import *
from ct_detect import *
import ct_scan import *

'''
def hu(p, material, reconstruction, scale):
	""" convert CT reconstruction output to Hounsfield Units
	calibrated = hu(p, material, reconstruction, scale) converts the reconstruction into Hounsfield
	Units, using the material coefficients, photon energy p and scale given."""

	# use water to calibrate

	# put this through the same calibration process as the normal CT data

	# use result to convert to hounsfield units
	# limit minimum to -1024, which is normal for CT data.

	return reconstruction
'''

def hu(photons_total, material, reconstruction, scale):
    n = 256
    #scale = 0.01  # cm per pixel
    names = ['Air', 'Adipose', 'Soft Tissue', 'Breast Tissue', 'Water']
    idx_water = names.index('Water')
    idx_air = names.index('Air')

    # water phantom
    phantom = np.full((n, n), idx_water, dtype=int)
    sinogram = ct_scan(photons_total, material, phantom, scale=scale, angles=256) # Generate sinogram of water
    attenuation_sinogram = ct_calibrate(photons_total, material, sinogram, scale)

    # μ of water is mean of uniform region (use center strip)
    mu_water = np.mean(attenuation_sinogram[:, n//2])

    print(f"μ_water ≈ {mu_water:.3f} cm⁻¹")

    hu_image = 1000 * (reconstruction - mu_water) / mu_water
    return hu_image
