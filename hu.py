import numpy as np
from attenuate import *
from ct_calibrate import *

def hu(p, material, reconstruction, scale):
    """Convert CT reconstruction output to Hounsfield Units (HU).
    
    calibrated = hu(p, material, reconstruction, scale) converts the reconstruction into 
    Hounsfield Units, using the material coefficients, photon energy p, and the scale given.
    """

    #Creating a 1x1 phantom representing a 1 cm path through either water or air
    water_phantom = np.ones((1, 1))

    #attenuation simulation for 1cm of both air and water
    I_water = attenuate(p, material, water_phantom, 'Water', scale=scale)
    I_air = attenuate(p, material, water_phantom, 'Air', scale=scale)

    # effective mu values obtained by calibrating the attenuated intensities
    mu_water = ct_calibrate(p, material, I_water, scale)[0, 0]
    mu_air = ct_calibrate(p, material, I_air, scale)[0, 0]

    #Hounsfield Units conversion
    hu_image = 1000 * (reconstruction - mu_air) / (mu_water - mu_air)

    return hu_image
