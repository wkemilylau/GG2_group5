import numpy as np
from ct_detect import ct_detect

import numpy as np

from attenuate import *

import numpy as np 

def beam_hardening_calibration_polyfit(photons, material, scale, degree=3, num_points=50):
    """
    Uses ct_detect to simulate beam hardening for calibration and fits a polynomial
    mapping log attenuation to true material thickness (for water).
    """

    # Get water attenuation coefficients
    water_coeffs = material.coeff('Water')  

    # Create thickness samples
    thicknesses = np.linspace(0, 20, num_points)

    # Compute I0 once (no attenuation)
    I0 = np.sum(photons)

    # Simulate attenuation for each thickness using ct_detect
    depth_array = np.array(thicknesses).reshape(1, -1)  
    coeffs = np.array(water_coeffs).reshape(1, -1)     
    photons_detected = ct_detect(photons, coeffs, depth_array)  

    # Compute linear attenuation
    pw = -np.log(np.clip(photons_detected, 1e-12, None) / I0)

    # Filter for valid (physical) range
    valid = (pw > 0) & (pw < 20)
    if np.sum(valid) < degree + 1:
        raise ValueError("Not enough valid points for polynomial fit")

    # Fit polynomial: pw → thickness
    return np.polyfit(pw[valid], thicknesses[valid], degree)



def ct_calibrate(photons, material, sinogram, scale):
    """
    Convert CT detection sinogram to linearised attenuation using beam hardening correction.

    This function applies:
    1. Air attenuation correction using Beer–Lambert law.
    2. Beam hardening correction using a polynomial fit of water attenuation.

    Args:
        photons (np.ndarray): Source photon distribution (E,)
        material (Material): Material class with .coeff() method
        sinogram (np.ndarray): Raw CT sinogram (angles x samples)
        scale (float): Pixel size in cm

    Returns:
        p_corrected (np.ndarray): Calibrated sinogram (angles x samples)
    """
    # Get image size (needed for air thickness estimation)
    n = sinogram.shape[1]

    # Estimate air-only detection using known attenuation through 2*image width of air
    air_mu = material.coeff('Air')                # shape (E,)
    air_thickness = 2 * n * scale                 # distance from source to detector in air
    I_air = photons * np.exp(-air_mu * air_thickness)  # shape (E,)
    I_air_total = np.sum(I_air)                   # total detected intensity through air

    # Stabilize sinogram values to avoid log(0)
    clipped_sinogram = np.clip(sinogram, 1e-12, None)

    # Apply log transform to compute line integrals
    p = -np.log(clipped_sinogram / I_air_total)

    #beam hardening
    poly_coeffs = beam_hardening_calibration_polyfit(photons, material, scale)
    return np.polyval(poly_coeffs, p)


