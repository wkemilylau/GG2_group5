from ct_scan import *
from ct_calibrate import *
from ct_lib import *
from ramp_filter import *
from back_project import *
from hu import *

def scan_and_reconstruct(photons, material, phantom, scale, angles, mas=10000, alpha=0.001):

    """ Simulation of the CT scanning process
    reconstruction = scan_and_reconstruct(photons, material, phantom, scale, angles, mas, alpha)
    takes the phantom data in phantom (samples x samples), scans it using the
    source photons and material information given, as well as the scale (in cm),
    number of angles, time-current product in mas, and raised-cosine power
    alpha for filtering. The output reconstruction is the same size as phantom."""

    # Convert photons per (mas,cm^2) to actual photons count (scale by mas and pixel size)
    photons_total = photons * mas * (scale**2)

    # Generate sinogram by scanning phantom
    sinogram = ct_scan(photons_total, material, phantom, scale, angles)

    # Calibrate sinogram to attenuation values
    calibrated = ct_calibrate(photons_total, material, sinogram, scale)

    # Filter the calibrated sinogram with Ram-Lak filter (raised cosine alpha)
    filtered = ramp_filter(calibrated, scale, alpha)

    # Reconstruct image by back-projecting the filtered sinogram
    reconstruction = back_project(filtered, skip=1)
    print(reconstruction [0])

    # Convert reconstructed linear attenuation coefficients to Hounsfield Units (HU)
    #hu_image = hu(photons_total, material, reconstruction, scale)

    return reconstruction

