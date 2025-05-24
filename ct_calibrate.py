import numpy as np
import scipy
from scipy import interpolate

def ct_calibrate(photons, material, sinogram, scale):

	""" ct_calibrate convert CT detections to linearised attenuation
	sinogram = ct_calibrate(photons, material, sinogram, scale) takes the CT detection sinogram
	in x (angles x samples) and returns a linear attenuation sinogram
	(angles x samples). photons is the source energy distribution, material is the
	material structure containing names, linear attenuation coefficients and
	energies in mev, and scale is the size of each pixel in x, in cm."""

	# Get dimensions and work out detection for just air of twice the side
	# length (has to be the same as in ct_scan.py)
	n = sinogram.shape[1]

	#ct_calibrate additions
	I0_E = np.sum(photons, axis=0)
	I0_total = np.sum(I0_E)

	# air attenuation correction
	air_mu = material.coeff('Air')      # mu for air (array of same shape as photons)
	air_thickness = 2 * n * scale       # distance from source to detector through air

	# apply Beer–Lambert law for each energy: I = I0 * exp(-μ*d)
    # compute air-attenuated intensity
	I_air = photons * np.exp(-air_mu * air_thickness)
	I_air_total = np.sum(I_air)         # what detector would actually measure with just air

	clipped_sinogram = np.clip(sinogram, 1e-12, None) #prevents extremely small values in the sinogram, which would cause extremely large values of p, leading to instability.
	p = -np.log(clipped_sinogram / I_air_total)

	return p
