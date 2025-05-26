import math
import numpy as np
import numpy.matlib
from scipy.fft import fft, ifft, fftfreq

def ramp_filter(sinogram, scale, alpha=0.001):
	""" Ram-Lak filter with raised-cosine for CT reconstruction

	fs = ramp_filter(sinogram, scale) filters the input in sinogram (angles x samples)
	using a Ram-Lak filter.

	fs = ramp_filter(sinogram, scale, alpha) can be used to modify the Ram-Lak filter by a
	cosine raised to the power given by alpha."""

	# get input dimensions
	angles = sinogram.shape[0]
	n = sinogram.shape[1]

	#Set up filter to be at least twice as long as input
	m = np.ceil(np.log(2*n-1) / np.log(2))
	m = int(2 ** m)

	# 2d FT sinogram to freq domain
	freq = fftfreq(m, scale)
	omega = 2 * np.pi * freq
	omega_max = np.max(np.abs(omega))
      
	# Define the filter in frequency domain
	filter_vals = np.abs(omega) / (2 * np.pi)
	filter_vals[0] = filter_vals[1]/6
	with np.errstate(invalid='ignore'):  # avoid warnings for invalid values
		cos_term = np.cos((omega / omega_max) * (np.pi / 2))
		cos_term[np.abs(omega) > omega_max] = 0  # zero out-of-band
		filter_vals *= cos_term ** alpha
		filter_vals[np.isnan(filter_vals)] = 0  # handle NaNs at ω = 0

	# apply filter to all angles
	filtered_sinogram = np.zeros_like(sinogram)
	for i in range(angles):
		projection = sinogram[i]
		projection_padded = np.zeros(m)
		projection_padded[:n] = projection  # Zero-pad to match filter size
		projection_fft = fft(projection_padded)
		filtered_fft = projection_fft * filter_vals
		filtered = np.real(ifft(filtered_fft))
		filtered_sinogram[i] = filtered[:n]  # Crop back to original size


	print('Ramp filtering')
	
	return filtered_sinogram
