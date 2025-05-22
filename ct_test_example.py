
import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from create_dicom import *
from attenuate import *
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr


# create object instances
material = Material()
source = Source()



def check_images():
    """
    This test checks that the reconstructed image structurally and statistically matches that of the phantom.
    Uses the SSIM and Pearson correlation as metrics.
    """

    # Generate phantom and simulate the scan
    p = ct_phantom(material.name, 256, 3)
    s = source.photon('100kVp, 3mm Al')
    y = scan_and_reconstruct(s, material, p, 0.01, 256)

    # Save results 
    save_draw(y, 'results', 'test_1_image')
    save_draw(p, 'results', 'test_1_phantom')

    # Compute SSIM (structural similarity)
    ssim_score = ssim(p, y, data_range=np.max(p) - np.min(p))

    # Compute Pearson correlation coefficient
    r, _ = pearsonr(p.flatten(), y.flatten())

    # Thresholds: ensuring a good structural and intensity match
    return ssim_score > 0.95 and r > 0.98


def check_geometry():
	# explain what this test is for
	"""
	Test 2 checks that a point-source phantom can be scanned and
	reconstructed to output something that is close to a delta function at 0
	"""

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 2)		# impulse
	s = source.photon('80kVp, 1mm Al')
	y = scan_and_reconstruct(s, material, p, 0.01, 256)

	# save some meaningful results
	save_plot(y[128,:], 'results', 'test_2_plot')		# plots values along y-axis

	# check peak location is at index 128
	peak_index = np.argmax(np.abs(y[128,:]))
	peak_value = y[128,:][peak_index]

	# check the rest of the signal is near 0 but allow for transition
	# band around 0 due to unideal filtering
	transition_band = np.arange(118, 138, 1)
	rest = np.delete(y[128,:], transition_band)
	rest_max = np.max(np.abs(rest))

	# print(peak_index == 128 and rest_max < 0.2*peak_value)

	return peak_index == 128 and rest_max < 0.2*peak_value



def check_values():
	'''
    Tests CT reconstruction with phantom 2 and a fake photon source.
    Saves phantom and reconstruction results.
    Validates reconstruction by comparing the mean of the central region to an analytically computed attenuation.
    Since using phantom 2, check that the material used is of 'Soft Tissue'
	'''
	p= ct_phantom(material.name, 256, 1, metal=None) #metal=None default goes to 'Soft Tissue'
	save_draw(p, 'results', 'test_3_phantom')

	s = fake_source(material.mev, 120, method='ideal') #ideal source, (len=200), all zero excpet final energy
	y = scan_and_reconstruct(s, material, p, 0.1, 256) #256x256

	# Compute the mean intensity in the central region
	central_mean = np.mean(y[64:192, 64:192])


	# Compute expected attenuation
	coeff_soft_tissue = material.coeff('Soft Tissue')  # linear attenuation coeff (len=200)
	residual = attenuate(s, coeff_soft_tissue, depth=1) #no of photons at each energy

	I0_E = np.sum(s)
	I1 = np.sum(residual)
	mu = - np.log(I1/I0_E)
	expected_value = mu

    # Save results to file
	with open('results/test_3_output.txt', 'w') as f:
		f.write(f'Mean reconstructed value: {central_mean:.4f}\n')
		f.write(f'Expected attenuation (ideal): {expected_value:.4f}\n')

    # Assertion for test pass/fail
	assert np.isclose(central_mean, expected_value, rtol=0.07), f"Reconstruction mean {central_mean:.4f} differs from expected {expected_value:.4f}"


# Run the various tests
print('Checking reconstructed image')
check_images()
print('Checking geometry')
print(check_geometry())
print('Checking values')
check_values()
