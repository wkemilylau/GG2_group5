
# these are the imports you are likely to need
import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from create_dicom import *
from attenuate import *


# create object instances
material = Material()
source = Source()

# define each end-to-end test here, including comments
# these are just some examples to get you started
# all the output should be saved in a 'results' directory

def test_1():
    # Checking that mean of reconstructed image is close to mean of phantom

    p = ct_phantom(material.name, 256, 3)
    s = source.photon('100kVp, 3mm Al')
    y = scan_and_reconstruct(s, material, p, 0.01, 256)

    save_draw(y, 'results', 'test_1_image')
    save_draw(p, 'results', 'test_1_phantom')

    mean_p = np.mean(p)
    mean_y = np.mean(y)

    # Checking that means are reasonably close (within 1%)
    return abs(mean_p - mean_y) / mean_p < 0.01


def test_2():
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



def test_3():
	'''
    Tests CT reconstruction with phantom 2 and a fake photon source.
    Saves phantom and reconstruction results.
    Validates reconstruction by comparing the mean of the central region to an analytically computed attenuation.
    Since using phantom 2, check that the material used is of 'Soft Tissue'
	'''
	p= ct_phantom(material.name, 256, 1, metal=None)
	save_draw(p, 'results', 'test_4_phantom')

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
	with open('results/test_4_output.txt', 'w') as f:
		f.write(f'Mean reconstructed value: {central_mean:.4f}\n')
		f.write(f'Expected attenuation (ideal): {expected_value:.4f}\n')

    # Assertion for test pass/fail
	assert np.isclose(central_mean, expected_value, rtol=0.07), f"Reconstruction mean {central_mean:.4f} differs from expected {expected_value:.4f}"


# Run the various tests
print('Test 1')
test_1()
print('Test 2')
print(test_2())
print('Test 3')
test_3()

print('Test 4')
test_4()
