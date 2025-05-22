
# these are the imports you are likely to need
import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from create_dicom import *

# create object instances
material = Material()
source = Source()

# define each end-to-end test here, including comments
# these are just some examples to get you started
# all the output should be saved in a 'results' directory

def test_1():
	# explain what this test is for

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 3)
	s = source.photon('100kVp, 3mm Al')
	y = scan_and_reconstruct(s, material, p, 0.01, 256)

	# save some meaningful results
	save_draw(y, 'results', 'test_1_image')
	save_draw(p, 'results', 'test_1_phantom')

	# how to check whether these results are actually correct?

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
	# explain what this test is for

	# work out what the initial conditions should be
	p = ct_phantom(material.name, 256, 1)
	save_draw(p, 'results', 'test_3_phantom')
	s = source.photon('80kVp, 1mm Al')
	y = scan_and_reconstruct(s, material, p, 0.1, 256)

	# save some meaningful results
	f = open('results/test_3_output.txt', mode='w')
	f.write('Mean value is ' + str(np.mean(y[64:192, 64:192])))
	f.close()

	# how to check whether these results are actually correct?
	expected_mean = 640  # hypothetical expected mean
	actual_mean = np.mean(y[64:192, 64:192])
	print(actual_mean)
	assert np.isclose(actual_mean, expected_mean, rtol=0.05), f"Mean {actual_mean} differs from expected {expected_mean}"


# Run the various tests
print('Test 1')
test_1()
print('Test 2')
print(test_2())
print('Test 3')
test_3()
