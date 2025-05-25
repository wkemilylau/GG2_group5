import numpy as np
import matplotlib.pyplot as plt
from material import *
from source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from ct_detect import *
from fake_source import *

# Setup
material = Material()
source = Source()
#photons = source.photon('100kVp, 2mm Al')
photons = fake_source(
    mev=material.mev,
    mvp=0.1,  # 100kVp = 0.1 MeV
    coeff=material.coeff('Aluminium'),
    thickness=2,
    method='ideal'
)

#materials_to_test= ['Air', 'Soft Tissue', 'Water', 'Blood', 'Bone', 'Titanium', 'Stainless Steel', 'Aluminium', 'Copper']
materials_to_test= ['Air', 'Soft Tissue', 'Water', 'Blood', 'Bone', 'Aluminium']
measured = []
expected = []
image_size = 256
scale = 0.1  # cm/pixel
center_pixel = image_size // 2
path_length = image_size * scale  # cm

colors = ['skyblue', 'blue', 'orange', 'red', 'green', 'purple']
markers = ['o', 's', 'D', '^', 'v', '>']


mono_energy = material.mev[np.argmax(photons)]

plt.figure(figsize=(7, 7))
for i, mat_name in enumerate(materials_to_test):
    # Create uniform phantom of this material
    phantom = np.full((image_size, image_size), fill_value=material.name.index(mat_name))

    # Simulate scan and calibrate
    sinogram = ct_scan(photons, material, phantom, scale=scale, angles=180)
    attenuation_sinogram = ct_calibrate(photons, material, sinogram, scale)

    # Interpolate mu at the monoenergetic photon energy
    mu_interp = np.interp(mono_energy, material.mev, material.coeff(mat_name))
    line_integral_expected = mu_interp * path_length
    line_integral_measured = np.mean(attenuation_sinogram[:, center_pixel])

    expected.append(line_integral_expected)
    measured.append(line_integral_measured)

    plt.scatter(line_integral_expected, line_integral_measured,
                label=mat_name, color=colors[i], marker=markers[i], s=80)

# Plot ideal reference line
lims = [0, max(expected + measured) * 1.1]
plt.plot(lims, lims, 'k--', label='Ideal (y = x)')

plt.xlabel('Expected Line Integral (μ × d)')
plt.ylabel('Measured Line Integral')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()