import matplotlib.pyplot as plt
import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
#from ct_lib import *
from scan_and_reconstruct import *
from create_dicom import *
from ct_detect import *

# create object instances
material = Material()
source = Source()


#y=residual energy after 100kVp x-ray source with 2mm Al filter
#passed through water for depths 0cm to 10cm in 0.1cm intervals
p = source.photon('100kVp, 2mm Al')
coeffs = material.coeff('Water')
depth = np.arange(0, 10.1, 0.1)
mas = 1
y = ct_detect(p, coeffs, depth, mas)

'''
plt.plot(np.log(y))
plt.xlabel('Depth (cm)')
plt.ylabel('Log detector signal (residual energy)')
plt.grid(True)
plt.show()
'''
####################

#plotting y=residual energy
#changing the incident photon energy and filter thickness

# Define sources: (label, photon spectrum, color, line width)
sources = [
    ('100kVp, 1mm Al', source.photon('100kVp, 1mm Al'), 'blue', 1),
    ('100kVp, 2mm Al', source.photon('100kVp, 2mm Al'), 'blue', 2),
    ('100kVp, 3mm Al', source.photon('100kVp, 3mm Al'), 'blue', 3),
    ('100kVp, 4mm Al', source.photon('100kVp, 4mm Al'), 'blue', 4),

    ('80kVp, 1mm Al', source.photon('80kVp, 1mm Al'), 'green', 1),
    ('80kVp, 2mm Al', source.photon('80kVp, 2mm Al'), 'green', 2),
    ('80kVp, 3mm Al', source.photon('80kVp, 3mm Al'), 'green', 3),
    ('80kVp, 4mm Al', source.photon('80kVp, 4mm Al'), 'green', 4),

    ('Ideal 100kVp, 2mm Al', fake_source(
        mev=material.mev,
        mvp=0.1,
        coeff=material.coeff('Aluminium'),
        thickness=2,
        method='ideal'
    ), 'red', 2, '--')  # Add linestyle
]

# Plot all sources
plt.figure(figsize=(10, 6))
for s in sources:
    label, photons, color, lw = s[:4]
    linestyle = s[4] if len(s) > 4 else '-'  # Default to solid line

    y = ct_detect(photons, coeffs, depth, mas)
    plt.plot(depth, np.log(y), label=label, color=color, linewidth=lw, linestyle=linestyle)

plt.xlabel('Depth in Water (cm)')
plt.ylabel('Log Residual Detector Signal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

##############################

# Define sources: (label, photon spectrum, color, line width)
#photons = source.photon('100kVp, 2mm Al')  # Or use fake_source if you prefer
photons = fake_source(
        mev=material.mev,
        mvp=0.1,
        coeff=material.coeff('Aluminium'),
        thickness=2,
        method='ideal'
    )

# Depth in water (same for all)
depth = np.arange(0, 10.1, 0.1)
mas = 1

# List of materials to compare
material_names = [
    'Air',
    'Soft Tissue',
    'Water',
    'Blood',
    'Bone',
    'Titanium',
    'Stainless Steel',
    'Aluminium',
    'Copper'
]

plt.figure(figsize=(12, 7))
for name in material_names:
    try:
        coeffs = material.coeff(name)
        y = ct_detect(photons, coeffs, depth, mas)
        plt.plot(depth, np.log(y), label=name)
    except Exception as e:
        print(f"Could not load material '{name}': {e}")

plt.xlabel('Depth (cm)')
plt.ylabel('Log Residual Detector Signal')
plt.title(f'Attenuation in Different Materials for {label}')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

######################

depth = np.arange(0, 10.1, 0.1)
mas = 1

material_names = [
    'Air', 'Soft Tissue', 'Water', 'Blood', 'Bone',
    'Titanium', 'Stainless Steel', 'Aluminium', 'Copper'
]

colors = plt.get_cmap('tab10')

real_label = '100kVp, 2mm Al'
real_photons = source.photon(real_label)

fake_photons = fake_source(
    mev=material.mev,
    mvp=0.1,  # 100kVp = 0.1 MeV
    coeff=material.coeff('Aluminium'),
    thickness=2,
    method='ideal'
)

plt.figure(figsize=(12, 7))

for idx, name in enumerate(material_names):
    try:
        coeffs = material.coeff(name)
        color = colors(idx % 10)

        # Real source (solid line)
        y_real = ct_detect(real_photons, coeffs, depth, mas)
        plt.plot(depth, np.log(y_real), label=f'{name} (Real)', color=color, linewidth=2)

        # Ideal source (dashed line)
        y_fake = ct_detect(fake_photons, coeffs, depth, mas)
        plt.plot(depth, np.log(y_fake), '--', label=f'{name} (Ideal)', color=color, linewidth=2)

    except Exception as e:
        print(f"Could not load material '{name}': {e}")

plt.xlabel('Depth (cm)')
plt.ylabel('Log Residual Detector Signal')
plt.grid(True)
plt.legend(ncol=2)
plt.tight_layout()
plt.show()