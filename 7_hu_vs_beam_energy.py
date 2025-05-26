
import matplotlib.pyplot as plt
import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
#from scan_and_reconstruct import *
from create_dicom import *
from attenuate import *

from ct_scan import *
from ct_calibrate import *
from ct_lib import *
from ramp_filter import *
from back_project import *
from hu import *
from scan_and_reconstruct import *

material = Material()
source = Source()

energies = [80, 100, 120]  # example energy levels in keV

phantom = ct_phantom(material.name, 256, 1) #creates 256x256 image

# Dictionary to hold HU lists for each material
hu_by_material = {
    mat: []
    for mat in material.name
}

mas=10000
scale = 0.01

for E in energies:
    mvp = E / 1000  # convert keV to MeV
    mev = material.mev  # energy bins, in MeV

    # Monoenergetic ideal source:
    source = fake_source(mev, mvp, method='ideal')
    source_input = source * mas * (scale ** 2)

    recon = scan_and_reconstruct(source_input, material, phantom, scale=0.01, angles=256)

    for mat in material.name:
        idx = material.name.index(mat)
        mean_hu = np.mean(recon[phantom == idx])
        hu_by_material[mat].append(mean_hu)


for mat, hu_values in hu_by_material.items():
    if any(np.isfinite(hu_values)) and any(np.array(hu_values) != 0):
        plt.plot(energies, hu_values, label=mat, marker='o')

plt.xlabel('Beam energy (keV)')
plt.ylabel('HU Value')
plt.title('Stability of HU Values with beam energy')
#plt.yscale('log')  # Logarithmic y-axis
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
