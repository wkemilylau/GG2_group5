import numpy as np
from attenuate import attenuate
from ct_calibrate import ct_calibrate

def hu(p, material, reconstruction, scale):
    depth = np.array([1.0])  # 1 cm path length

    coeff_water = material.coeff('Water')
    coeff_air = material.coeff('Air')

    I_water = attenuate(p, coeff_water, depth)
    I_air = attenuate(p, coeff_air, depth)

    sinogram_water = np.sum(I_water, axis=0).reshape(1, 1)
    sinogram_air = np.sum(I_air, axis=0).reshape(1, 1)

    mu_water = ct_calibrate(p, material, sinogram_water, scale)[0, 0]
    mu_air = ct_calibrate(p, material, sinogram_air, scale)[0, 0]

    denominator = mu_water - mu_air
    if abs(denominator) < 1e-8:
        denominator = 1e-8

    hu_image = 1000 * (reconstruction - mu_air) / denominator

    return hu_image

















