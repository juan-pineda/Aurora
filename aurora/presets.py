"""
=======
presets
=======

Aurora module that contains the preset values for key words essential for
the correct operation of the code, as well as the preset values for the
simulation of synthetic observations that mimic the real instruments.

Notes
-----
The preset instruments are:

> sinfoni
> sinfoni-ao
> eagle
> kmos
> muse-wide
> muse-narrow
> ghasp
> fake1
"""

import numpy as np
from astropy import units as unit

default_values = {}

default_values["geometry"] = {
    "theta": [0., unit.deg],
    "phi": [0., unit.deg],
    "barycenter": [False],
    "centerx": [0.],
    "centery": [0.],
    "centerz": [0.],
    "reference_frame": [""],
    "gas_minmax_keys": [""],
    "gas_min_values": [""],
    "gas_max_values": [""],
}

default_values["run"] = {
    "custom_dir": [""],
    "output_file": ["Ha_flux_cube.fits"],
    "instrument": ["custom_"+str(np.random.randint(low=1, high=100))],
    "nfft": [9],
    "fft_hsml_min": [8E-3, unit.pc],
    "fft_scales": ["Not"],
    "nvector": [4096],
    "simulation_id": [""],
    "snapshot_id": [""],
    "reference_id": [""],
    "overwrite": [False],
    "ncpu": [1],
    "ncpu_convolution": [1],
    "convolution_parallel_method": [1],
    "spectral_convolution": ["analytical"],
    "spatial_convolution": ["spatial_astropy"],
    "HSIM3" : ["False"]
}

default_values["spectrom"] = {
    "obs_type": ["Halpha"],
    "presets": [""],
    "oversampling": [1],
    "sigma_cont": [0.0],
    "redshift_ref": [0.],
    "kernel_scale": [1.],
    "lum_dens_relation": ["square"],
    "density_threshold": ["Not"],
    "equivalent_luminosity": ["min"],
    "density_floor": [np.nan],
    "lum_floor": [np.nan],
    "use_ionized_hydrogen": ["True"]
}

Instruments = {}

# Preset values stablished based on the 
# official operation manual of the instrument:
# (https://www.eso.org/sci/facilities/paranal
# /decommissioned/sinfoni/doc/VLT-MAN-ESO-
# 14700-3517_v87.pdf)
Instruments["sinfoni"] = {
    "spatial_sampl": "0.125",
    "spectral_sampl": "1.95",
    "spatial_res": "0.65",
    "spectral_res": " 3000",
    "spatial_dim": "38",
    "spectral_dim": "48",
    "target_snr": "0"
}

# Preset values stablished based on the 
# official operation manual of the instrument:
# (https://www.eso.org/sci/facilities/paranal
# /decommissioned/sinfoni/doc/VLT-MAN-ESO-
# 14700-3517_v87.pdf)
Instruments["sinfoni-ao"] = {
    "spatial_sampl": "0.05",
    "spectral_sampl": "1.95",
    "spatial_res": "0.20",
    "spectral_res": "3000",
    "spatial_dim": "64",
    "spectral_dim": "48",
    "target_snr": "0"
}

Instruments["eagle"] = {
    "spatial_sampl": "0.0375",
    "spectral_sampl": "2.625",
    "spatial_res": "0.0975",
    "spectral_res": "4000",
    "spatial_dim": "96",
    "spectral_dim": "64",
    "target_snr": "0"
}

# Preset values stablished based on the 
# official operation manual of the instrument:
# (https://www.eso.org/sci/facilities/paranal
# /instruments/kmos/doc/VLT-MAN-KMO-146603-
# 001_P100.pdf)
Instruments["kmos"] = {
    "spatial_sampl": "0.20",
    "spectral_sampl": "5.375",
    "spatial_res": "0.70",
    "spectral_res": "4000",
    "spatial_dim": "14",
    "spectral_dim": "64",
    "target_snr": "0"
}

# Preset values stablished based on the 
# official operation manual of the instrument:
# (https://www.eso.org/sci/facilities/develop
# /instruments/muse.html)
Instruments["muse-wide"] = {
    "spatial_sampl": "0.20",
    "spectral_sampl": "1.95",
    "spatial_res": "0.65",
    "spectral_res": "2000",
    "spatial_dim": "44",
    "spectral_dim": "64",
    "target_snr": "0"
}

# Preset values stablished based on the 
# official operation manual of the instrument:
# (https://www.eso.org/sci/facilities/develop
# /instruments/muse.html)
Instruments["muse-narrow"] = {
    "spatial_sampl": "0.025",
    "spectral_sampl": "1.95",
    "spatial_res": "0.04",
    "spectral_res": "2000",
    "spatial_dim": "44",
    "spectral_dim": "64",
    "target_snr": "0"
}

Instruments["ghasp"] = {
    "spatial_sampl": "0.68",
    "spectral_sampl": "0.30",
    "spatial_res": "2.0",
    "spectral_res": "10000",
    "spatial_dim": "512",
    "spectral_dim": "38",
    "target_snr": "0"
}

Instruments["fake1"] = {
    "spatial_sampl": "1.",
    "spectral_sampl": "0.5",
    "spatial_res": "2.0",
    "spectral_res": "10000",
    "spatial_dim": "200",
    "spectral_dim": "60",
    "target_snr": "0"
}
