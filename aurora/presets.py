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
    "nvector": [4096],
    "simulation_id": [""],
    "snapshot_id": [""],
    "reference_id": [""],
    "overwrite": [False],
    "ncpu": [1]
}

default_values["spectrom"] = {
    "presets": [""],
    "oversampling": [1],
    "sigma_cont": [0.0],
    "redshift_ref": [0.],
    "kernel_scale": [1.],
    "lum_dens_relation": ["square"],
    "density_threshold": ["Not"],
    "equivalent_luminosity": ["min"]

}

Instruments = {}

Instruments["sinfoni"] = {
    "spatial_sampl": "0.125",
    "spectral_sampl": "1.95",
    "spatial_res": "0.65",
    "spectral_res": "2500",
    "spatial_dim": "38",
    "spectral_dim": "48",
    "target_snr": "0"
}

Instruments["sinfoni-ao"] = {
    "spatial_sampl": "0.05",
    "spectral_sampl": "1.95",
    "spatial_res": "0.20",
    "spectral_res": "2500",
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

Instruments["kmos"] = {
    "spatial_sampl": "0.20",
    "spectral_sampl": "5.375",
    "spatial_res": "0.70",
    "spectral_res": "1800",
    "spatial_dim": "14",
    "spectral_dim": "64",
    "target_snr": "0"
}

Instruments["muse-wide"] = {
    "spatial_sampl": "0.20",
    "spectral_sampl": "1.95",
    "spatial_res": "0.65",
    "spectral_res": "300",
    "spatial_dim": "44",
    "spectral_dim": "64",
    "target_snr": "0"
}

Instruments["muse-narrow"] = {
    "spatial_sampl": "0.025",
    "spectral_sampl": "1.95",
    "spatial_res": "0.04",
    "spectral_res": "3000",
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
