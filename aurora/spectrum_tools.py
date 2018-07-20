import os
import gc
import sys
import math
import logging
import astropy
import numpy as np
from tqdm import tqdm
import astropy.convolution
from scipy import interpolate
from astropy import constants as const

from joblib import Parallel, delayed
import multiprocessing

from . import aurora as au
from . import snapshot_tools as snap
from . import gasProps_sBird as bird
from . import datacube as dc
from . import constants as ct
from . import emitters as emit
from . import convolutions as cv

def __project_all_chunks(geom, run, spectrom, data_gas):
    """
    Split gas particles into several chunks according to *nvector* and
    compute the projected flux one by one. The resulting structure is 4D


    Parameters
    ----------
    geom : object of class *geometry_obj*
        See definitions in configuration.py.
    run : object of class *run_obj*
        See definitions in configuration.py.
    spectrom : object of class *spectrom_obj*
        See definitions in configuration.py.

    Returns
    -------
    cube : 4D-array
        Contains the fluxes at each pixel and velocity channel produced
        by the gas particles with a given smoothing lengths separately.
    """

    # Code flow:
    # =====================
    # > Creates the 4D output array
    # > Define the number of chunks
    # > Project and add the fluxes iteratively

    nchunk = int(math.ceil(len(data_gas) / float(run.nvector)))

    if run.ncpu > 1:
        return get_cube_in_parallel(geom, run, spectrom, data_gas, nchunk)
    else:
        return get_cube_in_sequential(geom, run, spectrom, data_gas, nchunk)


def get_cube_in_parallel(geom, run, spectrom, data_gas, nchunk):
    """
    """
    cube_side, n_ch = spectrom.cube_dims()
    cube_size = np.zeros((n_ch, cube_side, cube_side, run.nfft)).nbytes
    memory_needed_ncores = int((cube_size/1e6) * min(run.ncpu, nchunk))
    memory_available = int(os.popen("free -m").readlines()[1].split()[-1])

    num_cores = int(run.ncpu)
    if memory_available > memory_needed_ncores:
        cube_list = Parallel(n_jobs=num_cores)(delayed(__project_spectrom_flux)
                                               (geom, run, spectrom, data_gas, i) for i in range(nchunk))
        return sum(cube_list)
    else:
        logging.warning(f"Not enough RAM left in your device for this operation in parallel.")
        logging.info(f"Needed {memory_needed_ncores}Mb, you have {memory_available}Mb Free.")
        logging.info(f"Using a single cpu mode...")
        return get_cube_in_sequential(geom, run, spectrom, data_gas, nchunk)


def get_cube_in_sequential(geom, run, spectrom, data_gas, nchunk):
    """
    """
    cube_side, n_ch = spectrom.cube_dims()
    cube_size = np.zeros((n_ch, cube_side, cube_side, run.nfft)).nbytes
    memory_needed_1core = int(cube_size/1e6)
    memory_available = int(os.popen("free -m").readlines()[1].split()[-1])

    if memory_available > memory_needed_1core:
        if abs(memory_available-memory_needed_1core) < 1000:
            logging.warning(f"Your computer may be slow during this operation, be patient.")
        cube = np.zeros((n_ch, cube_side, cube_side, run.nfft))
        for i in tqdm(range(nchunk)):
            start = i * run.nvector
            stop = start + min(run.nvector, len(data_gas) - start)
            __project_spectrom_flux(
                geom, run, spectrom, data_gas, start, stop, cube)           
        return cube
    else:
        raise MemoryError(f"Not enough RAM in your device.")


def __project_spectrom_flux(geom, run, spectrom, data_gas, *args):
    """
    Compute the H-alpha emission of a bunch of particles and project it
    to a 4D grid, keeping contributions from different scales separated

    Parameters
    ----------
    """
    cube_side, n_ch = spectrom.cube_dims()
    if len(args) == 1:
        i = args[0]
        start = i * run.nvector
        stop = start + min(run.nvector, len(data_gas) - start)
        cube = np.zeros((n_ch, cube_side, cube_side, run.nfft))
    else:
        start, stop, cube = args

#    if geom.redshift > 0:
#        dl = geom.dl.to("cm").value
#    else:
#        dl = geom.dl.to("pc").value

	# This object allows to calculate the Halpha flux, and line broadening
    em = emit.Emitters(data_gas[start:stop], spectrom.redshift_ref)
    em.get_state()
#    em.density_cut(spectrom.density_cut) # new feature in test !!!	
    em.get_luminosity(spectrom.lum_dens_rel, spectrom.density_cut) # new feature in test !!!
    em.get_vel_dispersion()

    x, y, index = spectrom.position_in_pixels(em.x,em.y)

	# scale to which each particle belongs according to its smoothing lenght
    scale = np.digitize(em.smooth.to("kpc"), 1.1 * run.fft_hsml_limits.to("kpc"))

    line_center, line_sigma, line_flux = em.get_vect_lines(n_ch)
    channel_center, channel_width = em.get_vect_channels(spectrom.vel_channels, spectrom.velocity_sampl, n_ch)

    # Spectral convolution
    if(spectrom.spectral_res > 0):
        psf_fwhm = ct.c/spectrom.spectral_res
        psf_sigma = psf_fwhm / ct.fwhm_sigma
        line_sigma = np.sqrt(line_sigma**2+psf_sigma**2)

    # Integrated flux inside each velocity channel given its position and width
    flux_in_channels = em.int_gaussian_with_units(channel_center, channel_width, line_center,
        line_sigma) * line_flux

    # Divide by the effective channel width
    flux_in_channels = flux_in_channels.to("erg s^-1").value / spectrom.velocity_sampl.to("km s^-1").value / spectrom.pixsize.to("pc").value**2
    # Compute the fluxes scale by scale
    for i in np.unique(scale):
        ok_level = np.where(scale == i)[0]
        nok_level = ok_level.size
        
        # Unique indices (pixels) to which particles in this group contribute
        unique_val, unique_ind = np.unique(index[ok_level], return_index=True)

        eff_flux = flux_in_channels[ok_level]

        # Sum all the lines for a given index
        for j in range(unique_val.size):
            to_sum = np.where(index[ok_level] == unique_val[j])[0]
            eff_flux[unique_ind[j], :] = np.sum(eff_flux[to_sum, :], axis=0)
        # Remove duplicated emission lines
        eff_flux = eff_flux[unique_ind, :]
        # Insert the line fluxes in the right positions at the right scale

        cube[:, y[ok_level[unique_ind]],
             x[ok_level[unique_ind]], i] += np.transpose(eff_flux)
    return cube


def __cube_convolution(geom, run, spectrom, cube):
    """
    Perform the spatial smoothing of fluxes projected to a 4D-grid.
    Consider two kernels: the multi-scale kernels of the simulation and
    the spatial PSF if it was defined

    Parameters
    ----------
    """
    cube_side, n_ch = spectrom.cube_dims()
    for i in range(run.nfft):
        logging.info(f"Preparing for spatial smoothing, kernel = {round(run.fft_hsml_limits[i].value*1000, 1)} pc")
        sys.stdout.flush()
        # Kernel smoothing
        scale_fwhm = (run.fft_hsml_limits[i] / spectrom.pixsize).decompose().value
        scale_sigma = spectrom.kernel_scale * scale_fwhm / ct.fwhm_sigma
        logging.info(f"Size of the kernel in pixels = {round(scale_sigma, 1)}")
        # Enlarge the kernel adding the effect of the PSF
        if(spectrom.spatial_res_kpc > 0):
            logging.info(f" (Including the effect of the PSF as well)")
####            psf_fwhm = spectrom.spatial_res.value / spectrom.spatial_sampl.value
            psf_fwhm = spectrom.spatial_res_kpc.to(
                "pc").value / spectrom.pixsize.to("pc").value
            psf_sigma = psf_fwhm / ct.fwhm_sigma
            logging.info(f"Size of the PSF in pixels = {round(psf_sigma, 1)}")
            scale_sigma = np.sqrt(scale_sigma**2+psf_sigma**2)
            logging.info(f"Combination kernel + PSF in pixels = {round(scale_sigma, 1)}")
        if (scale_sigma <= 0.5):
            logging.info(f"-- Small kernel -> skip convolution")
            sys.stdout.flush()
            continue

# This is the way I performed the psf convolution until 20/07/2018
# Then I realized the FFT is introducing non-isotropic noise, aligned along the axes
# And realized as well that shaping the psf in a square of a side = 8-sigma is not
# enough. When the log(flux) is considered, sharp edges and square patterns appear
#        m = dc.DatacubeObj()
#        m.cube = cube[:, :, :, i]
#        psf_fwhm = scale_sigma * ct.fwhm_sigma
#        cv.spatial_convolution_iter(m.cube, psf_fwhm)
#        cube[:, :, :, i] = m.cube

# So, in the same date 20/07/2018 I decided to recover a simpler scheme that might take
# longer, but it is way cleaner:
        for j in range(n_ch):
            if (np.nanmax(cube[j, :, :, i]) == 0):
                logging.info(f"No flux at this scale/velocity channel -> skip convolution")
                continue
            side = cv.next_odd(20*psf_sigma)  
            psf = astropy.convolution.Gaussian2DKernel(scale_sigma, x_size=side, y_size=side)
# FFT or spatial convolution? 
# It depends on whether the noise introduced by FFT schemes is important or not
#            channel = astropy.convolution.convolve_fft(cube[j,:,:,i], psf,psf_pad=True,normalize_kernel=np.sum,allow_huge=True)
            channel = astropy.convolution.convolve(cube[j,:,:,i],psf)
            cube[j, :, :, i] = 0.
            cube[j, :, :, i] += channel

