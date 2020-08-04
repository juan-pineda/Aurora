import os
import sys
import math
import logging
import astropy
import numpy as np
from tqdm import tqdm
import astropy.convolution

from joblib import Parallel, delayed

from . import constants as ct
from . import emitters as emit
from . import convolutions as cv

def __project_all_chunks(geom, run, spectrom, data_gas):
    """
    Split gas particles into several chunks according to *nvector* and
    compute the projected flux one by one. The resulting structure is a
    4D-array.

    Parameters
    ----------
    geom : aurora.configuration.GeometryObj
        Instance of class GeometryObj whose attributes make geometric 
        properties available. See definitions in configuration.py.
    run : aurora.configuration.RunObj
        Instance of class RunObj whose attributes make code computational
        performance properties available. See definitions in
        configuration.py.
    spectrom : aurora.configuration.SpectromObj
        Instance of class SpectromObj whose attributes make instrumental
        properties available. See definitions in configuration.py.
    data_gas : pynbody.snapshot.IndexedSubSnap 
        Gas particles identified in the input archive.

    Returns
    -------
    cube : ndarray (4D)
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
    Determine the availability of RAM memory to carry out the flow
    projection process, and sets the upper and lower limits on gas 
    particles according to *nchunks*.

    Parameters
    ----------
    geom : aurora.configuration.GeometryObj
        Instance of class GeometryObj whose attributes make geometric 
        properties available. See definitions in configuration.py.
    run : aurora.configuration.RunObj
        Instance of class RunObj whose attributes make code computational
        performance properties available. See definitions in
        configuration.py.
    spectrom : aurora.configuration.SpectromObj
        Instance of class SpectromObj whose attributes make instrumental
        properties available. See definitions in configuration.py.
    data_gas : pynbody.snapshot.IndexedSubSnap 
        Gas particles identified in the input archive.
    nchunk : int
        Number of chunks to divide the gas particles.
        
    Returns
    -------
    cube : ndarray (4D)
        Contains the fluxes at each pixel and velocity channel 
        produced by the gas particles with a given smoothing
        lengths separately.
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
    to a 4D grid, keeping contributions from different scales separated.

    Parameters
    ----------
    geom : aurora.configuration.GeometryObj
        Instance of class GeometryObj whose attributes make geometric 
        properties available. See definitions in configuration.py.
    run : aurora.configuration.RunObj
        Instance of class RunObj whose attributes make code computational
        performance properties available. See definitions in
        configuration.py.
    spectrom : aurora.configuration.SpectromObj
        Instance of class SpectromObj whose attributes make instrumental
        properties available. See definitions in configuration.py.
    data_gas : pynbody.snapshot.IndexedSubSnap 
        Gas particles identified in the input archive.
    *args : int, array_like
        Number of chunks to divide the gas particles, or the list of
        the upper and lower limits of the gas particles to be projected.
    
    Returns
    -------
    cube : ndarray (4D)
        Contains the fluxes at each pixel and velocity channel 
        produced by the gas particles with a given smoothing
        lengths separately.
    """
    
    cube_side, n_ch = spectrom.cube_dims()
    if len(args) == 1:
        i = args[0]
        start = i * run.nvector
        stop = start + min(run.nvector, len(data_gas) - start)
        cube = np.zeros((n_ch, cube_side, cube_side, run.nfft))
    else:
        start, stop, cube = args

    # This object allows to calculate the Halpha flux, and line broadening
    em = emit.Emitters(data_gas[start:stop], spectrom.redshift_ref)
    em.get_state()
    em.get_luminosity(spectrom.lum_dens_rel)
    em.density_cut(spectrom.density_threshold, spectrom.equivalent_luminosity)
    em.get_vel_dispersion()

    x, y, index = spectrom.position_in_pixels(em.x,em.y)

    # scale to which each particle belongs according to its smoothing lenght
    scale = np.digitize(em.smooth.to("kpc"), 1.1 * run.fft_hsml_limits.to("kpc"))

    line_center, line_sigma, line_flux = em.get_vect_lines(n_ch)
    channel_center, channel_width = em.get_vect_channels(spectrom.vel_channels, spectrom.velocity_sampl, n_ch)

    # Spectral convolution
    if(spectrom.spectral_res > 0 and run.spectral_convolution == 'analytical'):
        psf_fwhm = ct.c/spectrom.spectral_res
        psf_sigma = psf_fwhm / ct.fwhm_sigma
        line_sigma = np.sqrt(line_sigma**2+psf_sigma**2)

    # Integrated flux inside each velocity channel given its position and width
    
    # Example:
    # --------
    # With n particles, m velocity channels and X the integrated flux inside
    # each velocity channel, flux_in_channels is:
    #    
    # [ X11 X12 X13 ... X1m
    #   X21 X22 X23 ... X2m
    #    .   .   .  ...
    #    .   .   .  ...
    #   Xn1 Xn2 Xn3 ... Xnm]
    
    flux_in_channels = em.int_gaussian_with_units(channel_center, channel_width, line_center,
        line_sigma) * line_flux

    # Divide by the effective channel width
    flux_in_channels = (flux_in_channels.to("erg s^-1").value
        /spectrom.velocity_sampl.to("km s^-1").value/(4*np.pi*geom.dl.to("cm").value**2))

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

def __cube_spatial_convolution(run, spectrom, cube):
    """
    Perform the spatial smoothing of fluxes projected to a 4D-grid.
    
    Notes
    -----
    Consider two kernels: the multi-scale kernels of the simulation and
    the spatial PSF if spectrom.spatial_res was defined. 

    Parameters
    ----------
    run : aurora.configuration.RunObj
        Instance of class RunObj whose attributes make code computational
        performance properties available. See definitions in
        configuration.py.
    spectrom : aurora.configuration.SpectromObj
        Instance of class SpectromObj whose attributes make instrumental
        properties available. See definitions in configuration.py.
    cube : ndarray (4D)
        Contains the fluxes at each pixel and velocity channel 
        produced by the gas particles with a given smoothing
        lengths separately.
    """
    
    # Code flow:
    # ==========
    # > Create the kernel smoothing.
    # > Adds the effect of spatial resolution in the kernel.
    # > Apply the spatial convolution method as configured in
    #   spectrum.spatial_convolution
    cube_side, n_ch = spectrom.cube_dims()
    for i in range(run.nfft):
        logging.info(f"Preparing for spatial smoothing, kernel = {round(run.fft_hsml_limits[i].value*1000, 1)} pc")
        sys.stdout.flush()
        # Kernel smoothing
        scale_fwhm = (run.fft_hsml_limits[i] / spectrom.pixsize).decompose().value
        scale_sigma = spectrom.kernel_scale * scale_fwhm / ct.fwhm_sigma
        logging.info(f"Size of the kernel in pixels = {round(scale_sigma, 1)}")  
        # Enlarge the kernel adding the effect of the PSF
        psf = cv.create_psf(spectrom, scale_sigma)
        # Spatial convolution
        cube[:, :, :, i] = cv.mode_spatial_convolution(cube[:, :, :, i], psf, run.spatial_convolution)

def __cube_spectral_convolution(run, spectrom, cube, mode = 'analytical'):
    """
    Perform the spectral smoothing of fluxes projected to a 3D-grid.
    
    Notes
    -----
    If spectrom.spectral_res was defined, by default a Gaussian kernel
    would be created to represent the LSF to apply spectral smoothing.

    Parameters
    ----------
    run : aurora.configuration.RunObj
        Instance of class RunObj whose attributes make code computational
        performance properties available. See definitions in
        configuration.py.    
    spectrom : aurora.configuration.SpectromObj
        Instance of class SpectromObj whose attributes make instrumental
        properties available. See definitions in configuration.py.
    cube : ndarray (3D)
        Contains the fluxes at each pixel and velocity channel 
        produced by the gas particles.
    """
    # Code flow:
    # ==========
    # > Create the kernel. 
    # > Apply the spectral convolution method as configured in
    #   spectrum.spectral_convolution, in case it is not analytical convolution.
    if mode != 'analytical' and spectrom.spectral_res > 0:
        # Kernel create
        lsf = cv.create_lsf(spectrom)
        # Spectral convolution
        cube = cv.mode_spectral_convolution(cube, lsf, run.spectral_convolution)        