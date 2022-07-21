"""
==============
spectrum_tools
==============

This module contains the methods that generate the emission 
lines, using the emitter module, construct the data cube by 
projecting the emission lines, and those that apply spatial 
and spectral convolution.
"""

import os
import sys
import math
import time
import logging
import astropy
import numpy as np
from tqdm import tqdm
import astropy.convolution

import signal
import multiprocessing as mp
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

    # This object allows to calculate the HI or Halpha flux, and line broadening
    em = emit.Emitters(data_gas[start:stop], spectrom.redshift_ref)
    em.get_state(spectrom)
    if spectrom.obs_type == "HI":
        em.get_luminosityHI()
    else:
        em.get_luminosityHalpha(spectrom.lum_dens_rel)
    em.apply_density_cut(spectrom.density_threshold, spectrom.equivalent_luminosity,
                          spectrom.density_floor, spectrom.lum_floor)
    em.get_vel_dispersion()

    x, y, index = spectrom.position_in_pixels(em.x,em.y)

    # scale to which each particle belongs according to its smoothing lenght
    scale = np.digitize(em.smooth.to("kpc").value, 1.1 * run.fft_hsml_limits.to("kpc").value)

    line_center, line_sigma, line_flux = em.get_vect_lines(n_ch, spectrom.obs_type)
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
    if run.HSIM3 == True:
        flux_in_channels = (flux_in_channels.to("erg s^-1").value
        /spectrom.spectral_sampl.to("um").value/(4*np.pi*geom.dl.to("cm").value**2)
        /spectrom.spatial_sampl.to("arcsec").value**2)
    else:
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

def __cube_spatial_convolution_in_parallel_test(run, spectrom, cube):
    """
    """
    cube_side, n_ch = spectrom.cube_dims()
    cube_size = np.zeros((n_ch, cube_side, cube_side, run.nfft)).nbytes
    memory_needed_ncores = int((cube_size/1e6) * min(run.ncpu_convolution, run.nfft))
    memory_available = int(os.popen("free -m").readlines()[1].split()[-1])

    num_cores = int(run.ncpu_convolution)
    if memory_available > memory_needed_ncores:
        print('Start parallel convolution in the different spatial scale')
#cube_list = 
        cube_list = Parallel(n_jobs=num_cores)(delayed(__cube_scale_spatial_convolution)
                                               (run, spectrom, cube.copy(), i) for i in range(run.nfft))
        print(type(cube_list), len(cube_list), cube_list[0].shape,run.nfft)
        return np.sum(cube, axis=3)
    else:
        print('Not enough RAM left in your device for this operation in parallel')
        logging.warning(f"Not enough RAM left in your device for this operation in parallel.")
        logging.info(f"Needed {memory_needed_ncores}Mb, you have {memory_available}Mb Free.")
        logging.info(f"Using a single cpu mode...")
        #break

def __cube_spatial_convolution_in_parallel_1(run, spectrom, cube):
    """
    """
    cube_side, n_ch = spectrom.cube_dims()
    num_cores = int(run.ncpu_convolution)

    def init_worker():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    pool = mp.Pool(num_cores, init_worker)
    outputs = []
    print('Start parallel convolution in the different spatial scale')
    for scale_index in range(run.nfft):
        print('Scale: ', run.nfft-scale_index-1)
        logging.info(f"Preparing for spatial smoothing, kernel = {round(run.fft_hsml_limits[scale_index].value*1000, 1)} pc")
        sys.stdout.flush()
        # Kernel smoothing
        scale_fwhm = (run.fft_hsml_limits[run.nfft-scale_index-1] / spectrom.pixsize).decompose().value
        scale_sigma = spectrom.kernel_scale * scale_fwhm / ct.fwhm_sigma
        logging.info(f"Size of the kernel in pixels = {round(scale_sigma, 1)}")  
        # Enlarge the kernel adding the effect of the PSF
        psf = cv.create_psf(spectrom, scale_sigma)
        result = []
        outputs1 = []
        for i in range(n_ch):
            #Convolution in the slides of the 3D cube in the scale_index
           # slice_max = np.floor(np.log10(cube[i,:,:,run.nfft-scale_index-1].max()))
           # slice_min = np.floor(np.log10(cube[i,:,:,run.nfft-scale_index-1][cube[i,:,:,run.nfft-scale_index-1]>0].min()))
            #contrast = slice_max - slice_min
            #if (spectrom.slice_cut == 'contrast') and (contrast > 15):
             #   cube[i,:,:,run.nfft-scale_index-1] = cv.lum_cut_slice(spectrom, cube[i,:,:,run.nfft-scale_index-1])
            #else:
             #   print("Nothing to cut in the slice")
            result.append(pool.apply_async(astropy.convolution.convolve_fft, args=(cube[i,:,:,run.nfft-scale_index-1],
                                                  psf), kwds={'psf_pad' : True, 'fft_pad' : True, 'allow_huge' : True}))                    
        for r in result:                    
            try:
                outputs1.append(r.get())
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
        
        outputs.append(outputs1)  

    pool.close()
    pool.join() 
    cube = np.array(outputs)
    #mask = cube < 0
    #cube[mask] = 0
    return cube.sum(axis=0)

def __cube_spatial_convolution_in_parallel_2(run, spectrom, cube):
    """
    """
    cube_side, n_ch = spectrom.cube_dims()
    cube_size = np.zeros((n_ch, cube_side, cube_side, run.nfft)).nbytes
    memory_needed_ncores = int((cube_size/1e6) * min(run.ncpu_convolution, run.nfft))
    memory_available = int(os.popen("free -m").readlines()[1].split()[-1])

    num_cores = int(run.ncpu_convolution)
    if memory_available > memory_needed_ncores:
        print('Start parallel convolution in the different spatial scale')
        def init_worker():
	        signal.signal(signal.SIGINT, signal.SIG_IGN)

        pool = mp.Pool(num_cores, init_worker)	
        result = []
        outputs = []	
        for j in range(run.nfft):                
                r1 = pool.apply_async(__cube_scale_spatial_convolution, args=(run, spectrom, cube, j))
                result.append(r1)
        for r in result:
            try:
                outputs.append(r.get())
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()

        pool.close()
        pool.join() 
        print(type(outputs), len(outputs), run.nfft)
        return sum(outputs)
       # cube = np.sum(cube, axis=3)        
       # print(type(cube), cube.shape, run.nfft)
       # return cube
    else:
        print('Not enough RAM left in your device for this operation in parallel')
        logging.warning(f"Not enough RAM left in your device for this operation in parallel.")
        logging.info(f"Needed {memory_needed_ncores}Mb, you have {memory_available}Mb Free.")
#        logging.info(f"Using a single cpu mode...")

def __cube_scale_spatial_convolution(run, spectrom, cube, scale_index):
    """
    Perform the spatial smoothing of fluxes for a given spatial scale of the
    simulations in the cube.
    
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
    scale_index : integer                   
        Index of the spatial scale to perform the convolution.
    """
    
    # Code flow:
    # ==========
    # > Create the kernel smoothing.
    # > Adds the effect of spatial resolution in the kernel.
    # > Apply the spatial convolution method for a given 
    # scale as configured in spectrum.spatial_convolution
    cube_side, n_ch = spectrom.cube_dims()
    logging.info(f"Preparing for spatial smoothing, kernel = {round(run.fft_hsml_limits[scale_index].value*1000, 1)} pc")
    sys.stdout.flush()
    # Kernel smoothing
    scale_fwhm = (run.fft_hsml_limits[scale_index] / spectrom.pixsize).decompose().value
    scale_sigma = spectrom.kernel_scale * scale_fwhm / ct.fwhm_sigma
    logging.info(f"Size of the kernel in pixels = {round(scale_sigma, 1)}")  
    # Enlarge the kernel adding the effect of the PSF
    psf = cv.create_psf(spectrom, scale_sigma)
    # Spatial convolution
#    cube[:, :, :, scale_index] = cv.mode_spatial_convolution(cube[:, :, :, scale_index], psf, run.spatial_convolution)
#    print('Scale: ', run.nfft-scale_index-1)
    print('Scale: ', scale_index)
    cube_convolve = cv.mode_spatial_convolution(cube[:, :, :, scale_index], psf, run.spatial_convolution)
#    cube[:, :, :, scale_index] = cv.mode_spatial_convolution(cube[:, :, :, scale_index], psf, run.spatial_convolution)
    print('End scale: ', scale_index)
    return cube_convolve

def flux_cut(spectrom, cube):
    """
    Replaces the flux in the data cube by a user-defined maximum, minimum
    value or limits the contrast to 15 orders of magnitude (or a 
    user-defined  value) to avoid numerical instabilities in the spatial
    convolution using FFT.
    
    Parameters
    ----------
    spectrom : aurora.configuration.SpectromObj
        Instance of class SpectromObj whose attributes make instrumental
        properties available. See definitions in configuration.py.
    cube = ndarray (4D)
        Contains the fluxes at each pixel for a velocity/spectral
        channel produced by the gas particles with a smoothing lengths.

    """    
    # Code flow:
    # ==========
    # > Calculate the stddev of the LSF with the resolving power
    # > Kernel creation with Astropy
    # > Normalizes the kernel

    if spectrom.flux_cut_model == 'contrast_floor':    
        cube_max = np.floor(np.log10(cube.max()))
        cube_min = np.floor(np.log10(cube[cube>0].min()))
        floor_flux = 10**np.float(cube_max - spectrom.flux_cut[0])
        print(cube_max, cube_min, floor_flux)
        print("Contrast cutting a floor: ", floor_flux)
        logging.info(f"Contrast cut of: {floor_flux}")  
        cube[cube < floor_flux] = floor_flux
        print("New min: ", cube.min())
    elif spectrom.flux_cut_model == 'contrast_threshold':    
        cube_max = np.floor(np.log10(cube.max()))
        cube_min = np.floor(np.log10(cube[cube>0].min()))
        threshold_flux = 10**np.float(cube_min + spectrom.flux_cut[0])
        print(cube_max, cube_min, threshold_flux)
        print("Contrast Cutting a threshold: ", threshold_flux)
        logging.info(f"Contrast cut of: {threshold_flux}")  
        cube[cube > threshold_flux] = threshold_flux
        print("New max: ", cube.max())
    elif spectrom.flux_cut_model == 'contrast_range':    
        cube_max = np.floor(np.log10(cube.max()))
        cube_min = np.floor(np.log10(cube[cube>0].min()))
        #threshold
        threshold_flux = 10**np.float(-spectrom.flux_cut[0])
        print(cube_max, cube_min, threshold_flux)
        print("Contrast Cutting a threshold: ", threshold_flux)
        logging.info(f"Contrast cut of: {threshold_flux}")  
        cube[cube > threshold_flux] = threshold_flux
        #floor        
        cube_max = np.floor(np.log10(cube.max()))
        print("New max: ", cube.max())
        floor_flux = 10**np.float(-spectrom.flux_cut[1])
        print(cube_max, cube_min, floor_flux)
        print("Contrast cutting a floor: ", floor_flux)
        logging.info(f"Contrast cut of: {floor_flux}")  
        cube[cube < floor_flux] = floor_flux
        print("New min: ", cube.min())
    elif spectrom.flux_cut_model == 'flux_max':    
        print("Cutting at threshold: ", spectrom.flux_cut[0])
        logging.info(f"Cutting at threshold: {spectrom.luminosity_cut}")  
        cube[cube > 10**np.float(spectrom.flux_cut[0])] = 10**np.float(spectrom.flux_cut[0])
    elif spectrom.flux_cut_model == 'flux_min':    
        print("Cutting at floor value: ", spectrom.flux_cut[0])
        print("Cutting at floor value: ", 10**np.float(spectrom.flux_cut[0]))
        logging.info(f"Cutting at floor value: {spectrom.luminosity_cut}")  
        cube[cube < 10**np.float(spectrom.flux_cut[0])] = 10**np.float(spectrom.flux_cut[0])


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
