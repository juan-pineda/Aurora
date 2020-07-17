import numpy as np
import logging

import astropy.convolution
from bisect import bisect_left

from . import constants as ct


def next_fast_len(target):
    hams = (8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48,
            50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128,
            135, 144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 250,
            256, 270, 288, 300, 320, 324, 360, 375, 384, 400, 405, 432, 450,
            480, 486, 500, 512, 540, 576, 600, 625, 640, 648, 675, 720, 729,
            750, 768, 800, 810, 864, 900, 960, 972, 1000, 1024, 1080, 1125,
            1152, 1200, 1215, 1250, 1280, 1296, 1350, 1440, 1458, 1500, 1536,
            1600, 1620, 1728, 1800, 1875, 1920, 1944, 2000, 2025, 2048, 2160,
            2187, 2250, 2304, 2400, 2430, 2500, 2560, 2592, 2700, 2880, 2916,
            3000, 3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750, 3840,
            3888, 4000, 4050, 4096, 4320, 4374, 4500, 4608, 4800, 4860, 5000,
            5120, 5184, 5400, 5625, 5760, 5832, 6000, 6075, 6144, 6250, 6400,
            6480, 6561, 6750, 6912, 7200, 7290, 7500, 7680, 7776, 8000, 8100,
            8192, 8640, 8748, 9000, 9216, 9375, 9600, 9720, 10000)

    if target <= 6:
        return target

    # Quickly check if it"s already a power of 2
    if not (target & (target-1)):
        return target

    # Get result quickly for small sizes, since FFT itself is similarly fast.
    if target <= hams[-1]:
        return hams[bisect_left(hams, target)]

    match = float("inf")  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            p2 = 2**((quotient - 1).bit_length())

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match

def next_odd(x):
    x = np.ceil(x)
    if x % 2 == 1:
        return x
    else:
        return x+1
    
# Kernel create    

def create_psf(spectom, scale_sigma, size = 20):
    # Enlarge the kernel adding the effect of the PSF
    if(spectrom.spatial_res_kpc > 0):
        logging.info(f" (Including the effect of the PSF as well)")
        psf_fwhm = spectrom.spatial_res_kpc.to(
            "pc").value / spectrom.pixsize.to("pc").value
        psf_sigma = psf_fwhm / ct.fwhm_sigma
        logging.info(f"Size of the PSF in pixels = {round(psf_sigma, 1)}")
        scale_sigma = np.sqrt(scale_sigma**2+psf_sigma**2)
        logging.info(f"Combination kernel + PSF in pixels = {round(scale_sigma, 1)}")
    psf = astropy.convolution.Gaussian2DKernel(scale_sigma,
          x_size = next_odd(size * scale_sigma), y_size = next_odd(size * scale_sigma))
    psf = np.array(psf)
    psf = psf / psf.sum()
    return (psf, scale_sigma)

def create_lsf(lsf_fwhm, size = 20):
    lsf_sigma = lsf_fwhm / ct.fwhm_sigma
    lsf = astropy.convolution.Gaussian1DKernel(lsf_sigma, x_size = cv.next_odd(size * lsf_sigma))
    lsf = np.array(lsf)
    lsf = lsd / lsf.sum()
    return lsf

# Spatial convolutions

def spatial_astropy_convolution(cube, psf):
    n_ch = cube.shape[0]
    for j in range(n_ch):
        if (np.nanmax(cube[j, :, :]) == 0):
            logging.info(f"No flux at this scale/velocity channel -> skip convolution")
            continue
        cube[j, :, :] = astropy.convolution.convolve(cube[j,:,:],psf)
    return cube

def fft_spatial_astropy_convolution(cube, psf):   
    n_ch = cube.shape[0]
    for j in range(n_ch):
        if (np.nanmax(cube[j, :, :]) == 0):
            logging.info(f"No flux at this scale/velocity channel -> skip convolution")
            continue
        cube[j, :, :] = astropy.convolution.convolve_fft(cube[j, :, :], psf, psf_pad = True, 
                                                   fft_pad = True, allow_huge=True) 
    return cube

def fft_spatial_aurora_convolution(cube, psf):
    x, y, z = cube.shape
    
    fshape = next_fast_len(y + psf.shape[0])
    center = fshape - (fshape+1) // 2
    new_psf = np.zeros([1, fshape, fshape])
    index = slice(center - psf.shape[0] // 2, center + (psf.shape[0] + 1) // 2)
    new_psf[0,index, index] = psf
    
    psf = np.fft.fftshift(new_psf)
    psf = np.fft.rfft2(psf)
    
    index = slice(center - y // 2, center + (y + 1) // 2)
    
    lead_zeros = np.zeros([x, center - y // 2, z])
    trail_zeros = np.zeros([x, fshape - y - lead_zeros.shape[1], z])
    cube = np.concatenate((lead_zeros, cube, trail_zeros), axis=1)
    lead_zeros = np.zeros([x, cube.shape[1], center - z // 2])
    trail_zeros = np.zeros([x, cube.shape[1], fshape - lead_zeros.shape[2] - z])
    cube = np.concatenate((lead_zeros, cube, trail_zeros), axis=2)

    cube = np.fft.rfft2(cube)
    cube = cube * psf
    cube = np.fft.irfft2(cube)
    cube = cube[:, index, index]
    
    return cube

# Spectral convolutions

def spectral_astropy_convolution(cube, lsf):
    x, y, z = cube.shape
    
    for j in range(y):
        for i in range(z):
            cube[:, j, i] = astropy.convolution.convolve(cube[:,j,i],lsf)
    return cube


def fft_spectral_astropy_convolution(cube, lsf):     
    x, y, z = cube.shape
    
    for j in range(y):
        for i in range(z):
            cube[:, j, i] = astropy.convolution.convolve_fft(cube[:, j, i], lsf, psf_pad = True, 
                                                   fft_pad = True, allow_huge=True)
    return cube

def fft_spectral_aurora_convolution(cube, lsf):
    fshape = cv.next_fast_len(cube.shape[0]+lsf.shape[0])
    center = fshape - (fshape+1) // 2

    lead_zeros = np.zeros(center - lsf.shape[0] // 2)
    trail_zeros = np.zeros(fshape - lsf.shape[0] - lead_zeros.shape[0])
    lsf = np.concatenate((lead_zeros, lsf, trail_zeros), axis=0)
    lsf = np.fft.fftshift(lsf)
    lsf = psf.reshape(lsf.size, 1, 1)
    lsf = np.fft.rfft(lsf, axis=0)

    index = slice(center - cube.shape[0] //
                  2, center + (cube.shape[0] + 1) // 2)
    lead_zeros = np.zeros([center - cube.shape[0] // 2,
                           cube.shape[1], cube.shape[2]])
    trail_zeros = np.zeros(
        [fshape - cube.shape[0] - lead_zeros.shape[0], cube.shape[1], cube.shape[2]])
    cube = np.concatenate((lead_zeros, cube, trail_zeros), axis=0)
    cube = np.fft.rfft(cube, axis=0)

    cube = cube * lsf
    cube = np.fft.irfft(cube, fshape, axis=0)
    cube = cube[index, :, :]

    return cube
