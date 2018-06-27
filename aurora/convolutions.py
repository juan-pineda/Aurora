import numpy as np
import astropy.convolution
from bisect import bisect_left
from . import convolutions as cv

from . import constants as ct


def fft_spatial_convolution(cube, psf_fwhm):
    # Get cube dimensions and create the psf
    x, y, z = cube.shape
    psf = create_psf(psf_fwhm)

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
    lead_zeros = np.zeros([x, y, center - z // 2])
    trail_zeros = np.zeros([x, y, fshape - lead_zeros.shape[2] - z])
    cube = np.concatenate((lead_zeros, cube, trail_zeros), axis=2)

    cube = np.fft.rfft2(cube)
    cube = cube * psf
    cube = np.fft.irfft2(cube)
    cube = cube[:, index, index]

    return cube


# Aqui no puedo usar la funcion create_psf por que la normalizacion sobre la suma
# ocurre despues de crear el array 2D y modificar algunos de sus valores
def fft_spectral_convolution(m, psf_fwhm):
    psf_sigma = psf_fwhm / ct.fwhm_sigma
    psf = astropy.convolution.Gaussian1DKernel(psf_sigma)
    psf = np.array(psf)
    psf = psf / psf.sum()

#    fshape = int(2**np.ceil(np.log2(m.cube.shape[0]+psf.shape[0])))
    fshape = next_fast_len(m.cube.shape[0]+psf.shape[0])
    center = fshape - (fshape+1) // 2

    lead_zeros = np.zeros(center - psf.shape[0] // 2)
    trail_zeros = np.zeros(fshape - psf.shape[0] - lead_zeros.shape[0])
    psf = np.concatenate((lead_zeros, psf, trail_zeros), axis=0)
    psf = np.fft.fftshift(psf)
    psf = psf.reshape(psf.size, 1, 1)
    psf = np.fft.rfft(psf, axis=0)

    index = slice(center - m.cube.shape[0] //
                  2, center + (m.cube.shape[0] + 1) // 2)
    lead_zeros = np.zeros([center - m.cube.shape[0] // 2,
                           m.cube.shape[1], m.cube.shape[2]])
    trail_zeros = np.zeros(
        [fshape - m.cube.shape[0] - lead_zeros.shape[0], m.cube.shape[1], m.cube.shape[2]])
    m.cube = np.concatenate((lead_zeros, m.cube, trail_zeros), axis=0)
    m.cube = np.fft.rfft(m.cube, axis=0)

    m.cube = m.cube * psf
    m.cube = np.fft.irfft(m.cube, fshape, axis=0)
    m.cube = m.cube[index, :, :]


def __spectral_convolution_iter(m, psf_fwhm):
    psf_sigma = psf_fwhm / ct.fwhm_sigma
    psf = astropy.convolution.Gaussian2DKernel(psf_sigma)
    psf = np.array(psf)
    center = int(psf.shape[0]/2)  # this is the position of the central line
    psf[:, 0:center] = 0
    psf[:, center+1:] = 0
    psf = psf / psf.sum()

    # Padding to a square;
#    fshape = int(2**np.ceil(np.log2(np.max([m.cube.shape[0], m.cube.shape[1]])+psf.shape[0])))
    fshape = next_fast_len(
        np.max([m.cube.shape[0], m.cube.shape[1]])+psf.shape[0])
    # This is the center of the enlarged array
    center = fshape - (fshape+1) // 2

    new_psf = np.zeros([fshape, fshape])
    index = slice(center - psf.shape[0] // 2, center + (psf.shape[0] + 1) // 2)
    new_psf[index, index] = psf
    psf = np.fft.fftshift(new_psf)
    psf = np.fft.rfft2(psf)

    index_0 = slice(
        center - m.cube.shape[0] // 2, center + (m.cube.shape[0] + 1) // 2)
    index_1 = slice(
        center - m.cube.shape[1] // 2, center + (m.cube.shape[1] + 1) // 2)

    for i in range(m.cube.shape[2]):
        channel = np.zeros([fshape, fshape])
        channel[index_0, index_1] = m.cube[:, :, i]
        channel = np.fft.rfft2(channel)
        channel = channel * psf
        channel = np.fft.irfft2(channel)
        m.cube[:, :, i] = channel[index_0, index_1]  # NO ES CUADRADA !!! :O

####################################################################
#
# En au.spectrom_mock los metodos que siguen son los que uso;
# Todos los demás necesitan una revisión/mejora
#
####################################################################

def create_psf(psf_fwhm):
    psf_sigma = psf_fwhm / ct.fwhm_sigma
    psf = astropy.convolution.Gaussian2DKernel(psf_sigma)
    psf = np.array(psf)
    psf = psf / psf.sum()
    return psf

def spatial_convolution_iter(cube, psf_fwhm):
    # Get cube dimensions and create the psf
    x, y, z = cube.shape
    psf = create_psf(psf_fwhm)
    # Define the shape and center of the enlarged array
    fshape = next_fast_len(y + psf.shape[0])
    center = fshape - (fshape+1) // 2
    # create a zero-padded psf with the new dimensions
    new_psf = np.zeros([fshape, fshape])
    index = slice(center - psf.shape[0] // 2, center + (psf.shape[0] + 1) // 2)
    new_psf[index, index] = psf
    # take the fft of the enlarged psf
    psf = np.fft.fftshift(new_psf)
    psf = np.fft.rfft2(psf)
    # Define the index covering the psf data in the zero-padded version
    index = slice(center - y //2, center + (y + 1) // 2)
    # Perform a slice-by-slice spatial convolution
    for i in range(x):
        channel = np.zeros([fshape, fshape])
        channel[index, index] = cube[i, :, :]
        channel = np.fft.rfft2(channel)
        channel = channel * psf
        channel = np.fft.irfft2(channel)
        cube[i, :, :] = channel[index, index]
    return cube


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

    # Quickly check if it's already a power of 2
    if not (target & (target-1)):
        return target

    # Get result quickly for small sizes, since FFT itself is similarly fast.
    if target <= hams[-1]:
        return hams[bisect_left(hams, target)]

    match = float('inf')  # Anything found will be smaller
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







