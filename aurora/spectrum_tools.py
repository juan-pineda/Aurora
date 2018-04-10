import gc
import sys
import math
import numpy as np
from scipy import special
import astropy.convolution
from scipy import interpolate
from astropy import constants as const
from scipy.signal import fftconvolve
from bisect import bisect_left

from . import aurora as au
from . import snapshot_tools as snap
from . import gasProps_sBird as bird
from . import datacube as dc
from . import constants as ct

def int_gaussian(x,dx,mu,sigma):
    """
    Compute the integral of a normalized gaussian inside some limits.
    The center and width

    Parameters
    ----------
    x : float, array
        central position of the interval.
    dx : float, array
        width of the interval.
    mu: float, array
        mean of the gaussian.
    sigma: float, array
        standard deviation.
    """

    A = special.erf((x+dx/2-mu)/np.sqrt(2)/sigma)
    B = special.erf((x-dx/2-mu)/np.sqrt(2)/sigma)
    return np.abs((A-B)/2)


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
    cube_side, n_ch = spectrom.cube_dims()
    cube = np.float32(np.zeros((n_ch, cube_side, cube_side, run.nfft)))
    nchunk = int(math.ceil(len(data_gas) / float(run.nvector)))
    au.update_progress(0.0)
    sys.stdout.flush()
    for i in range(nchunk):
        start = i * run.nvector
        stop = start + min(run.nvector, len(data_gas) - start)
        __project_spectrom_flux(geom, run, spectrom, data_gas, start, stop, cube)
        au.update_progress(float(i + 1) / nchunk)
        sys.stdout.flush()
    return cube

def __project_spectrom_flux(geom, run, spectrom, data_gas, start, stop,cube):
    """
    Compute the H-alpha emission of a bunch of particles and project it
    to a 4D grid, keeping contributions from different scales separated

    Parameters
    ----------
    """

    cube_side, n_ch = spectrom.cube_dims()
    # Define the main constants in the right system of units to make it simpler
    # (some operations with sliced vectors were giving the wrong units)
    kpc = const.kpc.to('cm').value
    c = const.c.to('cm/s').value
    Halpha0 = ct.Halpha0.to('angstrom').value
    pixsize = spectrom.pixsize.to('kpc').value
    m_p = ct.m_p.to('g').value
    k_B = ct.k_B.to('g cm2 s-2 K-1').value
    h = ct.h.to('erg s').value

#    if geom.redshift > 0:
#        dl = geom.dl.to('cm').value
#    else:
#        dl = geom.dl.to('pc').value

    # Slice the info for the particles in this chunk
    vect_x      = snap.slice_variable(data_gas,'x',start,stop,'kpc')
    vect_y      = snap.slice_variable(data_gas,'y',start,stop,'kpc')
    vect_z      = snap.slice_variable(data_gas,'z',start,stop,'kpc')
    vect_dens   = snap.slice_variable(data_gas,'rho',start,stop,'g cm**-3')
    vect_vz     = snap.slice_variable(data_gas,'vz',start,stop,'cm s**-1')
    vect_metal  = snap.slice_variable(data_gas,'metal',start,stop)
    vect_temp   = snap.slice_variable(data_gas,'temp',start,stop,'K')
    vect_smooth = snap.slice_variable(data_gas,'smooth',start,stop,'kpc')
    # Pynbody makes some of these arrays 32L, and that causes troubles later on
    vect_x      = np.float64(vect_x)
    vect_y      = np.float64(vect_y)
    vect_z      = np.float64(vect_z)
    vect_vz     = np.float64(vect_vz)
    vect_dens   = np.float64(vect_dens)
    vect_metal  = np.float64(vect_metal)
    vect_smooth = np.float64(vect_smooth)
    vect_temp   = np.float64(vect_temp)
    if 'HII' in data_gas.keys():
        vect_HII = snap.slice_variable(data_gas,'HII',start,stop)
    else:
        vect_HII = np.zeros(stop - start)
        a = bird.GasProperties(spectrom.redshift_ref)
        vect_HII = 1 - a._neutral_fraction(ct.Xh * vect_dens/m_p, vect_temp)
    vect_mu = 4. / (3*ct.Xh + 1 + 4*vect_HII*ct.Xh)

    # We only want to use the ionised-gas cells inside the field of view
    x = (np.floor((vect_x + cube_side * pixsize/2.) / pixsize)).astype(int)
    y = (np.floor((vect_y + cube_side * pixsize/2.) / pixsize)).astype(int)
    ok = np.where((x >= 0) & (x < cube_side) & (y >= 0) & (y < cube_side) &
             (vect_HII > 0.))
    nok = len(ok[0])
    if(nok == 0):
        pass

    vect_x = vect_x[ok]
    vect_y = vect_y[ok]
    vect_z = vect_z[ok]
    vect_vz = vect_vz[ok]
    vect_dens = vect_dens[ok]
    vect_metal = vect_metal[ok]
    vect_temp = vect_temp[ok]
    vect_HII = vect_HII[ok]
    vect_mu = vect_mu[ok]
    vect_smooth = vect_smooth[ok]
    x = x[ok]
    y = y[ok]
    vect_index = x + cube_side * y

    # Velocity dispersion of the emission line, in [cm s-1]
    Halpha_sigma = np.sqrt(k_B * vect_temp / (vect_mu * m_p))

    # Determine the total emitted fluxes. We use case-B Hydrogen effective
    # recombination rate coefficient - Osterbrock & Ferland (2006)
    alphaH = ct.alphaH.to('cm3/s').value * (vect_temp / 1.0e4)**-0.845
    dens_ion = vect_dens * vect_HII * ct.Xh / m_p
    Halpha_lum = (vect_smooth*kpc)**3 * (dens_ion)**2 * (h*c/Halpha0) * alphaH # [erg.cm/A/s]

    # A factor 1e8 is needed to cancel out units [cm/A]
    # But we want to store in units of 1e16, we use a factor 1e-8
    Halpha_flux = Halpha_lum * 1e-8 / spectrom.pixsize.to('pc').value**2

    # Compute the fluxes scale by scale
    for i in range(run.nfft):
        if (i == 0):
            ok_level = np.where(vect_smooth < 1.1 * run.fft_hsml_limits[i])[0]
        elif (i == run.nfft - 1):
            ok_level = np.where(vect_smooth > 1.1 * run.fft_hsml_limits[i-1])[0]
        else:
            ok_level = np.where((vect_smooth < 1.1 * run.fft_hsml_limits[i]) & (
                vect_smooth > 1.1 * run.fft_hsml_limits[i - 1]))[0]
        nok_level = ok_level.size

        if(nok_level == 0):
            continue

        # Unique indices (pixels) to which particles in this group contribute
        unique_val, unique_ind = np.unique(vect_index[ok_level], return_index=True)

        # Retain only line centers/broadenings for particles in this group,
        # arranged in a matrix where each row is a particle, and columns
        # will serve to store fluxes at each of the cube spectral channels, e.g,
        # with n particles centered at l1, l2 ... Halpha_obs_level is:
            # [ l1 l1 l1 ... l1
            #   l2 l2 l2 ... l2
            #   .  .  .  ...
            #   .  .  .  ...
            #   ln ln ln ... ln]

        Ha_obs_level = np.transpose(np.tile(vect_vz[ok_level], (n_ch, 1))) # Now in velocity!
        Ha_sigma_level = np.transpose(np.tile(Halpha_sigma[ok_level], (n_ch, 1)))
        Ha_flux_level = np.transpose(np.tile(Halpha_flux[ok_level], (n_ch, 1)))

        # Spectral convolution
        if(spectrom.spectral_res > 0):
            psf_fwhm = c/spectrom.spectral_res
            psf_sigma = psf_fwhm / ct.fwhm_sigma
            Ha_sigma_level = np.sqrt(Ha_sigma_level**2+psf_sigma**2)

        # Emission line array creation. Flux in erg.s^-1.cm^-2.microns^-1
        line = np.tile(spectrom.vel_channels.to('cm s-1').value, (nok_level, 1))
        ch_width = np.ones([nok_level,n_ch])
        ch_width *= spectrom.velocity_sampl.to('cm s-1').value
        # Integrated flux inside each velocity channel given its position and width
        line_Ha = int_gaussian(line, ch_width, Ha_obs_level, Ha_sigma_level) * Ha_flux_level

        # Divide by the effective channel width
        line = line_Ha / spectrom.velocity_sampl.to('km s^-1').value

        # Sum all the lines for a given index
        for j in range(unique_val.size):
            to_sum = np.where(vect_index[ok_level] == unique_val[j])[0]
            line[unique_ind[j], :] = np.sum(line[to_sum, :], axis=0)
        # Remove duplicated emission lines
        line = line[unique_ind, :]
        # Insert the line fluxes in the right positions at the right scale
        cube[:, y[ok_level[unique_ind]], x[ok_level[unique_ind]], i] += np.float32(np.transpose(line))

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
        print ("Preparing for spatial smoothing, kernel = ", round(run.fft_hsml_limits[i]*1000,1)," pc")
        sys.stdout.flush()
        # Kernel smoothing
        scale_fwhm = run.fft_hsml_limits[i] / spectrom.pixsize.to('kpc').value
        scale_sigma = spectrom.kernel_scale * scale_fwhm / ct.fwhm_sigma
        print ("Size of the kernel in pixels = ", round(scale_sigma,1))
        # Enlarge the kernel adding the effect of the PSF
        if(spectrom.spatial_res_kpc > 0):
            print (" (Including the effect of the PSF as well)")
####            psf_fwhm = spectrom.spatial_res.value / spectrom.spatial_sampl.value
            psf_fwhm = spectrom.spatial_res_kpc.to('pc').value / spectrom.pixsize.to('pc').value
            psf_sigma = psf_fwhm / ct.fwhm_sigma
            print ("Size of the PSF in pixels = ",round(psf_sigma,1))
            scale_sigma = np.sqrt(scale_sigma**2+psf_sigma**2)
            print ("Combination kernel + PSF in pixels = ",round(scale_sigma,1))
        if (scale_sigma <= 0.5):
            print ("-- Small kernel -> skip convolution")
            sys.stdout.flush()
            continue

####### THESE LINES ARE A TEST !  I WILL TRY TO USE MY CONVOLUTION SCHEME TO SEE IF IT IS FASTER
####### 24/05 IN THE NIGHT

        m = dc.DatacubeObj()
        m.cube = cube[:,:,:,i]
        psf_fwhm = scale_sigma * ct.fwhm_sigma
        __spatial_convolution_iter(m,psf_fwhm)
        cube[:,:,:,i] = m.cube

#        for j in range(n_ch):
#            if (np.nanmax(cube[j, :, :, i]) == 0):
#                print "No flux at this scale/velocity channel -> skip convolution"
#                continue
#            psf = astropy.convolution.Gaussian2DKernel(scale_sigma)
#            channel = np.float32(astropy.convolution.convolve_fft(cube[j,:,:,i],
#                    psf,psf_pad=True,normalize_kernel=np.sum,allow_huge=True))
#            cube[j, :, :, i] = 0.
#            cube[j, :, :, i] += channel

# FORMER LINES ARE MY TESTED SCHEME, LETS TRY THE NEW ONE AND SEE IF IT WORKS FINE
#########################################

def __fft_spatial_convolution(m,psf_fwhm):
    psf_sigma = psf_fwhm / ct.fwhm_sigma
    psf = astropy.convolution.Gaussian2DKernel(psf_sigma)
    psf = np.array(psf)
    psf = psf / psf.sum()

#    fshape = int(2**np.ceil(np.log2(m.cube.shape[1]+psf.shape[0])))
    fshape = next_fast_len(m.cube.shape[1]+psf.shape[0])
    center = fshape - (fshape+1) // 2

    new_psf = np.zeros([fshape, fshape])
    index = slice(center - psf.shape[0] // 2, center + (psf.shape[0] + 1) // 2)
    new_psf[index,index] = psf
    new_psf = new_psf.reshape(1,new_psf.shape[0],new_psf.shape[1])
    psf = np.fft.fftshift(new_psf)
    psf = np.fft.rfft2(psf)

    index = slice(center - m.cube.shape[1] // 2, center + (m.cube.shape[1] + 1) // 2)

    lead_zeros = np.zeros([m.cube.shape[0], center - m.cube.shape[1] // 2, m.cube.shape[2]])
    trail_zeros = np.zeros([m.cube.shape[0], fshape - m.cube.shape[1] - lead_zeros.shape[1] , m.cube.shape[2]])
    m.cube = np.concatenate((lead_zeros, m.cube, trail_zeros),axis=1)
    lead_zeros = np.zeros([m.cube.shape[0], m.cube.shape[1], center - m.cube.shape[2] // 2])
    trail_zeros = np.zeros([m.cube.shape[0], m.cube.shape[1], fshape - lead_zeros.shape[2] - m.cube.shape[2]])
    m.cube = np.concatenate((lead_zeros, m.cube, trail_zeros),axis=2)
    m.cube = np.fft.rfft2(m.cube)

    m.cube = m.cube * psf
    m.cube = np.fft.irfft2(m.cube)
    m.cube = np.float32(m.cube[:,index,index])


def __fft_spectral_convolution(m,psf_fwhm):
    psf_sigma = psf_fwhm / ct.fwhm_sigma
    psf = astropy.convolution.Gaussian1DKernel(psf_sigma)
    psf = np.array(psf)
    psf = psf / psf.sum()

#    fshape = int(2**np.ceil(np.log2(m.cube.shape[0]+psf.shape[0])))
    fshape = next_fast_len(m.cube.shape[0]+psf.shape[0])
    center = fshape - (fshape+1) // 2

    lead_zeros = np.zeros(center - psf.shape[0] // 2)
    trail_zeros = np.zeros(fshape - psf.shape[0] - lead_zeros.shape[0])
    psf = np.concatenate((lead_zeros, psf, trail_zeros),axis=0)
    psf = np.fft.fftshift(psf)
    psf = psf.reshape(psf.size,1,1)
    psf = np.fft.rfft(psf,axis=0)

    index = slice(center - m.cube.shape[0] // 2,center + (m.cube.shape[0] + 1) // 2)
    lead_zeros = np.zeros([center - m.cube.shape[0] // 2, m.cube.shape[1], m.cube.shape[2]])
    trail_zeros = np.zeros([fshape - m.cube.shape[0] - lead_zeros.shape[0], m.cube.shape[1], m.cube.shape[2]])
    m.cube = np.concatenate((lead_zeros, m.cube, trail_zeros),axis=0)
    m.cube = np.fft.rfft(m.cube, axis=0)

    m.cube = m.cube * psf
    m.cube = np.fft.irfft(m.cube, fshape, axis=0)
    m.cube = np.float32(m.cube[index,:,:])


def __spectral_convolution_iter(m,psf_fwhm):
    psf_sigma = psf_fwhm / ct.fwhm_sigma
    psf = astropy.convolution.Gaussian2DKernel(psf_sigma)
    psf = np.array(psf)
    center = int(psf.shape[0]/2) # this is the position of the central line
    psf[:,0:center] = 0
    psf[:,center+1:] = 0
    psf = psf / psf.sum()

    # Padding to a square;
#    fshape = int(2**np.ceil(np.log2(np.max([m.cube.shape[0], m.cube.shape[1]])+psf.shape[0])))
    fshape = next_fast_len(np.max([m.cube.shape[0], m.cube.shape[1]])+psf.shape[0])
    center = fshape - (fshape+1) // 2 # This is the center of the enlarged array

    new_psf = np.zeros([fshape, fshape])
    index = slice(center - psf.shape[0] // 2, center + (psf.shape[0] + 1) // 2)
    new_psf[index,index] = psf
    psf = np.fft.fftshift(new_psf)
    psf = np.fft.rfft2(psf)

    index_0 = slice(center - m.cube.shape[0] // 2,center + (m.cube.shape[0] + 1) // 2)
    index_1 = slice(center - m.cube.shape[1] // 2,center + (m.cube.shape[1] + 1) // 2)

    for i in range(m.cube.shape[2]):
        channel = np.zeros([fshape, fshape])
        channel[index_0,index_1] = m.cube[:,:,i]
        channel = np.fft.rfft2(channel)
        channel = channel * psf
        channel = np.fft.irfft2(channel)
        m.cube[:,:,i] = np.float32(channel[index_0,index_1]) # NO ES CUADRADA !!! :O


def __spatial_convolution_iter(m,psf_fwhm):
    psf_sigma = psf_fwhm / ct.fwhm_sigma
    psf = astropy.convolution.Gaussian2DKernel(psf_sigma)
    psf = np.array(psf)
    psf = psf / psf.sum()

#    fshape = int(2**np.ceil(np.log2(np.max([m.cube.shape[0], m.cube.shape[1]])+psf.shape[0])))
    fshape = next_fast_len(m.cube.shape[1]+psf.shape[0])
    center = fshape - (fshape+1) // 2 # This is the center of the enlarged array

    new_psf = np.zeros([fshape, fshape])
    index = slice(center - psf.shape[0] // 2, center + (psf.shape[0] + 1) // 2)
    new_psf[index,index] = psf
    psf = np.fft.fftshift(new_psf)
    psf = np.fft.rfft2(psf)

    index = slice(center - m.cube.shape[1] // 2,center + (m.cube.shape[1] + 1) // 2)

    for i in range(m.cube.shape[0]):
        channel = np.zeros([fshape, fshape])
        channel[index,index] = m.cube[i,:,:]
        channel = np.fft.rfft2(channel)
        channel = channel * psf
        channel = np.fft.irfft2(channel)
        m.cube[i,:,:] = np.float32(channel[index,index])


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
