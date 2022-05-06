"""
========
datacube
========

This module contains the methods for constructing kinematic maps from a 
realistic datacube.
"""

import os
import re
import sys
import math
import numpy as np
import configparser
from astropy.io import fits
from astropy import units as unit
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm
from astropy.cosmology import Planck13 as cosmo

from . import presets
from . import constants as ct
from . import set_output as so
from . import spectrum_tools as spec
from . import configuration as config
from . import array_operations as arr

from pylab import *


class DatacubeObj():
    """
    This class handles the relevant information and operations related
    to the datacubes.
    """

    def __init__(self):
        pass

    def read_data(self, input_file):
        """
        Read the data and header of the specified datacube in fits format
        
        Parameters
        ----------        
        input_file : str
            Datacube name in fits format
        """

        hdulist = fits.open(input_file)
        self.cube = hdulist[0].data
        self.header = hdulist[0].header

    def get_attr(self):
        """
        Extract main information from the cards in the header
        """
        self.pixsize = self.header["CDELT1"] * unit.pc
        self.velocity_sampl = self.header["CDELT3"] * unit.km/unit.s
        self.spatial_dim = self.header["NAXIS1"]
        self.spectral_dim = self.header["NAXIS3"]
        self.fieldofview = self.spatial_dim * self.pixsize.to("kpc")
        self.velocity_range = self.spectral_dim * \
            self.velocity_sampl.to("km s^-1")
        self.channel_ref = self.header["CRPIX3"] - 1
        self.vel_ref = self.header["CRVAL3"] * unit.km/unit.s
        self.pixel_ref = self.header["CRPIX1"] - 1
        self.position_ref = self.header["CRVAL1"] * unit.pc
        self.HSIM3 = self.header["HSIM3"]
        self.channels = (np.arange(self.spectral_dim) - self.channel_ref) * (
            self.velocity_sampl.to("km s^-1").value) + self.vel_ref.to("km s^-1").value

    def get_attr_HSIM3(self):
        """
        Extract main information from the cards in the header for 
        data cube 
        """
        self.spatial_unit = self.header["CUNIT1"] 
        self.spatial_sampl = (self.header["CDELT1"] * unit.Unit(self.spatial_unit)).to('arcsec')
        self.spectral_unit = self.header["CUNIT3"] 
        self.spectral_sampl = (self.header["CDELT3"] * unit.Unit(self.spectral_unit)).to('AA') 
        self.spatial_dim = self.header["NAXIS1"]
        self.spectral_dim = self.header["NAXIS3"]
        self.FoV = self.spatial_dim * self.spatial_sampl.to("arcsec")
        self.spectral_range = self.spectral_dim * \
            self.spectral_sampl.to("AA")
        self.channel_ref = self.header["CRPIX3"] - 1
        self.lambda_obs = (self.header["CRVAL3"] * unit.Unit(self.spectral_unit)).to('AA') 
        self.pixel_ref = self.header["CRPIX1"] - 1
        self.position_ref = self.header["CRVAL1"] * unit.pc
        self.HSIM3 = self.header["HSIM3"]
        self.redshift =  (self.lambda_obs/ct.Halpha0) - 1
        self.dl = cosmo.luminosity_distance(self.redshift)
        self.dist_angular = cosmo.angular_diameter_distance(self.redshift)
        self.pixsize = self.spatial_sampl.to("rad").value * self.dist_angular.to("kpc")
        self.fieldofview = self.spatial_dim * self.pixsize.to("kpc")
        self.velocity_sampl = ct.c.to("km s-1") * self.spectral_sampl.to("angstrom").value / (
            self.lambda_obs.to("angstrom").value)
        self.vel_ref = 0. * unit.km / unit.s
        self.BUNIT = self.header["BUNIT"]
        self.channels = (np.arange(self.spectral_dim) - self.channel_ref) * (
            self.velocity_sampl.to("km s^-1").value) + self.vel_ref.to("km s^-1").value

        
    def assign_attr(self, pixsize, velocity_sampl, spatial_dim, spectral_dim):
        """
        Assign entered attributes.
        
        Parameters
        ----------  
        pixsize : float
            Pixel size of the instrument in (pc).
        velocity_sampl : float
            Spectral sampling of the instrument in velocity units
            (km s-1).
        spatial_dim : int
            Number of pixels per side of the field of view.
        spectral_dim : int
            Number of spectral channels of the instrument.                        
        """
        self.pixsize = pixsize * unit.pc
        self.velocity_sampl = velocity_sampl * unit.km/unit.s
        self.spatial_dim = spatial_dim
        self.spectral_dim = spectral_dim
        self.channels = np.arange(-spectral_dim/2 * velocity_sampl, spectral_dim/2 
                        * velocity_sampl, velocity_sampl)
       
    
    def get_one_channel(self, index):
        v_channel = (index - self.channel_ref)*self.velocity_sampl.to("km s-1")
        return v_channel + self.vel_ref.to("km s-1")

    def get_one_position(self, index):
        position = (index - self.pixel_ref) * self.pixsize.to("pc")
        return position + self.position_ref.to("pc")

    def spatial_degrade(self, geom, spectrom):
        """

        Parameters
        ----------

        WARNING !!!

        For the time being, this function bin the mastercube, and update
        the right instrumental values in the object *spectrom*, so the
        new cube can be stored, BUT is NOT updating the keywrds/header
        of the mastercube object itself!!!
        DO NOT use those parameters beyond this point !!!
        """
        # Code flow:
        # =====================
        # > Estimate binning factor, nx, and set the number of spatial
        # > pixels, N_pixels, as the maximum integer multiple of nx
        # > Cut a centered nx-wide portion of the array
        # > Bin in the spatial direction
        nx = int(spectrom.pixsize / self.pixsize)
        spectrom.pixsize = self.pixsize * nx
        spectrom.spatial_dim = int(self.spatial_dim / nx)
        spectrom.fieldofview = spectrom.pixsize * spectrom.spatial_dim
        N_pixels = nx * spectrom.spatial_dim
        shift = int((self.spatial_dim - N_pixels) / 2)
        self.cube = self.cube[:, shift:shift +
                              N_pixels, shift:shift + N_pixels]
        self.cube = arr.bin_array(self.cube, nx, axis=1)
        self.cube = arr.bin_array(self.cube, nx, axis=2)

        # > update the necessary keywords in the header
        spectrom.check_pixsize(geom)
        spectrom.check_fieldofview(geom)

        # > Determine the central position in the new spatial grid
        pos_low = self.get_one_position(shift)
        pos_hi = self.get_one_position(shift + N_pixels - 1)
        spectrom.position_ref = (pos_low.to("pc")+pos_hi.to("pc"))/2.
        spectrom.pixel_ref = (spectrom.spatial_dim - 1) / 2.

    def spectral_degrade(self, geom, spectrom):
        """

        Parameters
        ----------

        WARNING !!!

        For the time being, this function bin the mastercube, and update
        the right instrumental values in the object *spectrom*, so the
        new cube can be stored, BUT is NOT updating the keywrds/header
        of the mastercube object itself!!!
        DO NOT use those parameters beyond this point !!!
        """

        # Code flow:
        # =====================
        # > Estimate binning factor, nx, and set the number of spectral
        # > pixels, N_pixels, as the maximum integer multiple of nx
        # > Cut a centered nx-long portion of the array
        # > Bin in the spectral direction
        nx = int(spectrom.velocity_sampl / self.velocity_sampl)
        spectrom.velocity_sampl = self.velocity_sampl * nx
        spectrom.spectral_dim = int(self.spectral_dim / nx)
        spectrom.velocity_range = spectrom.velocity_sampl * spectrom.spectral_dim
        N_pixels = nx * spectrom.spectral_dim
        shift = int((self.spectral_dim - N_pixels) / 2)
        self.cube = self.cube[shift:shift + N_pixels, :, :]
        self.cube = arr.bin_array(self.cube, nx, axis=0)

        # > update the necessary keywords in the header
        spectrom.check_velocity_sampl(geom)
        spectrom.check_velocity_range(geom)
        spectrom.set_channels(geom)

        # > Determine the central velocity in the new reference channel
        vel_low = self.get_one_channel(
            shift+int(spectrom.spectral_dim/2.) * nx)
        vel_hi = self.get_one_channel(
            shift+int(spectrom.spectral_dim/2.) * nx + nx - 1)
        spectrom.vel_ref = (vel_low.to("km s-1")+vel_hi.to("km s-1"))/2.
        spectrom.channel_ref = int(spectrom.spectral_dim/2)

    def intensity_map(self):
        """
        Generates the fluxmap from the realistic datacube (3D). 
                
        Returns
        -------
        fluxmap : ndarray (2D)
            Fluxmap in erg s-1.
        """
        if self.HSIM3 == True:
            self.fluxmap = self.cube.sum(axis=0)  * \
                self.spectral_sampl.to("um") *  unit.Unit(self.BUNIT)
        else:
            self.fluxmap = self.cube.sum(axis=0) * \
                self.velocity_sampl.to("km s-1").value * unit.erg / unit.s / unit.cm**2

    def velocity_map(self):
        """
        Generates the velocity map from the realistic datacube (3D). 
                
        Returns
        -------
        velmap : ndarray (2D)
            Velocity map in km s-1.
        """
        if not hasattr(self,"fluxmap"):
            self.intensity_map()
        velmap = np.zeros(self.fluxmap.shape)
        for i in range(self.channels.size):
            if self.HSIM3 == True:
                velmap = velmap + \
                self.cube[i, :, :] * self.channels[i] * \
                self.spectral_sampl.to("um").value
            else:
                velmap = velmap + \
                self.cube[i, :, :] * self.channels[i] * \
                self.velocity_sampl.to("km s-1").value

        self.velmap = velmap / self.fluxmap.value

    def dispersion_map(self):
        """
        Generates the velocity dispersion map from the realistic datacube (3D). 
                
        Returns
        -------
        dispmap : ndarray (2D)
            Velocity dispersion map in km s-1.
        """
        if not hasattr(self,"fluxmap"):
            self.intensity_map()
        if not hasattr(self,"velmap"):
            self.velocity_map()
        disper = np.zeros(self.fluxmap.shape)
        for i in range(self.channels.size):
            disper += self.cube[i, :, :]*(self.channels[i] - self.velmap)**2
        if self.HSIM3 == True:
            self.dispmap = np.sqrt(disper * self.spectral_sampl.to("um").value / self.fluxmap.value)
        else:
            self.dispmap = np.sqrt(disper * self.velocity_sampl.to("km s-1").value / self.fluxmap.value)

    def all_maps(self):
        """
        Generates the kinematic maps. from the realistic datacube (3D). 
                
        Returns
        -------
        fluxmap : ndarray (2D)
            Fluxmap in erg s-1.
        velmap : ndarray (2D)
            Velocity map in km s-1.
        dispmap : ndarray (2D)
            Velocity dispersion map in km s-1.
        """
        self.intensity_map()
        self.velocity_map()
        self.dispersion_map()

    def plot_intensity_map(self, cmap = None, vmin=None, vmax=None,units_spacial = 'arcsec'):
        if units_spacial == 'arcsec':
            X = self.FoV.to('arcsec').value
        elif units_spacial == 'kpc':
            X = self.fieldofview.to('kpc').value
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        p = plt.imshow(self.fluxmap.value, cmap=cmap, norm=LogNorm(), extent = [-X/2,X/2,-X/2,X/2],
                  vmin=vmin, vmax=vmax)

        cbar = fig.colorbar(p, ax=ax)
        cbar.ax.get_yaxis().labelpad = 25
        cbar.ax.set_ylabel(r'Flux [{}]'.format(str(self.fluxmap.unit)), rotation=-270, fontsize=20)
        cbar.ax.tick_params(labelsize=23)
        plt.ylabel('\nY  [{}]'.format(units_spacial), fontsize = 25)
        plt.xlabel('\nX  [{}]'.format(units_spacial), fontsize = 25)
        plt.xticks(size = 18)
        plt.yticks(size = 18)
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))

    def plot_velocity_map(self, cmap = None, units_spacial = 'arcsec'):
        if units_spacial == 'arcsec':
            X = self.FoV.to('arcsec').value
        elif units_spacial == 'kpc':
            X = self.fieldofview.to('kpc').value
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        p = plt.imshow(self.velmap, cmap=cmap, extent = [-X/2,X/2,-X/2,X/2])

        cbar = fig.colorbar(p, ax=ax)
        cbar.ax.get_yaxis().labelpad = 25
        cbar.ax.set_ylabel(r'$V_{los} \ [km \ s^{-1}]$', rotation=-270, fontsize=20)
        cbar.ax.tick_params(labelsize=23)
        plt.ylabel('\nY  [{}]'.format(units_spacial), fontsize = 25)
        plt.xlabel('\nX  [{}]'.format(units_spacial), fontsize = 25)
        plt.xticks(size = 18)
        plt.yticks(size = 18)
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))

    def plot_dispersion_map(self, cmap = None, units_spacial = 'arcsec'):
        if units_spacial == 'arcsec':
            X = self.FoV.to('arcsec').value
        elif units_spacial == 'kpc':
            X = self.fieldofview.to('kpc').value
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        p = plt.imshow(self.dispmap, cmap=cmap, extent = [-X/2,X/2,-X/2,X/2])

        cbar = fig.colorbar(p, ax=ax)
        cbar.ax.get_yaxis().labelpad = 25
        cbar.ax.set_ylabel(r'$V_{los} \ [km \ s^{-1}]$', rotation=-270, fontsize=20)
        cbar.ax.tick_params(labelsize=23)
        plt.ylabel('\nY  [{}]'.format(units_spacial), fontsize = 25)
        plt.xlabel('\nX  [{}]'.format(units_spacial), fontsize = 25)
        plt.xticks(size = 18)
        plt.yticks(size = 18)
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))

    def clean_lowflux(self, thresh = 11):
        """
        Clear the background flux on the flux map, velocity map, 
        and dispersion map. It only stores the values greater 
        than threshold.
        
        Parameters
        ----------  
        thresh : int, opcional
            Threshold to store the values.
        """
        zeros = (np.log10(np.max(self.fluxmap)) - np.log10(self.fluxmap) > thresh)
        self.fluxmap[zeros] = np.nan
        self.velmap[zeros] = np.nan
        self.dispmap[zeros] = np.nan
