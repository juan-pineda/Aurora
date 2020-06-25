import os
import re
import sys
import math
import numpy as np
import configparser
from astropy.io import fits
from astropy import units as unit
from astropy.cosmology import Planck13 as cosmo

from . import presets
from . import constants as ct
from . import set_output as so
from . import spectrum_tools as spec
from . import array_operations as arr

from pylab import *


class DatacubeObj():
    """
    This class handles the relevant information and operations related
    to the datacubes
    """

    def __init__(self):
        pass

    def read_data(self, input_file):
        """
        Read the data and header of the specified datacube in fits format
        """

        hdulist = fits.open(input_file)
        self.cube = hdulist[0].data
        self.header = hdulist[0].header

    def get_attr(self):
        """
        Extract main information from the cards in the header
        """

        self.pixsize = self.header["CDELT1"] * unit.pc
        self.velocity_sampl = self.header["CDELT3"] * unit.km / unit.s
        self.spatial_dim = self.header["NAXIS1"]
        self.spectral_dim = self.header["NAXIS3"]
        self.fieldofview = self.spatial_dim * self.pixsize.to("kpc")
        self.velocity_range = self.spectral_dim * \
            self.velocity_sampl.to("km s^-1")
        self.channel_ref = self.header["CRPIX3"] - 1
        self.vel_ref = self.header["CRVAL3"] * unit.km / unit.s
        self.pixel_ref = self.header["CRPIX1"] - 1
        self.position_ref = self.header["CRVAL1"] * unit.pc
        self.channels = (np.arange(self.spectral_dim) - self.channel_ref) * (
            self.velocity_sampl.to("km s^-1").value) + self.vel_ref.to("km s^-1").value

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
        self.fluxmap = self.cube.sum(axis=0) * self.velocity_sampl.to("km s-1").value

    def velocity_map(self):
        if not hasattr(self,"fluxmap"):
            self.intensity_map()
        velmap = np.zeros(self.fluxmap.shape)
        for i in range(self.channels.size):
            velmap = velmap + \
                self.cube[i, :, :] * self.channels[i] * \
                self.velocity_sampl.to("km s-1").value
        self.velmap = velmap / self.fluxmap

    def dispersion_map(self):
        if not hasattr(self,"fluxmap"):
            self.intensity_map()
        if not hasattr(self,"velmap"):
            self.velocity_map()
        disper = np.zeros(self.fluxmap.shape)
        for i in range(self.channels.size):
            disper += self.cube[i, :, :]*(self.channels[i] - self.velmap)**2
        self.dispmap = np.sqrt(disper * self.velocity_sampl.to("km s-1").value / self.fluxmap)

    def all_maps(self):
        self.intensity_map()
        self.velocity_map()
        self.dispersion_map()


    def clean_lowflux(self, thresh = 11):
        zeros = (np.log10(np.max(self.fluxmap)) - np.log10(self.fluxmap) > thresh)
        self.fluxmap[zeros] = np.nan
        self.velmap[zeros] = np.nan
        self.dispmap[zeros] = np.nan






