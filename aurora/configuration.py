"""
=============
configuration
=============

Aurora module that contains methods in charge of building the
three main classes of objects in the code:

> RunObj : Group the main parameters regarding the computational aspects of the
    run, including the variables that determines the computational
    performance of the code, and additional information for the headers.

> GeometryObj : Group the main parameters related to the geometrical orientation
    adopted for the mock observations, operations to check them for
    self consistency, cuts to be applied to the snapshot particles
    before use their properties and operations to transform equivalent
    properties in different units.

> SpectromObj : Group the relevant parameters for the *instrumental* set up
    adopted for the mock observations, and operations to check
    them for self consistency.

Notes
-----
For the self consistency of the cosmological distances, the hierarchy for the
parameters inside the code is:

> redshift (redshift)
> dist_lum (luminosity distance)
> lambda_obs (central observed wavelength)
> dist_angular (angular diameter distance)

For the self consistency of the spatial features of the instrument, the length
parameters, like pixsize in (pc), will overwrite the angular parameters, like
spatial_sampl in (arsec).

For the self consistency of the spectral features of the instrument, the velocity
parameters, like velocity_range in (km s-1), will overwrite the wavelenght
parameters, like spectral_range in (angstrom).
"""

import os
import re
import sys
import logging
import numpy as np
import configparser
from astropy import units as unit
from astropy.cosmology import Planck13 as cosmo
from astropy.cosmology import z_at_value

from . import presets
from . import constants as ct
from . import set_output as so

# This variable stores the names of the parameters that were not passed
missing_params = []


def read_var(config_var, section, var, vartype, units=None):
    """
    Read a specific keyword from a section of a (loaded) ConfigFile,
    optionally specifying its units. Keywords not specified are picked
    from the presets file.

    Parameters
    ----------
    config_var : str
        Loaded ConfigFile.
    section : str
        Corresponding section on the ConfigFile.
    var : str
        Keyword to be loaded.
    vartype : type
        Specifies if the value is a float, int, bool, or str
    units : astropy.units.core.Unit, optional
        Adds units to the value of the variable

    Returns
    -------
    output : str, type, astropy.units.core.Unit
        Return the readed variable with its value and units.
    """

    # Code flow:
    # =====================
    # > Try to read the variable from the ConfigFile
    # > If not present, try to load a defaut value from the presets file
    # > If no default exists, the parameter is set to nan
    
    try:
        if vartype == float:
            output = config_var.getfloat(section, var)
        elif vartype == int:
            output = config_var.getint(section, var)
        elif vartype == str:
            output = config_var.get(section, var)
        elif vartype == bool:
            output = config_var.getboolean(section, var)
    except:
        global missing_params
        missing_params.append(var)
        try:
            output = presets.default_values[section][var][0]
            try:
                units = presets.default_values[section][var][1]
            except:
                units = None
        except:
            output = np.nan
    if units:
        output *= units
    return output


class RunObj():
    """
    Group the main parameters regarding the computational aspects of the
    run, including the variables that determines the computational
    performance of the code, and additional information for the headers.
    """

    def __init__(self):
        pass

    def parse_input(self, ConfigFile):
        """
        Extract the parameters from the section [run] of the ConfigFile.
        
        Parameters
        ----------
        ConfigFile : str
            Loaded configuration file.
        
        Returns
        -------
        input_file : str
            Path of the input file.
        output_file : str
            Path of the output file.
        output_dir : str
            Path of the output directory.
        instrument : str
            Keyword to include the name of the instrumentational
            set up of the mock observation to be added in the
            output_dir name if it was not previously specified.
        nvector : int
            Number of particles to be projected simultaneously.
        ncpu : int
            Number of cores for parallel execution.
        overwrite : bool
            Allows overwriting an existing file when saving the output.
        simulation_id : str
            Keyword to be store in the FITS header as the parent
            simulation identification.
        snapshot_id : str
            Keyword to be store in the FITS header as the snapshot
            identification.
        reference_id : str
            Keyword to be store in the FITS header as the reference
            frame identification for geometrical transformations.
        nfft : int
            Number of different scales of particles.
        fft_hsml_min : astropy.units.core.Unit
            Minimum size of particles in (pc).
        fft_scales : astropy.units.core.Unit
            Path of the file that list of different scales to pack the
            particles in (Kpc).
        spatial_convolution : str
            Keyword to store the method by which the spatial convolution
            will be performed.
            Options:
            *  spatial_astropy (default): Convolution using the Astropy
               library.
            *  spatial_astorpy_fft : Convolution using Fast Fourier
               transform from the Astropy library.
            *  spatial_aurora_fft : Convolution using Fast Fourier
               transform.
        spectral_convolution : str
            Keyword to store the method by which the spectral convolution
            will be performed.
            Options:
         *  analytical (default): Analytical convolution between the 
            Gaussian emission lines and the Gaussian kernel of the LSF.
         *  spectral_astropy : Convolution using the Astropy library.
         *  spectral_astorpy_fft : Convolution using Fast Fourier 
            transform from the Astropy library.
         *  spectral_aurora_fft : Convolution using Fast Fourier
            transform.
        """

        run_config = configparser.SafeConfigParser({}, allow_no_value=True)
        run_config.read(ConfigFile)

        self.input_file = read_var(run_config, "run", "input_file", str)
        self.output_file = read_var(run_config, "run", "output_file", str)
        self.output_dir = read_var(run_config, "run", "custom_dir", str)
        self.instrument = read_var(run_config, "run", "instrument", str)
        self.nvector = read_var(run_config, "run", "nvector", int)
        self.ncpu = read_var(run_config, "run", "ncpu", int)
        self.overwrite = read_var(run_config, "run", "overwrite", bool)
        self.simulation_id = read_var(run_config, "run", "simulation_id", str)
        self.snapshot_id = read_var(run_config, "run", "snapshot_id", str)
        self.reference_id = read_var(run_config, "run", "reference_id", str)
        self.nfft = read_var(run_config, "run", "nfft", int)
        self.fft_hsml_min = read_var(run_config, "run", "fft_hsml_min", float,
                                     unit.pc)
        self.fft_scales = read_var(run_config, "run", "fft_scales", str)
        self.spatial_convolution = read_var(run_config, "run", 
                                            "spatial_convolution", str)
        self.spectral_convolution = read_var(run_config, "run", 
                                             "spectral_convolution", str)

        
class GeometryObj():
    """
    Group the main parameters related to the geometrical orientation
    adopted for the mock observations, operations to check them for
    self consistency, cuts to be applied to the snapshot particles
    before use their properties and operations to transform equivalent
    properties in different units.
    """

    def __init__(self):
        pass

    def parse_input(self, ConfigFile):
        """
        Extract parameters from the section [geometry] of the ConfigFile.
        
        Parameters
        ----------
        ConfigFile : str
            Loaded configuration file.
        
        Returns
        -------
        redshift : float
            Redshift where the galaxy is located.
        dl : astropy.units.core.Unit 
            Luminosity distance where the galaxy is located in (Mpc)
        dist_angular : astropy.units.core.Unit
            Angular diameter distance in (Mpc).
        lambda_obs : astropy.units.core.Unit
            Central observed wavelength in (angstrom).
        theta : astropy.units.core.Unit
            Orientation angle of the major axis of the projected galaxy
            in (deg).
        phi : astropy.units.core.Unit
            Angle of inclination of the disc with respect to the line
            of sight in (deg).
        barycenter : bool
            Allows to calculate the center of the galaxy based on the 
            ssc centering-scheme of astropy.
        centerx : astropy.units.core.Unit
            Center of the galaxy in the X axis in (kpc).
        centery : astropy.units.core.Unit
            Center of the galaxy in the Y axis in (kpc).
        centerz : astropy.units.core.Unit
            Center of the galaxy in the Z axis in (kpc).
        reference : str
            Path of the reference file.
        gas_minmax_keys : str
            Specific properties to filter the gas
            particles.
        gas_minmax_units : str
            Units of the specific properties to filter 
            the gas particles.
        gas_min_values : str
            Minimum boundary to filter the gas particles.
        gas_max_values : str
            Maximum boundary to filter the gas particles.
        """

        g_conf = configparser.SafeConfigParser(allow_no_value=True)
        g_conf.read(ConfigFile)

        self.redshift = read_var(g_conf, "geometry", "redshift", float)
        self.dl = read_var(g_conf, "geometry", "dist_lum", float, unit.Mpc)
        self.dist_angular = read_var(
            g_conf, "geometry", "dist_angular", float, unit.Mpc)
        self.lambda_obs = read_var(
            g_conf, "geometry", "lambda_obs", float, unit.angstrom)
        self.theta = read_var(g_conf, "geometry", "theta", float, unit.deg)
        self.phi = read_var(g_conf, "geometry", "phi", float, unit.deg)
        self.barycenter = read_var(g_conf, "geometry", "barycenter", bool)
        self.centerx = read_var(g_conf, "geometry", "centerx", float, unit.kpc)
        self.centery = read_var(g_conf, "geometry", "centery", float, unit.kpc)
        self.centerz = read_var(g_conf, "geometry", "centerz", float, unit.kpc)
        self.reference = read_var(g_conf, "geometry", "reference_frame", str)

        self.gas_minmax_keys = read_var(
            g_conf, "geometry", "gas_minmax_keys", str)
        self.gas_minmax_units = read_var(
            g_conf, "geometry", "gas_minmax_units", str)
        self.gas_min_values = read_var(
            g_conf, "geometry", "gas_min_values", str)
        self.gas_max_values = read_var(
            g_conf, "geometry", "gas_max_values", str)


        # Filter the gas particles according to the specified properties and boundaries (if any)
        
        if(self.gas_minmax_keys != ""):
            self.gas_minmax_keys = re.split(
                ",|;", "".join(self.gas_minmax_keys.split()))
        if(self.gas_min_values != ""):
            self.gas_min_values = (np.array(
                re.split(",|;", "".join(self.gas_min_values.split())))).astype(np.float)
            if(len(self.gas_minmax_keys) != len(self.gas_min_values)):
                logging.error(
                    "The number of elements in gas_minmax_keys and gas_min_values should be equal")
                sys.exit()
        if(self.gas_max_values != ""):
            self.gas_max_values = (np.array(
                re.split(",|;", "".join(self.gas_max_values.split())))).astype(np.float)
            if(len(self.gas_minmax_keys) != len(self.gas_max_values)):
                logging.error(
                    "The number of elements in gas_minmax_keys and gas_max_values should be equal")
                sys.exit()

    def check_redshift(self):
        """
        Force consistency between redshift, luminosity distance, central
        observed wavelength and angular diameter distance, superseding
        the values of each one of this variables following the hierarchical
        order previously presented. All estimates of these quantities are 
        based on LambdaCDM cosmology.
        
        Returns
        -------
        dl : astropy.units.core.Unit 
            Luminosity distance where the galaxy is located in (Mpc)
        lambda_obs : astropy.units.core.Unit
            Central observed wavelength in (angstrom).        
        dist_angular : astropy.units.core.Unit
            Angular diameter distance in (Mpc).
        """
        
        if ~np.isnan(self.redshift):
            self.dl = cosmo.luminosity_distance(self.redshift)
            self.lambda_obs = (1 + self.redshift) * ct.Halpha0
            self.dist_angular = cosmo.angular_diameter_distance(self.redshift)
        
        else:
            if ~np.isnan(self.dl):
                self.redshift = z_at_value(cosmo.luminosity_distance, self.dl)
                self.lambda_obs = (1 + self.redshift) * ct.Halpha0
                self.dist_angular = cosmo.angular_diameter_distance(self.redshift)
            
            elif ~np.isnan(self.lambda_obs):
                self.redshift =  (self.lambda_obs/ct.Halpha0) - 1
                self.dl = cosmo.luminosity_distance(self.redshift)
                self.dist_angular = cosmo.angular_diameter_distance(self.redshift)
           
            # For the angular diameter distance there are multiple corresponding redshift
            # values for the LambdaCDM cosmology, so here we will choose the one that meets
            # the parameters established within the z_at_value module of astropy.
            elif ~np.isnan(self.dist_angular):
                self.redshift = z_at_value(cosmo.angular_diameter_distance, self.dist_angular)
                self.dl = cosmo.luminosity_distance(self.redshift)
                self.lambda_obs = (1 + self.redshift) * ct.Halpha0

    def kpc_to_arcsec(self, length):
        """
        Transform the lengths to arsec using the angular diameter
        distance.
        
        Parameters
        ----------
        length : astropy.units.quantity.Quantity
            Length to be transformed into arsec units.
        
        Returns
        -------
        length_arsec : astropy.units.quantity.Quantity
            Transformed length in (arsec).
        """
        length_arcsec = length.to("pc").value/self.dist_angular.to("pc").value
        length_arcsec = np.rad2deg(length_arcsec) * 3600 * unit.arcsec
        return length_arcsec

    def arcsec_to_kpc(self, angle):
        """
        Transform the angles to lengths using the angular diameter
        distance.
        
        Parameters
        ----------
        angle : astropy.units.quantity.Quantity
            Angle to be transformed into kpc units.
        
        Returns
        -------
        length : astropy.units.quantity.Quantity
            Transformed length in (kpc).
        """
        
        length = angle.to("rad").value * self.dist_angular.to("kpc")
        return length

    def vel_to_wavelength(self, vel):
        """
        Transform the velocities to wavelenghts using the
        central observed wavelength of the galaxy.
        
        Parameters
        ----------
        vel : astropy.units.quantity.Quantity
            Velocity to be transformed into angstrom units.
        
        Returns
        -------
        wavelength : astropy.units.quantity.Quantity
            Transformed velocity in (angstrom).
        """
        
        wavelength = self.lambda_obs * vel.to("km s-1").value / (
            ct.c.to("km s-1").value)
        return wavelength

    def wavelength_to_vel(self, wavelength):
        """
        Transform the wavelengths to velocities using the
        central observed wavelength of the galaxy.
        
        Parameters
        ----------
        wavelength : astropy.units.quantity.Quantity
            Wavelength to be transformed into km * s-1 units.
        
        Returns
        -------
        vel : astropy.units.quantity.Quantity
            Transformed wavelength in (km s-1).
        """
        vel = ct.c.to("km s-1") * wavelength.to("angstrom").value / (
            self.lambda_obs.to("angstrom").value)
        return vel


class SpectromObj():
    """
    Group the relevant parameters for the *instrumental* set up
    adopted for the mock observations, and operations to check
    them for self consistency.    
    """

    def __init__(self):
        pass

    def parse_input(self, ConfigFile):
        """
        Extract parameters from the section [spectrom] of the ConfigFile
        
        Parameters
        ----------
        ConfigFile : str
            Loaded configuration file.
        
        Returns
        -------
        presets : str
            Name of the instrument to mimic, some options are: sinfoni,
            eagle, ghasp, muse-wide, etc. See all options in presets.py.
        spatial_sampl : astropy.units.core.Unit 
            Pixel size of the instrument in (arsec).
        spectral_sampl : astropy.units.core.Unit
            Spectral sampling of the instrument in (angstrom).
        spatial_res : astropy.units.core.Unit
            Spatial resolution of the instrument in (arsec).
        spectral_res : float
            Spectral resolution of the instrument.
        spatial_dim : int
            Number of pixels per side of the field of view.
        spectral_dim : int
            Number of spectral channels of the instrument.
        sigma_cont : float
            Target signal to noise ratio of the instrument.
        redshift_ref : float
            Reference redshift used to calculate the fraction
            of ionized hydrogen using the procedure exposed in
            (Rahmati et al 2012). See more info in rahmati.py.
        pixsize : astropy.units.core.Unit
            Pixel size of the instrument in (pc).
        velocity_sampl : astropy.units.core.Unit
            Spectral sampling of the instrument in velocity units
            (km s-1).
        fieldofview : astropy.units.core.Unit
            Size of one side of the square field of view of the
            instrument in (kpc).
        FoV_arcsec : astropy.units.core.Unit
            Size of one side of the square field of view of the
            instrument in (arsec).
        velocity_range : astropy.units.core.Unit
            Spectral range of the instrument in velocity units
            (km s-1).
        spectral_range : astropy.units.core.Unit
            Spectral range of the instrument in (angstrom).
        spatial_res_kpc : astropy.units.core.Unit
            Spatial resolution of the instrument in (kpc).
        kernel_scale : float
            Constant that apply a higher smoothing to the projected
            luminosity of the particles, especially for RAMSES-type
            simulations.
        oversampling : int
            Number by which the pixel size is going to be oversampled
            when the convolution is carried out, to minimize
            numerical errors.
        lum_dens_rel : str
            Ions number density dependence to calculate the H-alpha
            emission. The options are: square (default), linear or
            root. See more info in emitters.py.
        density_threshold : str
            Density threshold that allows to change the H-alpha
            emission for certain gas particles that exceed the
            established limit. See more info in emitters.py.  
        equivalent_luminosity : str
            Equivalent luminosity that replace the H-alpha emission
            for certain gas particles that exceed the established
            density threshold. See more info in emitters.py. 
        """

        spec_conf = configparser.SafeConfigParser(allow_no_value=True)
        spec_conf.read(ConfigFile)
        self.presets = read_var(spec_conf, "spectrom", "presets", str)

        if self.presets in presets.Instruments.keys():
            self.spatial_sampl = presets.Instruments[self.presets]["spatial_sampl"]
            self.spatial_sampl = float(self.spatial_sampl) * unit.arcsec
            self.spectral_sampl = presets.Instruments[self.presets]["spectral_sampl"]
            self.spectral_sampl = float(self.spectral_sampl) * unit.angstrom
            self.spatial_res = presets.Instruments[self.presets]["spatial_res"]
            self.spatial_res = float(self.spatial_res) * unit.arcsec
            self.spectral_res = presets.Instruments[self.presets]["spectral_res"]
            self.spectral_res = float(self.spectral_res)
            self.spatial_dim = presets.Instruments[self.presets]["spatial_dim"]
            self.spatial_dim = int(self.spatial_dim)
            self.spectral_dim = presets.Instruments[self.presets]["spectral_dim"]
            self.spectral_dim = int(self.spectral_dim)
            self.sigma_cont = presets.Instruments[self.presets]["target_snr"]
            self.sigma_cont = float(self.sigma_cont)
        else:
            self.spatial_sampl = read_var(
                spec_conf, "spectrom", "spatial_sampl", float, unit.arcsec)
            self.spectral_sampl = read_var(
                spec_conf, "spectrom", "spectral_sampl", float, unit.angstrom)
            self.spatial_res = read_var(
                spec_conf, "spectrom", "spatial_res", float, unit.arcsec)
            self.spectral_res = read_var(
                spec_conf, "spectrom", "spectral_res", float)
            self.spatial_dim = read_var(
                spec_conf, "spectrom", "spatial_dim", int)
            self.spectral_dim = read_var(
                spec_conf, "spectrom", "spectral_dim", int)
            self.sigma_cont = read_var(
                spec_conf, "spectrom", "sigma_cont", float)

        self.redshift_ref = read_var(
            spec_conf, "spectrom", "redshift_ref", float)
        self.pixsize = read_var(spec_conf, "spectrom",
                                "pixsize", float, unit.pc)
        self.velocity_sampl = read_var(
            spec_conf, "spectrom", "velocity_sampl", float, unit.km/unit.s)
        self.fieldofview = read_var(
            spec_conf, "spectrom", "fieldofview", float, unit.kpc)
        self.FoV_arcsec = read_var(
            spec_conf, "spectrom", "FoV_arsec", float, unit.arcsec)
        self.velocity_range = read_var(
            spec_conf, "spectrom", "velocity_range", float, unit.km/unit.s)
        self.spectral_range = read_var(
            spec_conf, "spectrom", "spectral_range", float, unit.angstrom)
        self.spatial_res_kpc = read_var(
            spec_conf, "spectrom", "spatial_res_kpc", float, unit.kpc)
        self.kernel_scale = read_var(
            spec_conf, "spectrom", "kernel_scale", float)
        self.oversampling = read_var(
            spec_conf, "spectrom", "oversampling", int)
        self.lum_dens_rel = read_var(
            spec_conf, "spectrom", "lum_dens_relation", str)
        self.density_threshold = read_var(
            spec_conf, "spectrom", "density_threshold", str)
        self.equivalent_luminosity = read_var(
            spec_conf, "spectrom", "equivalent_luminosity", str)

    def cube_dims(self):
        """
        Just a shortcut for the cube dimensions.
        
        Returns
        -------
        cube_side : int
            Number of pixels per side of the
            field of view.
        n_ch : int
            Number of spectral channels.
        """

        cube_side = self.spatial_dim
        n_ch = self.spectral_dim
        return cube_side, n_ch

    def position_in_pixels(self,x,y):
        """
        Calculate the x and y position in pixels for
        each particle, and the positional index along
        the face on slide of the cube.

        Parameters
        ----------
        x : astropy.units.quantity.Quantity
            Loaded position in X axis (Kpc) from
            CofigFile for each particle.
        y : astropy.units.quantity.Quantity
            Loaded position in Y axis (Kpc) from
            CofigFile for each particle.

        Returns
        -------
        x : astropy.units.quantity.Quantity
            Position in X axis (pixels) for each particle.
        y : astropy.units.quantity.Quantity
            Position in Y axis (pixels) for each particle.
        index : astropy.units.quantity.Quantity
            Positional index in pixels for each particle.
        """
        
        cube_side, n_ch = self.cube_dims()
        x = (x / self.pixsize).decompose()
        y = (y / self.pixsize).decompose()
        x = (np.floor(x + cube_side / 2)).astype(int)
        y = (np.floor(y + cube_side / 2)).astype(int)

        # Due to errors in float claculations, eventually one particle may lie
        # just outside of the box
        x[x == cube_side] = (cube_side -1)
        x[x == -1] = 0
        y[y == cube_side] = (cube_side -1)
        y[y == -1] = 0

        index = x + cube_side * y
        return x,y,index

    def oversample(self):
        """
        Adjust the configuration parameters to oversample the target
        spatial resolution according to *self.oversampling*.
        
        Returns
        -------
        pixsize : astropy.units.core.Unit
            Pixel size of the instrument in (pc).
        spatial_dim : int
            Number of pixels per side of the field of view.
        spatial_sampl : astropy.units.core.Unit 
            Pixel size of the instrument in (arsec).
        """

        self.pixsize = self.pixsize / self.oversampling
        self.spatial_dim = self.spatial_dim * self.oversampling
        self.spatial_sampl = self.spatial_sampl / self.oversampling

    def undersample(self):
        """
        Adjust the configuration parameters to undersample the target
        spatial resolution according to *self.oversampling*.
        
        Returns
        -------
        pixsize : astropy.units.core.Unit
            Pixel size of the instrument in (pc).
        spatial_dim : int
            Number of pixels per side of the field of view.
        spatial_sampl : astropy.units.core.Unit 
            Pixel size of the instrument in (arsec).
        """

        self.pixsize = self.pixsize * self.oversampling
        self.spatial_dim = self.spatial_dim / self.oversampling
        self.spatial_sampl = self.spatial_sampl * self.oversampling

    def check_params(self, geom):
        """
        Check the self-consistency of related parameters such
        as pixsize and spatial_sampl, and adjust them as
        necessary according to the hierarchy specified in the
        documentation file.
        """
        pass
        self.check_pixsize(geom)
        self.check_spatial_res_kpc(geom)
        self.check_fieldofview(geom)
        self.check_velocity_sampl(geom)
        self.check_velocity_range(geom)

    def check_pixsize(self, geom):
        """
        Force consistency between pixsize and spatial_sampl,
        superseding the passed value for the latter if necessary.
        
        Returns
        -------
        spatial_sampl : astropy.units.core.Unit 
            Pixel size of the instrument in (arsec).
        pixsize : astropy.units.core.Unit
            Pixel size of the instrument in (pc).
        """

        if ~np.isnan(self.pixsize):
            self.spatial_sampl = geom.kpc_to_arcsec(self.pixsize)
        else:
            if ~np.isnan(self.spatial_sampl):
                self.pixsize = geom.arcsec_to_kpc(self.spatial_sampl).to("pc")

    def check_spatial_res_kpc(self, geom):
        """
        Force consistency between spatial_res_kpc and spatial_res,
        superseding the passed value for the latter if necessary.
        
        Returns
        -------
        spatial_res : astropy.units.core.Unit
            Spatial resolution of the instrument in (arsec).
        spatial_res_kpc : astropy.units.core.Unit
            Spatial resolution of the instrument in (kpc).
        """

        if ~np.isnan(self.spatial_res_kpc):
            self.spatial_res = geom.kpc_to_arcsec(self.spatial_res_kpc)
        else:
            if ~np.isnan(self.spatial_res):
                self.spatial_res_kpc = geom.arcsec_to_kpc(
                    self.spatial_res)

    def check_fieldofview(self, geom):
        """
        Force consistency between fieldofview, FoV_arcsec, and spatial_dim,
        possibly superseding their input values as necessary.
        
        Returns
        -------
        fieldofview : astropy.units.core.Unit
            Size of one side of the square field of view of the
            instrument in (kpc).
        FoV_arcsec : astropy.units.core.Unit
            Size of one side of the square field of view of the
            instrument in (arsec).
        spatial_dim : int
            Number of pixels per side of the field of view.
        """

        if ~np.isnan(self.fieldofview):
            self.spatial_dim = int(self.fieldofview.to("pc"
                                                       ).value / self.pixsize.to("pc").value)
            self.fieldofview = self.spatial_dim * self.pixsize.to("kpc")
            self.FoV_arcsec = geom.kpc_to_arcsec(self.fieldofview)
        else:
            if ~np.isnan(self.FoV_arcsec):
                self.fieldofview = geom.arcsec_to_kpc(self.FoV_arcsec)
                self.check_fieldofview(geom)
            elif ~np.isnan(self.spatial_dim):
                self.fieldofview = self.spatial_dim * self.pixsize.to("kpc")
                self.check_fieldofview(geom)

    def check_velocity_sampl(self, geom):
        """
        Force consistency between velocity_sampl and spectral_sampl,
        possibly superseding the latter as necessary.
        
        Returns
        -------
        spectral_sampl : astropy.units.core.Unit
            Spectral sampling of the instrument in (angstrom).
        velocity_sampl : astropy.units.core.Unit
            Spectral sampling of the instrument in velocity units
            (km s-1).
        """

        if ~np.isnan(self.velocity_sampl):
            self.spectral_sampl = geom.vel_to_wavelength(self.velocity_sampl)
        else:
            if ~np.isnan(self.spectral_sampl):
                self.velocity_sampl = geom.wavelength_to_vel(
                    self.spectral_sampl)

    def check_velocity_range(self, geom):
        """
        Force consistency between velocity_range, spectral_range, and
        spectral_dim, possibly superseding their input values.
        
        Returns
        -------        
        velocity_range : astropy.units.core.Unit
            Spectral range of the instrument in velocity units
            (km s-1).
        spectral_range : astropy.units.core.Unit
            Spectral range of the instrument in (angstrom).
        spectral_dim : int
            Number of spectral channels of the instrument.
        """

        if ~np.isnan(self.velocity_range):
            self.spectral_dim = int(self.velocity_range.to("km s-1").value / (
                self.velocity_sampl.to("km s-1").value))
            self.velocity_range = self.spectral_dim * self.velocity_sampl
            self.spectral_range = geom.vel_to_wavelength(self.velocity_range)
        else:
            if ~np.isnan(self.spectral_range):
                self.velocity_range = geom.wavelength_to_vel(
                    self.spectral_range)
                self.check_velocity_range(geom)
            elif ~np.isnan(self.spectral_dim):
                self.velocity_range = self.spectral_dim * self.velocity_sampl
                self.check_velocity_range(geom)

    def set_channels(self, geom):
        """
        Set the central values of each one of the spectral 
        channels.
        
        Returns
        -------        
        vel_channels : astropy.units.quantity.Quantity
            Central spectral values of each channel in
            velocity units (km s-1).
        lambda_channels : astropy.units.quantity.Quantity
            Central spectral values of each channel in
            (angstrom).
        """
        
        channel_array = np.arange(self.spectral_dim) - self.spectral_dim / 2
        self.lambda_channels = geom.lambda_obs + channel_array * self.spectral_sampl
        self.vel_channels = channel_array * self.velocity_sampl

    def set_reference(self):
        """
        Set the reference values for each channel, pixel
        position and velocity.
        
        Returns
        -------        
        channel_ref : int
            Reference value for each channel.
        vel_ref : astropy.units.quantity.Quantity
            Reference value for the velocity.
        pixel_ref : float
            Reference value for each pixel.
        position_ref : astropy.units.quantity.Quantity
            Reference position.
        """
        
        self.channel_ref = int(self.spectral_dim / 2.)
        self.vel_ref = 0. * unit.km / unit.s
        self.pixel_ref = (self.spatial_dim - 1) / 2.
        self.position_ref = 0. * unit.pc

    def get_one_channel(self, index):
        """
        Get the velocity of each particle in (km s-1) 
        for each positional index, taking into account
        the reference velocity.
        
        Parameters
        ----------
        index : astropy.units.quantity.Quantity
            Positional index in pixels for each particle.
        
        Returns
        -------        
        v_channel + vel_ref : astropy.units.quantity.Quantity
            Velocity of each particle in (km s-1) for each index.        
        """
        
        v_channel = (index - self.channel_ref)*self.velocity_sampl.to("km s-1")
        return v_channel + self.vel_ref.to("km s-1")

    def get_one_position(self, index):
        """
        Get the position of each particle in (pc) for
        each positional index, taking into account the
        reference position.
        
        Parameters
        ----------
        index : astropy.units.quantity.Quantity
            Positional index in pixels for each particle.
        
        Returns
        -------        
        position + position_ref : astropy.units.quantity.Quantity
            Position of each particle in (pc) for each index.        
        """        
        position = (index - self.pixel_ref) * self.pixsize.to("pc")
        return position + self.position_ref.to("pc")

    
def get_allinput(ConfigFile):
    """
    Read all attributes from the ConfigFile and adjust them for
    self-consistency, setting the missing parameters as necessary.

    Parameters
    ----------
    ConfigFile : str
        Loaded configuration file.
    
    Returns
    -------  
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
    """

    # Code flow:
    # =====================
    # > Parse the input parameters from ConfigFile
    # > Set the output name and check if it has the rights to write it
    
    if(not os.path.isfile(ConfigFile)):
        logging.error(f"// {ConfigFile} not found")
        return
    geom = GeometryObj()
    geom.parse_input(ConfigFile)
    run = RunObj()
    run.parse_input(ConfigFile)
    spectrom = SpectromObj()
    spectrom.parse_input(ConfigFile)
    if (spectrom.presets in presets.Instruments.keys()):
        run.instrument = spectrom.presets
    run.output_name = so.set_output_filename(geom, run)

    # Code flow:
    # =====================
    # > Adjust the parameters for self-consistency
    # > Defines the velocity/wavelength channels of the cube
    geom.check_redshift()
    spectrom.check_params(geom)
    spectrom.set_channels(geom)
    spectrom.set_reference()

    # These prints might go to a file set by the user in the future
    logging.warning(f"\n............ConfigFile Warning................")
    logging.warning(f"The following vars were NOT passed. Aurora will set them to")
    logging.warning(f"default values or will adjust them for self-consistency:")
    global missing_params
    logging.warning(missing_params)

    logging.info(f"\n::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    logging.info(f"Aurora is using the following parameters: \n")

    logging.info(f"input file = {run.input_file}")
    logging.info(f"output file = {run.output_file}")
    logging.info(f"output dir = {run.output_dir}")
    logging.info(f"overwrite = {run.overwrite}\n")

    logging.info(f"redshift = {geom.redshift}")
    logging.info(f"luminosity distance = {geom.dl}")
    logging.info(f"angular distance = {geom.dist_angular}")
    logging.info(f"redshifted wavelength = {geom.lambda_obs}")
    logging.info(f"barycenter = {geom.barycenter}")
    logging.info(f"theta = {geom.theta}")
    logging.info(f"phi = {geom.phi}")
    logging.info(f"redshift_ref= {spectrom.redshift_ref}\n")

    logging.info(f"field of view = {spectrom.fieldofview}")
    logging.info(f"pixsize = {spectrom.pixsize}")
    logging.info(f"spatial_dim = {spectrom.spatial_dim}")
    logging.info(f"spatial_res_kpc = {spectrom.spatial_res_kpc}\n")

    logging.info(f"FoV arcsec = {spectrom.FoV_arcsec}")
    logging.info(f"spatial_sampl = {spectrom.spatial_sampl}")
    logging.info(f"spatial_res = {spectrom.spatial_res}\n")

    logging.info(f"oversampling = {spectrom.oversampling}")
    logging.info(f"kernel scale = {spectrom.kernel_scale}")
    logging.info(f"target noise = {spectrom.sigma_cont}\n")

    logging.info(f"velocity_range = {spectrom.velocity_range}")
    logging.info(f"velocity_sampl = {spectrom.velocity_sampl}")
    logging.info(f"spectral_dim = {spectrom.spectral_dim}")
    logging.info(f"spectral_res = {spectrom.spectral_res}\n")

    logging.info(f"spectral_range = {spectrom.spectral_range}")
    logging.info(f"spectral_sampl = {spectrom.spectral_sampl}")
    logging.info(f"::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
    return geom, run, spectrom
