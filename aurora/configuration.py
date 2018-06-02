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

# This variable stores the names of the parameters that were not passed
missing_params = []


def read_var(config_var, section, var, vartype, units=None):
    """
    Read a specific keyword from a section of a (loaded) ConfigFile,
    optionally specifying its units. Keywords not specified are picked
    from the presets file.

    Parameters
    ----------
    config_var : loaded ConfigFile
    section : corresponding section on the ConfigFile
    var : keyword to be loaded
    vartype : specifies if the value is a float, int, bool, or str
    units : (optional)
        adds units to the value of the variable
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
    run and additional information for the headers.
    """

    def __init__(self):
        pass

    def parse_input(self, ConfigFile):
        """
        Extract the parameters from the section [run] of the ConfigFile
        """

        run_config = configparser.SafeConfigParser({}, allow_no_value=True)
        run_config.read(ConfigFile)

        self.input_file = read_var(run_config, 'run', 'input_file', str)
        self.output_file = read_var(run_config, 'run', 'output_file', str)
        self.output_dir = read_var(run_config, 'run', 'custom_dir', str)
        self.instrument = read_var(run_config, 'run', 'instrument', str)
        self.nvector = read_var(run_config, 'run', 'nvector', int)
        self.ncpu = read_var(run_config, 'run', 'ncpu', int)
        self.overwrite = read_var(run_config, 'run', 'overwrite', bool)
        self.simulation_id = read_var(run_config, 'run', 'simulation_id', str)
        self.snapshot_id = read_var(run_config, 'run', 'snapshot_id', str)
        self.reference_id = read_var(run_config, 'run', 'reference_id', str)
        self.nfft = read_var(run_config, 'run', 'nfft', int)
        self.fft_hsml_min = read_var(run_config, 'run', 'fft_hsml_min', float,
                                     unit.pc)


class GeometryObj():
    """
    Group the main parameters related to the geometrical orientation
    adopted for the mock observations, and cuts to be applied to the
    snapshot particles before projecting their properties.
    """

    def __init__(self):
        pass

    def parse_input(self, ConfigFile):
        """
        Extract parameters from the section [geometry] of the ConfigFile
        """

        g_conf = configparser.SafeConfigParser(allow_no_value=True)
        g_conf.read(ConfigFile)

        self.redshift = read_var(g_conf, 'geometry', 'redshift', float)
        self.dl = read_var(g_conf, 'geometry', 'dist_lum', float, unit.Mpc)
        self.dist_angular = read_var(
            g_conf, 'geometry', 'dist_angular', float, unit.Mpc)
        self.lambda_em = read_var(
            g_conf, 'geometry', 'lambda_em', float, unit.angstrom)
        self.theta = read_var(g_conf, 'geometry', 'theta', float, unit.deg)
        self.phi = read_var(g_conf, 'geometry', 'phi', float, unit.deg)
        self.barycenter = read_var(g_conf, 'geometry', 'barycenter', bool)
        self.centerx = read_var(g_conf, 'geometry', 'centerx', float, unit.kpc)
        self.centery = read_var(g_conf, 'geometry', 'centery', float, unit.kpc)
        self.centerz = read_var(g_conf, 'geometry', 'centerz', float, unit.kpc)
        self.reference = read_var(g_conf, 'geometry', 'reference_frame', str)

        self.gas_minmax_keys = read_var(
            g_conf, 'geometry', 'gas_minmax_keys', str)
        self.gas_min_values = read_var(
            g_conf, 'geometry', 'gas_min_values', str)
        self.gas_max_values = read_var(
            g_conf, 'geometry', 'gas_max_values', str)
        self.star_minmax_keys = read_var(
            g_conf, 'geometry', 'star_minmax_keys', str)
        self.star_min_values = read_var(
            g_conf, 'geometry', 'star_min_values', str)
        self.star_max_values = read_var(
            g_conf, 'geometry', 'star_max_values', str)
        self.dm_minmax_keys = read_var(
            g_conf, 'geometry', 'dm_minmax_keys', str)
        self.dm_min_values = read_var(g_conf, 'geometry', 'dm_min_values', str)
        self.dm_max_values = read_var(g_conf, 'geometry', 'dm_max_values', str)

        # Filter the gas particles according to the specified properties and boundaries (if any)
        if(self.gas_minmax_keys != ''):
            self.gas_minmax_keys = re.split(
                ',|;', ''.join(self.gas_minmax_keys.split()))
        if(self.gas_min_values != ''):
            self.gas_min_values = (np.array(
                re.split(',|;', ''.join(self.gas_min_values.split())))).astype(np.float)
            if(len(self.gas_minmax_keys) != len(self.gas_min_values)):
                print(
                    'The number of elements in gas_minmax_keys and gas_min_values should be equal')
                sys.exit()
        if(self.gas_max_values != ''):
            self.gas_max_values = (np.array(
                re.split(',|;', ''.join(self.gas_max_values.split())))).astype(np.float)
            if(len(self.gas_minmax_keys) != len(self.gas_max_values)):
                print(
                    'The number of elements in gas_minmax_keys and gas_max_values should be equal')
                sys.exit()
        # Filter the stellar particles according to the specified properties and boundaries (if any)
        if(self.star_minmax_keys != ''):
            self.star_minmax_keys = re.split(
                ',|;', ''.join(self.star_minmax_keys.split()))
        if(self.star_min_values != ''):
            self.star_min_values = (np.array(
                re.split(',|;', ''.join(self.star_min_values.split())))).astype(np.float)
            if(len(self.stars_minmax_keys) != len(self.stars_min_values)):
                print(
                    'The number of elements in star_minmax_keys and star_min_values should be equal')
                sys.exit()
        if(self.star_max_values != ''):
            self.star_max_values = (np.array(
                re.split(',|;', ''.join(self.star_max_values.split())))).astype(np.float)
            if(len(self.star_minmax_keys) != len(self.star_max_values)):
                print(
                    'The number of elements in star_minmax_keys and star_max_values should be equal')
                sys.exit()
        # Filter the DM particles according to the specified properties and boundaries (if any)
        if(self.dm_minmax_keys != ''):
            self.dm_minmax_keys = re.split(
                ',|;', ''.join(self.dm_minmax_keys.split()))
        if(self.dm_min_values != ''):
            self.dm_min_values = (np.array(
                re.split(',|;', ''.join(self.dm_min_values.split())))).astype(np.float)
            if(len(self.dms_minmax_keys) != len(self.dms_min_values)):
                print(
                    'The number of elements in dm_minmax_keys and dm_min_values should be equal')
                sys.exit()
        if(self.dm_max_values != ''):
            self.dm_max_values = (np.array(
                re.split(',|;', ''.join(self.dm_max_values.split())))).astype(np.float)
            if(len(self.dm_minmax_keys) != len(self.dm_max_values)):
                print(
                    'The number of elements in dm_minmax_keys and dm_max_values should be equal')
                sys.exit()

# NOTA MENTAL!
# Aqui necesito una correccion: Que se puedan superseed *dist_angular* y *dl*

    def check_redshift(self):
        if ~np.isnan(self.redshift):
            self.dl = cosmo.luminosity_distance(self.redshift)
            self.lambda_em = (1 + self.redshift) * ct.Halpha0
            self.dist_angular = cosmo.kpc_proper_per_arcmin(self.redshift).to(
                'Mpc / rad').value * unit.Mpc

    def kpc_to_arcsec(self, length):
        length_arcsec = length.to('pc').value/self.dist_angular.to('pc').value
        length_arcsec = np.rad2deg(length_arcsec) * 3600 * unit.arcsec
        return length_arcsec

    def arcsec_to_kpc(self, angle):
        length = angle.to('rad').value * self.dist_angular.to('kpc')
        return length

    def vel_to_wavelength(self, vel):
        wavelength = self.lambda_em * vel.to('km s-1').value / (
            ct.c.to('km s-1').value)
        return wavelength

    def wavelength_to_vel(self, wavelength):
        vel = ct.c('km s-1') * wavelength.to('angstrom').value / (
            self.lambda_em.to('angstrom').value)
        return vel


class SpectromObj():
    """
    Group the relevant parameters for the *instrumental* set up
    and operations to check them for self consistency
    """

    def __init__(self):
        pass

    def parse_input(self, ConfigFile):
        """
        Extract parameters from the section [spectrom] of the ConfigFile
        """

        spec_conf = configparser.SafeConfigParser(allow_no_value=True)
        spec_conf.read(ConfigFile)
        self.presets = read_var(spec_conf, 'spectrom', 'presets', str)

        if self.presets in presets.Instruments.keys():
            self.spatial_sampl = presets.Instruments[self.presets]['spatial_sampl']
            self.spatial_sampl = float(self.spatial_sampl) * unit.arcsec
            self.spectral_sampl = presets.Instruments[self.presets]['spectral_sampl']
            self.spectral_sampl = float(self.spectral_sampl) * unit.angstrom
            self.spatial_res = presets.Instruments[self.presets]['spatial_res']
            self.spatial_res = float(self.spatial_res) * unit.arcsec
            self.spectral_res = presets.Instruments[self.presets]['spectral_res']
            self.spectral_res = float(self.spectral_res)
            self.spatial_dim = presets.Instruments[self.presets]['spatial_dim']
            self.spatial_dim = int(self.spatial_dim)
            self.spectral_dim = presets.Instruments[self.presets]['spectral_dim']
            self.spectral_dim = int(self.spectral_dim)
            self.sigma_cont = presets.Instruments[self.presets]['target_snr']
            self.sigma_cont = float(self.sigma_cont)
        else:
            self.spatial_sampl = read_var(
                spec_conf, 'spectrom', 'spatial_sampl', float, unit.arcsec)
            self.spectral_sampl = read_var(
                spec_conf, 'spectrom', 'spectral_sampl', float, unit.angstrom)
            self.spatial_res = read_var(
                spec_conf, 'spectrom', 'spatial_res', float, unit.arcsec)
            self.spectral_res = read_var(
                spec_conf, 'spectrom', 'spectral_res', float)
            self.spatial_dim = read_var(
                spec_conf, 'spectrom', 'spatial_dim', int)
            self.spectral_dim = read_var(
                spec_conf, 'spectrom', 'spectral_dim', int)
            self.sigma_cont = read_var(
                spec_conf, 'spectrom', 'sigma_cont', float)

        self.redshift_ref = read_var(
            spec_conf, 'spectrom', 'redshift_ref', float)
        self.pixsize = read_var(spec_conf, 'spectrom',
                                'pixsize', float, unit.pc)
        self.velocity_sampl = read_var(
            spec_conf, 'spectrom', 'velocity_sampl', float, unit.km/unit.s)
        self.fieldofview = read_var(
            spec_conf, 'spectrom', 'fieldofview', float, unit.kpc)
        self.velocity_range = read_var(
            spec_conf, 'spectrom', 'velocity_range', float, unit.km/unit.s)
        self.spatial_res_kpc = read_var(
            spec_conf, 'spectrom', 'spatial_res_kpc', float, unit.kpc)
        self.kernel_scale = read_var(
            spec_conf, 'spectrom', 'kernel_scale', float)
        self.oversampling = read_var(
            spec_conf, 'spectrom', 'oversampling', int)

    def cube_dims(self):
        """
        Just a shortcut for the cube dimensions
        """

        cube_side = self.spatial_dim
        n_ch = self.spectral_dim
        return cube_side, n_ch

    def oversample(self):
        """
        Adjust the configuration parameters to oversample the target
        spatial resolution according to *self.oversampling*
        """

        self.pixsize = self.pixsize / self.oversampling
        self.spatial_dim = self.spatial_dim * self.oversampling
        self.spatial_sampl = self.spatial_sampl / self.oversampling

    def undersample(self):
        """
        Adjust the configuration parameters to undersample the target
        spatial resolution according to *self.oversampling*
        """

        self.pixsize = self.pixsize * self.oversampling
        self.spatial_dim = self.spatial_dim / self.oversampling
        self.spatial_sampl = self.spatial_sampl * self.oversampling

    def check_params(self, geom):
        """
        Check the self-consistency of related parameters such as pixsize
        and spatial_sampl, and adjust them as necessary according to the
        hierarchy specified in the documentation file.
        """
        pass
        self.check_pixsize(geom)
        self.check_spatial_res_kpc(geom)
        self.check_fieldofview(geom)
        self.check_velocity_sampl(geom)
        self.check_velocity_range(geom)
        self.check_redshift_ref(geom)

    def check_pixsize(self, geom):
        """
        Force consistency between pixsize and spatial_sampl, superseding
        the passed value for the latter if necessary.
        """

        if ~np.isnan(self.pixsize):
            self.spatial_sampl = geom.kpc_to_arcsec(self.pixsize)
        else:
            if ~np.isnan(self.spatial_sampl):
                self.pixsize = geom.arcsec_to_kpc(self.spatial_sampl).to('pc')

    def check_spatial_res_kpc(self, geom):
        """
        Force consistency between spatial_res_kpc and spatial_res,
        superseding the passed value for the latter if necessary.
        """

        if ~np.isnan(self.spatial_res_kpc):
            self.spatial_res = geom.kpc_to_arcsec(self.spatial_res_kpc)
        else:
            if ~np.isnan(self.spatial_res):
                self.spatial_res_kpc = geom.arcsec_to_kpc(
                    self.spatial_res)
                print("yeah baby love", self.spatial_res_kpc)

    def check_fieldofview(self, geom):
        """
        Force consistency between fieldofview, FoV_arcsec, and spatial_dim,
        possibly superseding their input values as necessary.
        """

        if ~np.isnan(self.fieldofview):
            self.spatial_dim = int(self.fieldofview.to('pc'
                                                       ).value / self.pixsize.to('pc').value)
            self.fieldofview = self.spatial_dim * self.pixsize.to('kpc')
            self.FoV_arcsec = geom.kpc_to_arcsec(self.fieldofview)
        else:
            if ~np.isnan(self.FoV_arcsec):
                self.fieldofview = geom.arcsec_to_kpc(self.FoV_arcsec)
                self.check_fieldofview(geom)
            elif ~np.isnan(self.spatial_dim):
                self.fieldofview = self.spatial_dim * self.pixsize.to('kpc')
                self.check_fieldofview(geom)

    def check_velocity_sampl(self, geom):
        """
        Force consistency between velocity_sampl and spectral_sampl,
        possibly superseding the latter as necessary.
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
        """

        if ~np.isnan(self.velocity_range):
            self.spectral_dim = int(self.velocity_range.to('km s-1').value / (
                self.velocity_sampl.to('km s-1').value))
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
        channel_array = np.arange(self.spectral_dim) - self.spectral_dim / 2
        self.lambda_channels = geom.lambda_em + channel_array * self.spectral_sampl
        self.vel_channels = channel_array * self.velocity_sampl

    def check_redshift_ref(self, geom):
        """
        If redshift_ref was not defined, but redshift was, set them to
        the same value
        """

        global missing_params
        if self.redshift_ref in missing_params:
            if ~np.isnan(geom.redshift):
                self.redshift_ref = geom.redshift

    def set_reference(self):
        self.channel_ref = int(self.spectral_dim / 2.)
        self.vel_ref = 0. * unit.km / unit.s
        self.pixel_ref = (self.spatial_dim - 1) / 2.
        self.position_ref = 0. * unit.pc

    def get_one_channel(self, index):
        v_channel = (index - self.channel_ref)*self.velocity_sampl.to('km s-1')
        return v_channel + self.vel_ref.to('km s-1')

    def get_one_position(self, index):
        position = (index - self.pixel_ref) * self.pixsize.to('pc')
        return position + self.position_ref.to('pc')


def get_allinput(ConfigFile):
    """
    Read all attributes from the ConfigFile and adjust them for
    self-consistency, setting the missing parameters as necessary.

    Parameters
    ----------
    ConfigFile : location of the configuration file.
    """

    # Code flow:
    # =====================
    # > Parse the input parameters from ConfigFile
    # > Set the output name and check if it has the rights to write it
    if(not os.path.isfile(ConfigFile)):
        print('// ' + ConfigFile + ' not found')
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
    print("\n............ConfigFile Warning................")
    print("The following vars were NOT passed. Aurora will set them to")
    print("default values or will adjust them for self-consistency:")
    global missing_params
    print(missing_params)

    print("\n::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    print("Aurora is using the following parameters: \n")

    print("input file = ", run.input_file)
    print("output file = ", run.output_file)
    print("output dir = ", run.output_dir)
    print("overwrite = ", run.overwrite, "\n")

    print("redshift = ", geom.redshift)
    print("luminosity distance = ", geom.dl)
    print("angular distance = ", geom.dist_angular)
    print("redshifted wavelength = ", geom.lambda_em)
    print("barycenter = ", geom.barycenter)
    print("theta = ", geom.theta)
    print("phi = ", geom.phi)
    print("redshift_ref", spectrom.redshift_ref, "\n")

    print("field of view = ", spectrom.fieldofview)
    print("pixsize = ", spectrom.pixsize)
    print("spatial_dim = ", spectrom.spatial_dim)
    print("spatial_res_kpc = ", spectrom.spatial_res_kpc, "\n")

    print("FoV arcsec = ", spectrom.FoV_arcsec)
    print("spatial_sampl = ", spectrom.spatial_sampl)
    print("spatial_res = ", spectrom.spatial_res, "\n")

    print("oversampling = ", spectrom.oversampling)
    print("kernel scale = ", spectrom.kernel_scale)
    print("target noise = ", spectrom.sigma_cont, "\n")

    print("velocity_range = ", spectrom.velocity_range)
    print("velocity_sampl = ", spectrom.velocity_sampl)
    print("spectral_dim = ", spectrom.spectral_dim)
    print("spectral_res = ", spectrom.spectral_res, "\n")

    print("spectral_range = ", spectrom.spectral_range)
    print("spectral_sampl = ", spectrom.spectral_sampl)
    print("::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
    return geom, run, spectrom
