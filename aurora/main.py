"""
====
main
====
The functions in this module allow the post-processing of
hydrodynamical simulations to create mock H-alpha or HI 
observations using the whole physical and geometrical 
information of the particles in the simulation.
"""

import gc
import re
import os
import sys
import pdb
import math
import time
import logging
import pynbody
import warnings
import numpy as np
import configparser
import scipy.fftpack
from scipy import ndimage
from pympler import asizeof
from astropy.io import fits
import multiprocessing as mp
from scipy.stats import norm
import matplotlib.pyplot as plt
from astropy import units as unit
from sklearn.neighbors import KDTree
from astropy import constants as const
from astropy.cosmology import Planck13 as cosmo

from . import presets
from . import constants as ct
from . import set_output as so
from . import datacube as dc
from . import snapshot_tools as snap
from . import spectrum_tools as spec
from . import configuration as config
from . import array_operations as arr
from . import rahmati as rahmati
from . import simu_projection as spj

warnings.filterwarnings("ignore")

def __setup_logging():
    logging.basicConfig(
        format="%(levelname)s:%(message)s", 
        filename="aurora.log", 
        level=logging.INFO)

def __setup_logging_spj():
    logging.basicConfig(
        format="%(levelname)s:%(message)s", 
        filename="aurora_spj.log", 
        level=logging.INFO)

def __aurora_version():
    """
    Print the version of Aurora being used for future references.
    """
    print("   ___               ")
    print("  / _ |__ _________  _______ _   ")
    print(" / __ / // / __/ _ \/ __/ _ `/   ")
    print("/_/ |_\___/_/  \___/_/  \___/    ")
    print("////// Version 2.1")


def spectrom_mock(ConfigFile):
    """
    Map the estimated H-alpha or HI flux from the simulation to a mock data
    cube and stores the output in fits format.

    Parameters
    ----------
    ConfigFile : location of the configuration file containing the input
        parameters needed.
    """
    __setup_logging()
    __aurora_version()

    # Code flow:
    # =====================
    # > Load the input parameters from ConfigFile
    # > Read the snapshot
    # > Set geometrical orientation and retain only the desired gas particles
    geom, run, spectrom = config.get_allinput(ConfigFile)
    data = snap.read_snap(run.input_file)
    data_gas = snap.set_snapshots_ready(geom, run, data)[0]
    del data
    gc.collect()
    
    # > Retain only those gas particles wich lie inside the field of view
    lim = spectrom.fieldofview.to("kpc").value/2.
    data_gas = snap.filter_array(data_gas,["x"],[-lim+geom.centerx.to("kpc").value],[lim+geom.centerx.to("kpc").value],2*["kpc"])
    data_gas = snap.filter_array(data_gas,["y"],[-lim+geom.centery.to("kpc").value],[lim+geom.centery.to("kpc").value],2*["kpc"])
    mask = (data_gas['u']<0)|(data_gas['mass']<0)|(data_gas['rho']<0)|(data_gas['temp']<0)
    data_gas = data_gas[~mask]

    # Code flow:
    # =====================
    # > Determine the smoothing lengths
    # > Increase target resolution to minimize geometrical concerns
    # > Compute the fluxes separately for each AMR scale
    # > Change the flux in the cube to avoid numerical errors.
    # > Smooth the fluxes from each scale and collapse them
    snap.set_hsml_limits(run, data_gas)
    spectrom.oversample()
    cube = spec.__project_all_chunks(geom, run, spectrom, data_gas)
    cube_max = np.floor(np.log10(cube.max()))
    cube_min = np.floor(np.log10(cube[cube>0].min()))
    contrast = cube_max - cube_min
    print(cube_max,cube_min)    
    if (spectrom.flux_cut_model != 'Not') and (contrast > 15):
        spec.flux_cut(spectrom, cube)
    else:
        print("Nothing to cut in the cube")
    if run.ncpu_convolution > 1:
        if run.convolution_parallel_method == 1:
            cube = spec.__cube_spatial_convolution_in_parallel_1(run, spectrom, cube)
        if run.convolution_parallel_method == 2:
            cube = spec.__cube_spatial_convolution_in_parallel_2(run, spectrom, cube)            
    else:        
        spec.__cube_spatial_convolution(run, spectrom, cube)
        cube = np.sum(cube, axis=3)
    spec.__cube_spectral_convolution(run, spectrom, cube)
    cube_max = np.floor(np.log10(cube.max()))
    cube_min = np.floor(np.log10(cube[cube>0].min()))
    print(cube_max,cube_min)    


    # Code flow:
    # =====================
    # > Bin to recover the target spatial sampling
    # > Inject noise
    # > Store the final datacube

    cube = arr.bin_array(cube, spectrom.oversampling, axis=1)
    cube = arr.bin_array(cube, spectrom.oversampling, axis=2)
    spectrom.undersample()

    if(spectrom.sigma_cont > 0.):
        logging.info(f"// Noise injection")
        cube_noise = cube + np.random.normal(0.0, spectrom.sigma_cont, cube.shape)

    logging.info(f"Created file {run.output_name}")
    if run.HSIM3 == True:
        so.writing_datacube_HSIM3(geom, spectrom, run, cube)
    else:        
        so.writing_datacube(geom, spectrom, run, cube)


def Simu_2D_projection(ConfigFile):
    """
    Map the estimated H-alpha or HI flux from the simulation to a mock data
    cube and stores the output in fits format.

    Parameters
    ----------
    ConfigFile : location of the configuration file containing the input
        parameters needed.
    """
    __setup_logging_spj()

    # Code flow:
    # =====================
    # > Load the input parameters from ConfigFile
    # > Read the snapshot
    # > Set geometrical orientation and retain only the desired gas particles
    geom, run, spectrom = config.get_allinput(ConfigFile)
    data = snap.read_snap(run.input_file)
    data_gas = snap.set_snapshots_ready(geom, run, data)[0]
    del data
    gc.collect()
    
    # > Retain only those gas particles wich lie inside the field of view
    lim = spectrom.fieldofview.to("kpc").value/2.
    data_gas = snap.filter_array(data_gas,["x","y"],2*[-lim],2*[lim],2*["kpc"])

    # Code flow:
    # =====================
    # > Determine the smoothing lengths
    # > Increase target resolution to minimize geometrical concerns
    # > Compute the fluxes separately for each AMR scale
    # > Smooth the fluxes from each scale and collapse them
    snap.set_hsml_limits(run, data_gas)

    logging.info(f"Start the proyection")
    proyection = spj.Projection_2D(data_gas)
    proyection.get_project_2D_histogram(run, spectrom)

    logging.info(f"Created file {run.output_name}")
    so.writing_projection(geom, spectrom, run, proyection.cube)


def degrade_mastercube(ConfigFile):
    """
    Degrade a master datacube to a lower resolution.

    Parameters
    ----------
    ConfigFile : location of the configuration file containing the input
        parameters needed.
    """

    __aurora_version()

    # Code flow:
    # =====================
    # > Load the input parameters from ConfigFile
    # > Load the master datacube and its header information
    # > Performs spatial and spectral convolutions
    # > Bin the spatial and spectral pixels
    geom, run, spectrom = config.get_allinput(ConfigFile)
    m = dc.DatacubeObj()
    m.read_data(run.input_file)
    m.get_attr()

    psf_fwhm = spectrom.spatial_res_kpc / m.pixsize.to("kpc")
    spec.__spatial_convolution(m.cube, psf_fwhm)
    psf_fwhm = ct.c.to("km s-1") / spectrom.spectral_res / m.velocity_sampl
    spec.__spectral_convolution(m.cube, psf_fwhm)

    m.spatial_degrade(geom, spectrom)
    m.spectral_degrade(geom, spectrom)

    # Aqui falta que *spectrom* absorba keywords del header de m
    geom.theta = m.header["THETA"] * unit.deg
    geom.phi = m.header["PHI"] * unit.deg
    run.simulation_id = m.header["SIMULAT"]
    run.snapshot_id = m.header["SNAPSHOT"]
    run.reference_id = m.header["SNAP_REF"]
    spectrom.redshift_ref = m.header["Z_REF"]

    so.writing_datacube(geom, spectrom, run, m.cube)

    # NOTE: If I want to degrade several without reading again over the
    # object m, for each configuration I can create a new object, and
    # copy the attributes m.cube and m.header => .gett_attr()
    # WARNING!!! Can NOT copy the object itself, because then modifying
    # the new one does change the original one

    # > ALTERNATIVE:
    # Resample by interpolation when the new pixel size is not an
    # integer multiple of the former one (we do not use it yet)
    # arr.cube_resampling(spectrom,m)
