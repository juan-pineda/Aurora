"""
.. py:module:: snapshot_tools

Methods that prepare the snapshot to build the synthetic cube.
"""

import gc
import sys
import logging
import numpy as np

import pynbody
from astropy import units as unit

from . import constants as ct

def read_snap(input_file):
    """
    Reads the simulation snapshot file.
    
    :param str input_file: File name
    :return data: data snapshot file
    :type data: pynbody.snapshot
    """
    
    try:
        logging.info(f"// Pynbody -> Reading file {input_file}")
        sys.stdout.flush()
        data = pynbody.load(input_file)
        return data
    except IOError:
        logging.error(f"// The input file specified cannot be read")
        sys.exit()

def filter_array(data, prop, mini, maxi, units):
    """
    For a given array, and a given property (key), it drops all data outside some
    provided boundaries.
    
    :param data: Array to filter
    :type data: pynbody.snapshot.FamilySubSnap
    :param prop: Propierty (key)
    :type prop: list[str] or str
    :param mini: Lower boundaries
    :type mini: list[int, float], floar or int
    :param maxi: Upper boundaries
    :type maxi: list[int, float], floar or int
    :type data: pynbody.snapshot
    :return output: Filtered array
    :type output: pynbody.snapshot.FamilySubSnap
    """
   
    if type(prop) == list:
        output = data
        for i in range(len(prop)):
            logging.info(f"Filtering property: {prop[i]}, with min/max: {mini[i]},{maxi[i]}, in units {units[i]}")
            output = filter_array(output, prop[i], mini[i], maxi[i], units[i])
    else:
        ok = np.where((data[prop].in_units(units) >= mini) & (data[prop].in_units(units) <= maxi))
        if(ok[0].size == 0):
            logging.error(f"// No data within specified boundaries")
            sys.exit()
        else:
            output = data[ok]
    return output

def set_hsml_limits(run, data_gas):
    """
    Determines the number of scales in the data. If there are between
    2 and 20 smoothing lengths, the *nfft* provided will be superseded.
    If the smoothing lengths are greater than 20, the *nfft* and 
    *fft_hsml_min* must be defined.
    
    :param run: run object
    :type run: aurora.configuration.RunObj
    :param data_gas: Gas array
    :type data_gas: pynbody.snapshot.FamilySubSnap
    """
    
    # Code flow:
    # =====================
    # > Find the unique elements of data_gas["smooth"]
    # > Assign the smoothing lengths limit values as the case may be
    if(len(data_gas) > 0):
        smooth = np.unique(data_gas["smooth"])
        n_smooth = len(smooth)
        if((n_smooth > 1) & (n_smooth < 20)):
            run.fft_hsml_limits = smooth.in_units("kpc")*unit.kpc
            run.nfft = n_smooth
        else:
            run.fft_hsml_limits = np.arange(1.0, run.nfft + 1)
            run.fft_hsml_limits = run.fft_hsml_limits * run.fft_hsml_min.to("kpc") * unit.kpc
    else:
        logging.error(f"No gas elements in this snapshot")
        sys.exit()
    logging.info(f"// {str(run.nfft).strip()} levels for adaptive smoothing")

def set_snapshots_ready(geom, run, data):
    """
    Fix center and orientation of the disc, separates the data file into 
    three types (*star*, *gas* and *dark matter*) and filter gas data 
    if specified.
    
    :param geom: geom object
    :type geom: aurora.configuration.GeometryObj
    :param run: run object
    :type run: aurora.configuration.RunObj
    :param data: data snapshot file
    :type data: pynbody.snapshot
    :return data_gas: Gas array
    :type data_gas: pynbody.snapshot.FamilySubSnap
    :return data_star: Star array
    :type data_star: pynbody.snapshot.FamilySubSnap
    :return data_dm: Dark matter array
    :type data_dm: pynbody.snapshot.FamilySubSnap
    """
    
    # Code flow:
    # =====================
    # > Center and apply the transformation to the snapshot 
    # > Assign the particle families of the snapshot in different instances
    # > Filters gas particles
    
    # If a reference snapshot was specified, compute the transformations using that one first
    if geom.reference != "":
        data_ref = read_snap(geom.reference)
        if geom.barycenter == True:
            pynbody.analysis.angmom.config["centering-scheme"] = "ssc"
            tr = pynbody.analysis.angmom.faceon(data_ref)
            tr.apply_to(data)
        # Set a coherent unit system
        data_ref.physical_units(velocity="cm s**-1",
                                distance="kpc", mass="1.99e+43 g")
        data.physical_units(velocity="cm s**-1",
                            distance="kpc", mass="1.99e+43 g")
        # Set the (inclination,position angle) of the disc
        logging.info(f"// Inclination: Rotating along y {str(geom.theta).strip()}")
        logging.info(f"// Positon angle: Rotating along z {str(geom.phi).strip()}")
        sys.stdout.flush()
        tr = data_ref.rotate_y(geom.theta.value)
        tr.apply_to(data)
        tr = data_ref.rotate_z(geom.phi.value)
        tr.apply_to(data)
        del data_ref
        gc.collect()
    else:
        if geom.barycenter == True:
            pynbody.analysis.angmom.config["centering-scheme"] = "ssc"
            pynbody.analysis.angmom.faceon(data)
        # Set a coherent unit system
        data.physical_units(velocity="cm s**-1",
                            distance="kpc", mass="1.99e+43 g")
        # Set the (inclination,position angle) of the disc
        logging.info(f"// Inclination: Rotating along y {str(geom.theta).strip()}")
        logging.info(f"// Positon angle: Rotating along z {str(geom.phi).strip()}")
        sys.stdout.flush()
        data.rotate_y(geom.theta.value)
        data.rotate_z(geom.phi.value)
    # Free some memory allocation
    data_gas = data.gas
    data_star = data.star
    data_dm = data.dm
    del data
    gc.collect()
    # Apply filters to gas particles if it was specified
    if(geom.gas_minmax_keys != ""):
        data_gas = filter_array(
            data_gas, geom.gas_minmax_keys, geom.gas_min_values, geom.gas_max_values)
    # Informative prints
    nstars = len(data_star)
    ngas = len(data_gas)
    ndm = len(data_dm)
    logging.info(f"// n_stars   -> {nstars}")
    logging.info(f"// n_gas     -> {ngas}")
    logging.info(f"// n_dm      -> {ndm}")
    return [data_gas, data_star, data_dm]
