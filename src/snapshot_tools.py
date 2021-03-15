"""
==============
snapshot_tools
==============

This module contains the methods that prepare the snapshot to 
build the synthetic cube.
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
    
    Parameters
    ----------
    input_file : str
        Snapshot file name. It can be a simulation file RAMSES or GADGET.
    
    Returns
    -------
    data : pynbody.snapshot
        All data of the snapshot file. Includes the header information and particle
        information in the simulation.
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
    
    
    Parameters
    ----------    
    data : pynbody.snapshot.FamilySubSnap 
        Array of particles (*gas*, *star*, *dark matter*) to filter. 
     prop : list[str] or str
        Property (key) on which the filter is applied.
    mini : list[int, float], floar or int
        Lower boundaries.
    maxi : list[int, float], floar or int
        Upper boundaries.
        
    Returns
    -------
    output : pynbody.snapshot.FamilySubSnap
        Filtered particle array.
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
    
    Parameters
    ----------
    run : aurora.configuration.RunObj
        Instance of class RunObj whose attributes make code computational
        performance properties available. See definitions in
        configuration.py.
    data_gas : pynbody.snapshot.IndexedSubSnap 
        Gas particles identified in the input file.
    """
    
    # Code flow:
    # =====================
    # > Find the unique elements of data_gas["smooth"]
    # > Assign the smoothing lengths limit values as the case may be
    if run.fft_scales != "Not":
        run.fft_hsml_limits = np.loadtxt(run.fft_scales)*unit.kpc
        run.nfft = len(np.loadtxt(run.fft_scales)+1)
    else:
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
    
    Parameters
    ----------
    geom : aurora.configuration.GeometryObj
        Instance of class GeometryObj whose attributes make geometric 
        properties available. See definitions in configuration.py.
    run : aurora.configuration.RunObj
        Instance of class RunObj whose attributes make code computational
        performance properties available. See definitions in
        configuration.py.
    data : pynbody.snapshot
        All data of the snapshot file. Includes the header information and particle
        information in the simulation.
    
    Returns
    -------
    data_gas : pynbody.snapshot.IndexedSubSnap 
        Gas particles identified in the input file.
    data_star : pynbody.snapshot.IndexedSubSnap 
        Star particles identified in the input file.
    data_dm : pynbody.snapshot.IndexedSubSnap 
        Dark matter particles identified in the input file.
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
        # Note: The rotation must be applied for each family of particles in order to 
        # have compatibility with gadget simulations.
        tr = data_ref.rotate_y(geom.theta.value)
        tr.apply_to(data.gas)
        tr.apply_to(data.star)
        tr.apply_to(data.dm)
        tr = data_ref.rotate_z(geom.phi.value)
        tr.apply_to(data.gas)
        tr.apply_to(data.star)
        tr.apply_to(data.dm)
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
        
        # Note: The rotation must be applied for each family of particles in order to 
        # have compatibility with gadget simulations.
        data.gas.rotate_y(geom.theta.value)
        data.star.rotate_y(geom.theta.value)
        data.dm.rotate_y(geom.theta.value)
        
        data.gas.rotate_z(geom.phi.value)
        data.star.rotate_z(geom.phi.value)
        data.dm.rotate_z(geom.phi.value)        
    
    # Free some memory allocation
    data_gas = data.gas
    data_star = data.star
    data_dm = data.dm
    del data
    gc.collect()
    # Apply filters to gas particles if it was specified
    if(geom.gas_minmax_keys != ""):
        data_gas = filter_array(
            data_gas, geom.gas_minmax_keys, geom.gas_min_values, geom.gas_max_values, [geom.gas_minmax_units])
    # Informative prints
    nstars = len(data_star)
    ngas = len(data_gas)
    ndm = len(data_dm)
    logging.info(f"// n_stars   -> {nstars}")
    logging.info(f"// n_gas     -> {ngas}")
    logging.info(f"// n_dm      -> {ndm}")
    return [data_gas, data_star, data_dm]
