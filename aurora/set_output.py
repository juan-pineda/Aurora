"""
==========
set_output
========== 

Methods to prepare the output files of the processed cubes.
"""

import os
import sys
import logging

from astropy.io import fits

from . import constants as ct
from . import spectrum_tools as spec


def set_output_filename(geom, run):
    """
    Set output name and tries to create the necessary directories recursively.
    
    Parameters
    ----------
    geom : aurora.configuration.GeometryObj
        Instance of class GeometryObj whose attributes make geometric 
        properties available. See definitions in configuration.py.
    run : aurora.configuration.RunObj
        Instance of class RunObj whose attributes make code computational
        performance properties available. See definitions in
        configuration.py.
        
    Returns
    -------
    output_name : str
        Name for the output file.
    """    
    
    # Code flow:
    # =====================
    # > Create an output dir
    # > Create a name for the output file
    
    # Check if there is an output dir, otherwise create it
    if(run.output_dir == ""):
        cwd = os.getcwd()
        path, input_file = os.path.split(os.path.realpath(run.input_file))
        run.output_dir = input_file.split(".")[0] + "_" + run.instrument
        run.output_dir = os.path.join(cwd, run.output_dir)
        if(geom.redshift < 0.01):
            z = str(int(geom.redshift * 1000) / 1000.)
        else:
            z = str(int(geom.redshift * 100) / 100.)
        theta = str(int(geom.theta.value))
        phi = str(int(geom.phi.value))
        params_dir = "z" + z + "_theta" + theta + "_phi" + phi
        run.output_dir = os.path.join(run.output_dir, params_dir)
    output_name = os.path.join(run.output_dir, run.output_file)
    # Checks if the intended output file exists, and whether it can be overwritten
    if (os.path.isfile(output_name) and not(run.overwrite)):
        logging.warning(f"// " + output_name +
              " already exists, and [overwrite=False] !!!")
        sys.exit()
    # Split the full output filename into subdirectories,
    # and tries to build the path as necessary
    output_name = os.path.realpath(output_name)
    path_list = output_name.split(os.sep)
    rebuild_path = "/"
    for subdir in path_list[:-1]:
        rebuild_path = os.path.join(rebuild_path, subdir)
        try:
            os.mkdir(rebuild_path)
        except:
            continue
    return output_name


def old_writing_datacube(geom, spectrom, dataset, output_name):
    hdu = fits.PrimaryHDU(dataset)
    hdulist = fits.HDUList([hdu])
    prihdr = hdu.header
    prihdr["NAXIS"] = 3
    prihdr["NAXIS3"] = spectrom.spectral_dim
    prihdr["BUNIT"] = "E16 ERG.S^-1.CM^-2.MICRONS^-1"
    prihdr["CRPIX3"] = int(spectrom.spectral_dim / 2.) + 1
    Halpha_em = (1.0 + geom.redshift) * ct.Halpha0
    prihdr["CRVAL3"] = Halpha_em.to("micron").value
    prihdr["CDELT3"] = spectrom.spectral_sampl.to("micron").value
    prihdr["CTYPE3"] = "WAVELENGTH"
    prihdr["CUNIT3"] = "MICRONS"
    prihdr["NAXIS1"] = spectrom.spatial_dim
    prihdr["NAXIS2"] = spectrom.spatial_dim
    prihdr["BSCALE"] = 1.0
    prihdr["BZERO"] = 0
    prihdr["CDELT1"] = -spectrom.spatial_sampl.to("deg").value
    prihdr["CDELT2"] = spectrom.spatial_sampl.to("deg").value
    prihdr["CD1_1"] = -spectrom.spatial_sampl.to("deg").value
    prihdr["CD1_2"] = 0.
    prihdr["CD2_1"] = 0.
    prihdr["CD2_2"] = spectrom.spatial_sampl.to("deg").value
    prihdr["CRVAL1"] = 0.
    prihdr["CRVAL2"] = 0.
    prihdr["CUNIT1"] = "DEG"
    prihdr["CUNIT2"] = "DEG"
    prihdr["CRPIX1"] = (spectrom.spatial_dim + 1) / 2.
    prihdr["CRPIX2"] = (spectrom.spatial_dim + 1) / 2.
    prihdr["CTYPE1"] = "RA---TAN"
    prihdr["CTYPE2"] = "DEC--TAN"
    prihdr["RA"] = 0.
    prihdr["DEC"] = 0.
    # Observation coordinate information
    prihdr["RADECSYS"] = "FK5"
    prihdr["EQUINOX"] = 2000.   
    hdulist.writeto(output_name, clobber=True)




def writing_datacube(geom, spectrom, run, dataset):
    """
    Create and write the output file in FITS format of the cube processed.
    Write the main information of the realistic mock observation (stored 
    in *geom*, *spectrom *and *run*) in the file header.
    
    Parameters
    ----------
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
    dataset : ndarray (3D)
        Processed cube. Contains the fluxes at each pixel and velocity 
        channel produced by the gas particles.
    """
    
    # Code clow
    # =====================
    # > Create the FITS file with the processed cube
    # > Write main information in FITS file header
    # > Save the final FITS file in the dir and with the supplied name
    hdu = fits.PrimaryHDU(dataset)
    hdulist = fits.HDUList([hdu])
    prihdr = hdu.header
    prihdr["NAXIS"] = 3
    prihdr["NAXIS3"] = spectrom.spectral_dim
    prihdr["BUNIT"] = "ERG.S^-1.cm^-2.KM^-1.S"
    prihdr["CRPIX3"] = (spectrom.channel_ref + 1,
                        "Center of the first pixel is 1")
    prihdr["CRVAL3"] = spectrom.vel_ref.to("km s-1").value
    prihdr["CDELT3"] = spectrom.velocity_sampl.to("km s^-1").value
    prihdr["CTYPE3"] = "VELOCITY"
    prihdr["CUNIT3"] = "KM.S^-1"
    prihdr["NAXIS1"] = spectrom.spatial_dim
    prihdr["NAXIS2"] = spectrom.spatial_dim
    prihdr["BSCALE"] = 1.0
    prihdr["BZERO"] = 0
    prihdr["CTYPE1"] = "X_POS"
    prihdr["CTYPE2"] = "Y_POS"
    prihdr["CDELT1"] = spectrom.pixsize.to("pc").value
    prihdr["CDELT2"] = spectrom.pixsize.to("pc").value
    prihdr["CD1_1"] = spectrom.pixsize.to("pc").value
    prihdr["CD1_2"] = 0.
    prihdr["CD2_1"] = 0.
    prihdr["CD2_2"] = spectrom.pixsize.to("pc").value
    prihdr["CRVAL1"] = spectrom.position_ref.to("pc").value
    prihdr["CRVAL2"] = spectrom.position_ref.to("pc").value
    prihdr["CUNIT1"] = "PC"
    prihdr["CUNIT2"] = "PC"
    prihdr["CRPIX1"] = (spectrom.pixel_ref + 1,
                        "Center of the first pixel is (1,1)")
    prihdr["CRPIX2"] = (spectrom.pixel_ref + 1,
                        "Center of the first pixel is (1,1)")
    prihdr["SIMULAT"] = (run.simulation_id, "Parent Simulation")
    prihdr["SNAPSHOT"] = run.snapshot_id
    prihdr["SNAP_REF"] = (
        run.reference_id, "Reference frame for geom. transf.")
    prihdr["THETA"] = (geom.theta.value, "Inclination angle in DEG")
    prihdr["PHI"] = (geom.phi.value, "Position angle in DEG")
    prihdr["Z_REF"] = (spectrom.redshift_ref,
                       "Redshift used to model HII fraction")
    prihdr["SPAT_RES"] = (spectrom.spatial_res_kpc.to(
        "kpc").value, "Spatial resolution in [kpc]")
    prihdr["SPEC_RES"] = (spectrom.spectral_res, "Spectral resolution, R")
    hdulist.writeto(run.output_name, clobber=True)
    hdulist.close()


def writing_maps(cube, dataset, datatype, output_name):
    """
    Create and write the output file in FITS format of the processed map.
    Write the main information of the processed map (stored in *cube*) 
    in the file header.
    
    
    Parameters
    ----------
    cube : aurora.datacube.DatacubeObj
        Instance of class DatacubeObj whose attributes make code computational
        performance properties available. See definitions in datacube.py
    dataset : ndarray (2D)
        Map generated to write.
    datatype : str
        Map type ("flux", "velocity", "dispersion").
    output_name : str
        Name for the output file.
    """
    
    # Code flow:
    # =====================
    # > Create the FITS file with the processed map
    # > Write main information in FITS file header
    # > Save the final FITS file in the dir and with the supplied output_name
    hdu = fits.PrimaryHDU(dataset)
    hdulist = fits.HDUList([hdu])
    prihdr = hdu.header
    prihdr["NAXIS"] = 2
    if datatype == "flux":
        prihdr["BUNIT"] = "ERG.S^-1.cm^-2"  
    elif datatype == "velocity":
        prihdr["BUNIT"] = "KM.S^-1"
    elif datatype == "dispersion":
        prihdr["BUNIT"] = "KM.S^-1"
    prihdr["NAXIS1"] = cube.spatial_dim
    prihdr["NAXIS2"] = cube.spatial_dim
    prihdr["BSCALE"] = 1.0
    prihdr["BZERO"] = 0
    prihdr["CTYPE1"] = "X_POS"
    prihdr["CTYPE2"] = "Y_POS"
    prihdr["CDELT1"] = cube.pixsize.to("pc").value
    prihdr["CDELT2"] = cube.pixsize.to("pc").value
    prihdr["CD1_1"] = cube.pixsize.to("pc").value
    prihdr["CD1_2"] = 0.
    prihdr["CD2_1"] = 0.
    prihdr["CD2_2"] = cube.pixsize.to("pc").value
    prihdr["CRVAL1"] = cube.position_ref.to("pc").value
    prihdr["CRVAL2"] = cube.position_ref.to("pc").value
    prihdr["CUNIT1"] = "PC"
    prihdr["CUNIT2"] = "PC"
    prihdr["CRPIX1"] = (cube.pixel_ref + 1, "Center of the first pixel is (1,1)")
    prihdr["CRPIX2"] = (cube.pixel_ref + 1, "Center of the first pixel is (1,1)")
    prihdr["SIMULAT"] = (cube.header["SIMULAT"], "Parent Simulation")
    prihdr["SNAPSHOT"] = cube.header["SNAPSHOT"]
    prihdr["SNAP_REF"] = (cube.header["SNAP_REF"],
                          "Reference frame for geom. transf.")
    prihdr["THETA"] = (cube.header["THETA"], "Inclination angle in DEG")
    prihdr["PHI"] = (cube.header["PHI"], "Position angle in DEG")
    prihdr["Z_REF"] = (cube.header["Z_REF"],
                       "Redshift used to model HII fraction")
    prihdr["SPAT_RES"] = (cube.header["SPAT_RES"], "Spatial resolution in [kpc]")
    prihdr["SPEC_RES"] = (cube.header["SPEC_RES"], "Spectral resolution, R")
    hdulist.writeto(output_name, clobber=True)
    hdulist.close()
