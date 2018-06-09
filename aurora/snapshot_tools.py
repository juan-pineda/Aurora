import gc
import sys
import pynbody
import numpy as np
from . import constants as ct

# Reads the simulation snapshot file


def read_snap(input_file):
    try:
        print("// Pynbody -> Reading file " + input_file)
        sys.stdout.flush()
        data = pynbody.load(input_file)
        return data
    except IOError:
        print('// The input file specified cannot be read')
        sys.exit()

# For a given array, and a given property (key), it drops all data outside some
# provided boundaries


def filter_array(data, prop, mini, maxi):
    if type(prop) == list:
        output = data
        for i in range(len(prop)):
            print("Filtering property: ", i,
                  "  , with min/max: ", mini[i], maxi[i])
            output = filter_array(output, prop[i], mini[i], maxi[i])
    else:
        ok = np.where((data[prop] >= mini) & (data[prop] <= maxi))
        if(ok[0].size == 0):
            print('// No data within specified boundaries')
            sys.exit()
        else:
            output = data[ok]
    return output

# Determines the number of scales in the data. If there are between
# 2 and 20 smoothing lengths, the *nfft* provided wil be superseded


def set_hsml_limits(run, data_gas):
    if(len(data_gas) > 0):
        smooth = np.unique(data_gas['smooth'])
        n_smooth = len(smooth)
        if((n_smooth > 1) & (n_smooth < 20)):
            run.fft_hsml_limits = np.sort(smooth).in_units('kpc')
            run.nfft = n_smooth
        else:
            run.fft_hsml_limits = np.arange(1.0, run.nfft + 1)
            run.fft_hsml_limits *= run.fft_hsml_min.to('kpc')
    else:
        print('No gas elements in this snapshot')
        sys.exit()
    print('// ' + str(run.nfft).strip() + ' levels for adaptive smoothing')

# Fix center and orientation of the disc, and filter out data if specified


def set_snapshots_ready(geom, run, data):
    # If a reference snapshot was specified, compute the transformations using that one first
    if geom.reference != '':
        data_ref = read_snap(geom.reference)
        if geom.barycenter == True:
            pynbody.analysis.angmom.config['centering-scheme'] = 'ssc'
            tr = pynbody.analysis.angmom.faceon(data_ref)
            tr.apply_to(data)
        # Set a coherent unit system
        data_ref.physical_units(velocity='cm s**-1',
                                distance='kpc', mass='1.99e+43 g')
        data.physical_units(velocity='cm s**-1',
                            distance='kpc', mass='1.99e+43 g')
        # Set the (inclination,position angle) of the disc
        print('// Inclination: Rotating along y ' + str(geom.theta).strip())
        print('// Positon angle: Rotating along z ' + str(geom.phi).strip())
        sys.stdout.flush()
        tr = data_ref.rotate_y(geom.theta.value)
        tr.apply_to(data)
        tr = data_ref.rotate_z(geom.phi.value)
        tr.apply_to(data)
        del data_ref
        gc.collect()
    else:
        if geom.barycenter == True:
            pynbody.analysis.angmom.config['centering-scheme'] = 'ssc'
            pynbody.analysis.angmom.faceon(data)
        # Set a coherent unit system
        data.physical_units(velocity='cm s**-1',
                            distance='kpc', mass='1.99e+43 g')
        # Set the (inclination,position angle) of the disc
        print('// Inclination: Rotating along y ' + str(geom.theta).strip())
        print('// Positon angle: Rotating along z ' + str(geom.phi).strip())
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
    if(geom.gas_minmax_keys != ''):
        data_gas = filter_array(
            data_gas, geom.gas_minmax_keys, geom.gas_min_values, geom.gas_max_values)
    # Apply filters to stellar particles if it was specified
    if(geom.star_minmax_keys != ''):
        data_star = filter_array(
            data_star, geom.star_minmax_keys, geom.star_min_values, geom.star_max_values)
    # Apply filters to DM particles if it was specified
    if(geom.dm_minmax_keys != ''):
        data_dm = filter_array(data_dm, geom.dm_minmax_keys,
                               geom.dm_min_values, geom.dm_max_values)
    # Informative prints
    nstars = len(data_star)
    ngas = len(data_gas)
    ndm = len(data_dm)
    print('// n_stars   -> ', nstars)
    print('// n_gas     -> ', ngas)
    print('// n_dm      -> ', ndm)
    return [data_gas, data_star, data_dm]

####################################
# WARNING IN THE NEXT TWO FUNCTIONS  !!!!!!!!!!!!!!!!!!!! WARNING WARNING WARNING
############################################################################################################
# Naive way of approximating the mean molecular weight
# this is CUSTOMIZED for Mirage project, for which there is no ElectroAbundance information stored !!!


def get_mean_weight(temp):
    mu = np.ones(len(temp))
    mu[np.where(temp >= 1e4)[0]] = 0.59
    mu[np.where(temp < 1e4)[0]] = 1.3
    return mu

# WARNING IN THE NEXT FUNCTION
# Slices a variable, removing the units information (it was problematic for some operations)


def slice_variable(data, var, start, stop, units=None):
    if var == 'temp':
        u = np.array(data['u'][start:stop], dtype=np.float64)
        mu = np.ones(len(u))
        m_p = ct.m_p.to('g').value
        k_B = ct.k_B.to('g cm2 s-2 K-1').value
        temp = (5./3 - 1) * mu * m_p * u / k_B
        mu = get_mean_weight(temp)
        return temp
    if units:
        return np.array(data[var][start:stop].in_units(units), dtype=np.float64)
    else:
        return np.array(data[var][start:stop], dtype=np.float64)
