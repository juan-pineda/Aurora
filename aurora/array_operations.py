import sys
import logging
import numpy as np
from scipy import ndimage

def bin_array(x, n, axis=0):
    """
    Bin the elements in one direction of a 2D or 3D array.
    
    Parameters
    ----------
    input_file : str
        Snapshot file name. It can be a simulation file RAMSES or GADGET.
    x : ndarray (3D or 2D)
        Array to be binned. 
    n : int
        Binning factor, i.e, number of elements to be added together
        must be a divisor of the number of elements on the axis.
    axis : int 
        Axis along which the binning operation takes place.
    
    Returns
    -------
    array : ndarray (3D or 2D)
        Binned Array.
    """

    # Code flow:
    # =====================
    # > Switch axes to work along axis 0 as default
    # > Perform the binning operation 
    x = np.swapaxes(x, 0, axis)
    if x.shape[0]%n != 0:
        logging.error(f"// n is not a divisor of the number of elements of the axis")
        sys.exit()
    dim = int(x.shape[0] / n)
    if len(x.shape) == 2:
        x = x[:dim*n, :]
        array = np.zeros([dim, x.shape[1]])
        for i in range(dim):
            array[i, :] = x[i*n:(i+1)*n, :].sum(axis = 0)
    elif len(x.shape) == 3:
        x = x[:dim*n, :, :]
        array = np.zeros([dim, x.shape[1], x.shape[2]])
        for i in range(n):
            array += x[i::n, :, :]
    array = np.swapaxes(array, 0, axis)
    return array


def cube_resampling(cube, new_cube):
    """
    Interpolate a 3D master datacube to new spatial/spectral coordinates
    at once.
    
    Parameters
    ----------
    cube : aurora.datacube.DatacubeObj
        Old datacube. Instance of class DatacubeObj whose attributes make 
        code computational performance properties available. See definitions
        in datacube.py
    
    new_cube : aurora.datacube.DatacubeObj
        New datacube. Must store information about spatial/spectral coordinates.
        Instance of class DatacubeObj whose attributes make code computational 
        performance properties available. See definitions in datacube.py
    """
    
    # Code flow:
    # =====================
    # > Calculate the transformation factor to the new spatial/spectral 
    #   coordinates
    # > Determines the spatial coordinates of the new system with respect to 
    #   the original
    # > Interpolates into the new data_cube    
    pixratio = cube.pixsize.value / new_cube.pixsize.value
    channelratio = cube.velocity_sampl.value/new_cube.velocity_sampl.value
    origin_spatial = (cube.spatial_dim - new_cube.spatial_dim/pixratio   
              - 1. + 1/pixratio) / 2
    new_positions = origin_spatial + np.arange(new_cube.spatial_dim)/pixratio
    origin_spectral = (cube.spectral_dim - new_cube.spectral_dim/channelratio
               - 1. + 1/channelratio) / 2
    new_channels = origin_spectral + np.arange(new_cube.spectral_dim)/channelratio
    X, Y, Z = np.meshgrid(new_positions, new_positions, new_channels)
    new_cube.cube = ndimage.map_coordinates(cube.cube, [Z, X, Y], order=1).T
    new_cube.cube = new_cube.cube / (pixratio**2 * channelratio)
