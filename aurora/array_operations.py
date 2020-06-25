import sys
import math
import numpy as np
from scipy import ndimage


def bin_array(x, n, axis=0):
    """
    Bin the elements in one direction of a 2D or 3D array
    
    :param array x: Array to be binned
    :param int n: Binning factor, i.e, number of elements to be added together
                  must be a divisor of the number of elements on the axis
    :param int axis: Axis along which the binning operation takes place
    :return array array: Binned Array
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


def cube_resampling(spectrom, m):
    """
    Interpolate a 3D master datacube to new spatial/spectral coordinates
    at once.

    Parameters
    ----------
    """

    # New array to store the version of the mastercube spatially binned
    cube_side, n_ch = spectrom.cube_dims()
    cube = np.zeros((m.spectral_dim, cube_side, cube_side))
    # Step to augment 1 pixel of the new array, in units of original mastercube pixels
    pixratio = spectrom.pixsize/m.pixsize
    # position of the first pixel of the new array, in units of original mastercube pixels
    origin = (m.spatial_dim - spectrom.spatial_dim *
              pixratio + pixratio - 1.) / 2
    new_positions = origin + pixratio * np.arange(spectrom.spatial_dim)
    # And now the same thing for the spectral direction
    channelratio = spectrom.velocity_sampl/m.velocity_sampl
    origin = (m.spectral_dim - spectrom.spectral_dim *
              channelratio + channelratio - 1.) / 2
    new_channels = origin + channelratio * np.arange(spectrom.spectral_dim)
    X, Y, Z = np.meshgrid(new_positions, new_positions, new_channels)
    m.cube = ndimage.map_coordinates(m.cube, [X, Z, Y], order=1).T
###########
# One tricky part is missing
# What is the right normalization and units we want ???????
    # The total flux
####    cube = cube * pixratio**2
