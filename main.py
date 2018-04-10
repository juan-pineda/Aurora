from aurora import *
from aurora import datacube as dc
import numpy as np
import matplotlib.pyplot as plt

a = dc.DatacubeObj()
a.read_data('test.fits')
a.get_attr()
ins = a.intensity_map()
vel = a.velocity_map()
plt.imshow(vel, cmap='rainbow')
plt.show()
plt.imshow(np.log(ins), cmap='rainbow')
plt.show()
