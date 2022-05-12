
import numpy as np
from astropy import units as unit
from aurora import constants as ct

def get_vel_dispersion(temp, mu):
    """
    Calculate the velocity dispersion of each particle in (cm s**-1), following the
    Maxwell-Boltzmann distribution.

    Returns
    -------
    dispersion : astropy.units.quantity.Quantity
        Velocity dispersion in (cm s**-1) for a bunch of particles.
    """

    sigma = np.sqrt(ct.k_B * temp / (mu * ct.m_p))
    dispersion = sigma.to("km s**-1")
    return dispersion

def position_in_pixels(pixsize,x,y,cube_side):
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

    x = (x / pixsize).decompose()
    y = (y / pixsize).decompose()
    x = (np.floor(x + cube_side / 2)).astype(int)
    y = (np.floor(y + cube_side / 2)).astype(int)


    # Due to errors in float claculations, eventually one particle may lie
    # just outside of the box
    x[x == cube_side] = (cube_side -1)
    x[x == -1] = 0
    y[y == cube_side] = (cube_side -1)
    y[y == -1] = 0

    #index = x + cube_side * y
    return x,y

def histogram_cube(x,y, x_bins, y_bins, dens, mass, temp, vel, dispersion_vel, pixsize, dim, cube):

#    min_celx = np.min(x)
 #   min_cely = np.min(y)
  #  lim = spectrom.fieldofview.to("pc").value/2
   # minbins_x = min_celx - int(np.abs(min_celx.value+lim)/pixsize.value)*pixsize
   # minbins_y = min_cely - int(np.abs(min_cely.value+lim)/pixsize.value)*pixsize
    #minbins_x = -lim - int(np.abs(min_celx.value+lim)/pixsize.value)*pixsize
   # minbins_y = min_cely - int(np.abs(min_cely.value+lim)/pixsize.value)*pixsize

    #x_bins = np.arange(-lim - pixsize.value/2, lim + pixsize.value/2, pixsize.value)
    #y_bins = np.arange(-lim - pixsize.value/2, lim + pixsize.value/2, pixsize.value)
    map_number = np.histogram2d(x.to("pc").value,y.to("pc").value, 
                bins=[x_bins,y_bins])[0]
    map_dens = np.histogram2d(x.to("pc").value,y.to("pc").value, 
                bins=[x_bins,y_bins], weights=dens)[0]
    #map_dens = map_dens/map_number
    map_mass = np.histogram2d(x.to("pc").value,y.to("pc").value, 
                bins=[x_bins,y_bins], weights=mass)[0]
   # map_mass = map_mass/map_number

    map_temp = np.histogram2d(x.to("pc").value,y.to("pc").value, 
                bins=[x_bins,y_bins], weights=temp*dens)[0]
  #  map_temp = map_temp/map_mass

    map_vel = np.histogram2d(x.to("pc").value,y.to("pc").value, 
                bins=[x_bins,y_bins], weights=vel*dens)[0]
  #  map_vel = map_vel/map_mass

    map_disp = np.histogram2d(x.to("pc").value,y.to("pc").value, 
                bins=[x_bins,y_bins], weights=dispersion_vel*dens)[0]
   # map_disp = map_disp/map_mass
    cube[0, :, :] = map_dens
    cube[1, :, :] = map_mass
    cube[2, :, :] = map_temp
    cube[3, :, :] = map_vel
    cube[4, :, :] = map_disp
    cube[5, :, :] = map_number


class Projection_2D:
    """
    Group the main parameters to compute the H-alpha emission of a 
    bunch of particles using the main physical quantities in the 
    simulation, and deriving other important physical quantites 
    from them.
    """
    
    # Get the main physical quantities in the simulation, converting pynbody
    # instances to astropy ones, to assure compatibility across operations.
    def __init__(self, data_gas):
        
        self.dens = np.array(data_gas["rho"].in_units("g cm**-3"))*unit.g/unit.cm**3
        self.mass = (np.array(data_gas["mass"].in_units("1.99e+43 g"))*
                               1.99e+43*unit.g).to("Msun")
        self.temp = np.abs(np.array(data_gas["temp"].in_units("K"))*unit.K)
        self.u = np.array(data_gas["u"].in_units("cm**2 s**-2"))*unit.cm**2/unit.s**2
        mask = (self.temp<0)|(self.mass<0)|(self.dens<0)
        self.dens = self.dens[~mask]
        self.mass = self.mass[~mask]
        self.temp = self.temp[~mask]
        self.u = self.u[~mask]
        self.N = len(data_gas[~mask])
        self.x = (np.array(data_gas["x"].in_units("pc"))*unit.pc)[~mask]
        self.y = (np.array(data_gas["y"].in_units("pc"))*unit.pc)[~mask]
        self.z = (np.array(data_gas["z"].in_units("pc"))*unit.pc)[~mask]
        self.vz = (np.array(data_gas["vz"].in_units("km s**-1"))*unit.km/unit.s)[~mask]
        self.smooth = (np.array(data_gas["smooth"].in_units("kpc"))*unit.kpc)[~mask]
        self.mu = np.array(data_gas['mu'])[~mask]
        self.dispersion_vel = get_vel_dispersion(self.temp, self.mu)
        del mask


    def __project_spectrom_flux(self, run, spectrom, data_gas, *args):
        """
        Compute the H-alpha emission of a bunch of particles and project it
        to a 4D grid, keeping contributions from different scales separated.

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
        data_gas : pynbody.snapshot.IndexedSubSnap 
		Gas particles identified in the input archive.
        *args : int, array_like
		Number of chunks to divide the gas particles, or the list of
		the upper and lower limits of the gas particles to be projected.
	    
        Returns
        -------
        cube : ndarray (4D)
		Contains the fluxes at each pixel and velocity channel 
		produced by the gas particles with a given smoothing
		lengths separately.
        """
	    
        scale = np.digitize(self.smooth.to("kpc").value,
                1.1 * run.fft_hsml_limits.to("kpc").value)
        pixsize = run.fft_hsml_limits[0].to("pc")
        ok_level = np.where(scale == pixsize)[0]
	    
        nupos_x = self.x[ok_level].to("pc")
        nupos_y = self.y[ok_level].to("pc")
        nupos_z = self.z[ok_level].to("pc")
        numass_t = self.mass[ok_level].to("Msun")
        nutemp_t = self.temp[ok_level].to("K")
        nudens_t = self.dens[ok_level].to("g cm^-3")
	        
        for n in range(len(run.fft_hsml_limits[1:])):
            ok_level = np.where(scale == n)[0]
            nok_level = ok_level.size
            n_cell_small = int(run.fft_hsml_limits.to("pc")[n]/pixsize.to("pc"))
            a = np.array([i-(n_cell_small-1)/2 for i in range(n_cell_small)])*pixsize.value
            nupos = np.zeros((nok_level*n_cell_small**3,3))
            X,Y,Z = np.meshgrid(a,a,a)
            shiftx=X.reshape(1,n_cell_small**3)[0]
            shifty=Y.reshape(1,n_cell_small**3)[0]
            shiftz=Z.reshape(1,n_cell_small**3)[0]
            #x
            u,d = np.meshgrid(shiftx,self.x[ok_level].to("pc").value)
            nupos[:,0]=((u+d)).reshape(1,
            shiftx.size*nok_level)[0]
            
            #y
            u,d = np.meshgrid(shifty,self.y[ok_level].to("pc").value)
            nupos[:,1]=((u+d)).reshape(1,
            shifty.size*nok_level)[0]

            #z
            u,d = np.meshgrid(shiftz,self.z[ok_level].to("pc").value)
            nupos[:,2]=((u+d)).reshape(1,
            shiftz.size*nok_level)[0]

            nupos = nupos*unit.pc

            #components
            #temp
            u,d = np.meshgrid(np.zeros(n_cell_small**3),
                          self.temp[ok_level])
            nutemp = (u+d.value).reshape(1,nok_level*n_cell_small**3)*unit.K
            #dens
            u,d = np.meshgrid(np.zeros(n_cell_small**3),
                          self.dens[ok_level])
            nudens = (u+d.value).reshape(1,nok_level*n_cell_small**3)*unit.g/unit.cm**3   
            #mas
            u,d = np.meshgrid(np.zeros(n_cell_small**3),
                          self.mass[ok_level]/n_cell_small**3)
            numass = (u+d.value).reshape(1,nok_level*n_cell_small**3)*unit.solMass

            #temp
            u,d = np.meshgrid(np.zeros(n_cell_small**3),
                          self.temp[ok_level])
            nutemp = (u+d.value).reshape(1,nok_level*n_cell_small**3)*unit.K
            #dens
            u,d = np.meshgrid(np.zeros(n_cell_small**3),
                          self.dens[ok_level])
            nudens = (u+d.value).reshape(1,nok_level*n_cell_small**3)*unit.g/unit.cm**3   
            

            nupos_x = np.concatenate((nupos_x, nupos[:,0]))
            nupos_y = np.concatenate((nupos_y, nupos[:,1]))
            nupos_z = np.concatenate((nupos_z, nupos[:,2]))
            del nupos
            numass_t = np.concatenate((numass_t,
                                      numass.reshape(nok_level*n_cell_small**3)))
            del numass_t
            nutemp_t = np.concatenate((nutemp_t,
                                      nutemp.reshape(nok_level*n_cell_small**3)))
            del nutemp_t
            nudens_t = np.concatenate((nudens_t,
		                          nudens.reshape(nok_level*n_cell_small**3)))

        return [nupos_x,nupos_y,nupos_z,numass_t,nutemp_t,nudens_t]
	    

    def get_project_2D_histogram(self, run, spectrom):
        """
        Compute the H-alpha emission of a bunch of particles and project it
        to a 4D grid, keeping contributions from different scales separated.

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
        data_gas : pynbody.snapshot.IndexedSubSnap 
		Gas particles identified in the input archive.
        *args : int, array_like
		Number of chunks to divide the gas particles, or the list of
		the upper and lower limits of the gas particles to be projected.
	    
        Returns
        -------
        cube : ndarray (4D)
		Contains the fluxes at each pixel and velocity channel 
		produced by the gas particles with a given smoothing
		lengths separately.
        """
        scale = np.digitize(self.smooth.to("kpc").value,
                1.1 * run.fft_hsml_limits.to("kpc").value)
        pixsize = run.fft_hsml_limits[0].to("pc")
        ok_level = np.where(self.smooth == pixsize)[0]

        dim = int((spectrom.fieldofview.to("pc")/pixsize.to("pc")).value)        	
        spectrom.spatial_dim = dim
        spectrom.pixsize = np.round(pixsize,2)

        min_celx = np.min(self.x[ok_level].to("pc").value)
        min_cely = np.min(self.y[ok_level].to("pc").value)
        lim = spectrom.fieldofview.to("pc").value/2
        minbins_x = min_celx - int(np.abs(min_celx+lim)/pixsize.value)*pixsize.value
        minbins_y = min_cely - int(np.abs(min_cely+lim)/pixsize.value)*pixsize.value

        x_bins = np.arange(minbins_x - pixsize.value/2, -minbins_x + pixsize.value/2, pixsize.value)
        y_bins = np.arange(minbins_y - pixsize.value/2, -minbins_y + pixsize.value/2, pixsize.value)
        self.cube = np.zeros((6, x_bins.size-1, y_bins.size-1, run.nfft))
        
        histogram_cube(self.x[ok_level], self.y[ok_level], x_bins, y_bins, self.dens[ok_level], self.mass[ok_level], 
                      self.temp[ok_level], self.vz[ok_level], self.dispersion_vel[ok_level], pixsize,
                      dim, self.cube[:,:,:,0])

        for n in range(1,len(run.fft_hsml_limits)):
            print(n)
            ok_level = np.where(self.smooth == run.fft_hsml_limits[n])[0]
            nok_level = ok_level.size
            n_cell_small = int(run.fft_hsml_limits.to("pc")[n]/pixsize.to("pc"))
            a = np.array([i-(n_cell_small-1)/2 for i in range(n_cell_small)])*round(pixsize.value,0)
            nupos = np.zeros((nok_level*n_cell_small**3,3))
            X,Y,Z = np.meshgrid(a,a,a)
            shiftx=X.reshape(1,n_cell_small**3)[0]
            shifty=Y.reshape(1,n_cell_small**3)[0]
            shiftz=Z.reshape(1,n_cell_small**3)[0]
            #x
            u,d = np.meshgrid(shiftx,self.x[ok_level].to("pc").value)
            nupos[:,0]=((u+d)).reshape(1,
            shiftx.size*nok_level)[0]
            
            #y
            u,d = np.meshgrid(shifty,self.y[ok_level].to("pc").value)
            nupos[:,1]=((u+d)).reshape(1,
            shifty.size*nok_level)[0]

            #z
            u,d = np.meshgrid(shiftz,self.z[ok_level].to("pc").value)
            nupos[:,2]=((u+d)).reshape(1,
            shiftz.size*nok_level)[0]

            nupos = nupos*unit.pc

            #components
            #temp
            u,d = np.meshgrid(np.zeros(n_cell_small**3),
                          self.temp[ok_level])
            nutemp = (u+d.value).reshape(nok_level*n_cell_small**3)*unit.K
            #dens
            u,d = np.meshgrid(np.zeros(n_cell_small**3),
                          self.dens[ok_level])
            nudens = (u+d.value).reshape(nok_level*n_cell_small**3)*unit.g/unit.cm**3   
            #mas
            u,d = np.meshgrid(np.zeros(n_cell_small**3),
                          self.mass[ok_level]/n_cell_small**3)
            numass = (u+d.value).reshape(nok_level*n_cell_small**3)*unit.solMass

            #vel
            u,d = np.meshgrid(np.zeros(n_cell_small**3),
                          self.vz[ok_level])
            nuvel = (u+d.value).reshape(nok_level*n_cell_small**3)*unit.km/unit.s
            #disp
            u,d = np.meshgrid(np.zeros(n_cell_small**3),
                          self.dispersion_vel[ok_level])
            nudis = (u+d.value).reshape(nok_level*n_cell_small**3)*unit.km/unit.s  
            print(numass.shape)            
            histogram_cube(nupos[:,0], nupos[:,1], x_bins, y_bins, nudens, numass, 
                      nutemp, nuvel, nudis, pixsize, dim, self.cube[:,:,:,n])
            del nupos
            del numass
            del nudens
            del nuvel
            del nudis
        self.cube = np.nansum(self.cube, axis = 3)
        
     #   return self.cube







































