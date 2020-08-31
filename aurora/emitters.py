"""
========
emitters
========

Aurora module that contains methods in charge of building the objects of the
class Emitters, responsible for grouping the main parameters to compute the
H-alpha emission of a bunch of particles using the main physical quantities
in the simulation, and deriving other important physical quantites from them.

Notes
-----
Calculations of the luminosity of the gas particles from the physical properties
stored in the simulation can be done in many different ways depending on the
keywords used in the configuration file, please review the code flow presented
below to make sure that the luminosity built into your run corresponds to the
proper logic.

Code flow:
==========
> Try to read the fraction of ionized hydrogen (HII) stored in the simulation.
> Try to read if the redshift_ref was stored in the ConfigFile.
> If redshif_ref is different from zero use the (Rahmati et al 2012) method.
> If redshift_ref is equal to zero use the stored HII.
> If redshift_ref is equal to zero and there is not stored HII info, use a 
  naive approximation of the HII coefficient setting it to zero for particles
  under 10e4 (K) and setting it to one for particles above that temperature.
> Use the fraction of ionized hydrogen to calculate mu, dens_ion and temp.
> Calculate the alphaH coeficient.
> Calculate the luminosity of each particle taking into account the dens_ion
  dependence stored in lum_dens_rel in the ConfigFile.
> Store the total luminosity, the central wavelenght and the broadening of
  each emision line for each gas particle.
"""


import numpy as np
from scipy import special
from astropy import units as unit

from . import constants as ct
from . import rahmati as rahmati

class Emitters:
    """
    Group the main parameters to compute the H-alpha emission of a 
    bunch of particles using the main physical quantities in the 
    simulation, and deriving other important physical quantites 
    from them.
    """
    
    # Get the main physical quantities in the simulation, converting pynbody
    # instances to astropy ones, to assure compatibility across operations.
    def __init__(self, data_gas, redshift=None):
        self.N = len(data_gas)
        self.data = data_gas
        self.redshift = redshift
        self.x = np.array(data_gas["x"].in_units("kpc"))*unit.kpc
        self.y = np.array(data_gas["y"].in_units("kpc"))*unit.kpc
        self.z = np.array(data_gas["z"].in_units("kpc"))*unit.kpc
        self.dens = np.array(data_gas["rho"].in_units("g cm**-3"))*unit.g/unit.cm**3
        self.vz = np.array(data_gas["vz"].in_units("cm s**-1"))*unit.cm/unit.s
        self.smooth = np.array(data_gas["smooth"].in_units("kpc"))*unit.kpc
        self.u = np.array(data_gas["u"].in_units("cm**2 s**-2"))*unit.cm**2/unit.s**2

    def get_state(self):
        """
        Calculate the temperature, the fraction of ionized hydrogen, the average
        molecular weight and the ions number density of a bunch of particles using
        the main physical quantitties in the simulation.

        Returns
        -------
        temp : astropy.units.quantity.Quantity
            Temperature in (K) for a bunch of particles.
        HII : ndarray
            Fraction of ionized hydrogen for a bunch of particles.
        mu : ndarray
            Average molecular weight for a bunch of particles.
        dens_ion : astropy.units.quantity.Quantity
            Ions number density in (cm**-3) for a bunch of particles.
        """
        # Code flow:
        # ==========
        # > Try to read the fraction of ionized hydrogen (HII) stored in the simulation.
        # > Try to read if the redshift_ref was stored in the ConfigFile.
        # > If redshif_ref is different from zero use the (Rahmati et al 2012) method.
        # > If redshift_ref is equal to zero use the stored HII.
        # > If redshift_ref is equal to zero and there is not stored HII info, use a 
        #   naive approximation of the HII coefficient setting it to zero for particles
        #   under 10e4 (K) and setting it to one for particles above that temperature.
        # > Use the fraction of ionized hydrogen to calculate mu, dens_ion and temp.
        
        if "HII" in self.data.keys():
            self.get_HII()
            self.get_mu()
            self.get_dens_ion()
            self.get_temp()
            if self.redshift != 0.:
                self.get_Rahmati_HII()
                self.get_dens_ion()
            else:
                pass
        
        elif "NeutralHydrogenAbundance" in self.data.loadable_keys():
            self.HII = 1 - self.data["NeutralHydrogenAbundance"]
            self.get_mu()
            self.get_dens_ion()
            self.get_temp()
            if self.redshift != 0.:
                self.get_Rahmati_HII()
                self.get_dens_ion()
            else:
                pass
        else:
            self.get_all_params()
            if self.redshift != 0.:
                self.get_Rahmati_HII()
                self.get_dens_ion()
            else:
                pass
        
    def get_luminosity(self, mode):
        """
        Calculate the H-alpha emission for each particle in (erg s**-1), based
        on the alphaH coefficient with different ions number density dependence.
        
        Parameters
        ----------
        mode : str
            Stablish the ions density dependence for H-alpha 
            emission calculation. Can be 'square', 'linear'
            or 'root'.
        
        Returns
        -------
        Halpha_lum : astropy.units.quantity.Quantity
            H-alpha emission in (erg s**-1) for a bunch of particles.
        """
        
        self.get_alphaH()
        luminosity = (self.smooth)**3 * (ct.h*ct.c/ct.Halpha0) * self.alphaH
        if mode == "square":
            Halpha_lum = luminosity * (self.dens_ion)**2
        elif mode == "linear":
            Halpha_lum = luminosity * (self.dens_ion.value) * unit.cm**-6
        elif mode == "root":
            Halpha_lum = luminosity * (self.dens_ion.value)**0.5 * unit.cm**-6
        self.Halpha_lum = Halpha_lum.to("erg s**-1")

    def density_cut(self, density_threshold = "Not", equivalent_luminosity = "min"):
        """
        Replaces the H-alpha emission for an equivalent luminosity, for certain
        gas particles that exceed the established density threshold.
        
        Parameters
        ----------
        density_threshold : str or float, optional
            For 'polytrope' the density cut function will apply a threshold
            for the density of particles based on the polytrope equation.
            For a float value, the density threshold will be calculated using
            the float input as a power of ten and (6.77e-23 g cm**-3) as units.
        equivalent_luminosity : str or float, optional
            For 'min' the equivalent luminosity will be set as the minimun value
            of the H-alpha emission. For a float value, the equivalent
            luminosity will be set as the float input in (erg s**-1).
            
        Returns
        -------
        Halpha_lum : astropy.units.quantity.Quantity
            H-alpha emission in (erg s**-1) for a bunch of particles.
        """
        
        if density_threshold == "Not":
            print("Nothing to cut")
        elif density_threshold == "polytrope":
            print("polytropic cut to be implemented")
            logdens = np.log10(self.dens.to("6.77e-23 g cm**-3").value)
            logtemp = np.log10(self.temp.to("K").value)
            tokill = ((logdens > logtemp - 3.5) & (logdens > 0.7))
            if equivalent_luminosity == "min":
                self.Halpha_lum[tokill] = np.min(self.Halpha_lum)
            else:
                self.Halpha_lum[tokill] = np.float(equivalent_luminosity) * unit.erg * unit.s**-1
        else:
            thresh = 10**np.float(density_threshold)
            print("Cutting a threshold: ",thresh)
            tokill = (self.dens.to("6.77e-23 g cm**-3").value > thresh)
            if equivalent_luminosity == "min":
                self.Halpha_lum[tokill] = np.min(self.Halpha_lum)
            else:
                self.Halpha_lum[tokill] = np.float(equivalent_luminosity) * unit.erg * unit.s**-1

    def get_vel_dispersion(self):
        """
        Calculate the velocity dispersion of each particle in (cm s**-1), following the
        Maxwell-Boltzmann distribution.

        Returns
        -------
        dispersion : astropy.units.quantity.Quantity
            Velocity dispersion in (cm s**-1) for a bunch of particles.
        """
        
        sigma = np.sqrt(ct.k_B * self.temp / (self.mu * ct.m_p))
        self.dispersion = sigma.to("cm s**-1")

    def get_all_params(self):
        """
        Calculate the temperature, the fraction of ionized hydrogen, the average
        molecular weight and the ions number density of a bunch of particles, as
        a first aproximation for input archives for which there is no electron
        abundance information stored.

        Returns
        -------
        temp : astropy.units.quantity.Quantity
            Temperature in (K) for a bunch of particles.
        HII : ndarray
            Fraction of ionized hydrogen for a bunch of particles.
        mu : ndarray
            Average molecular weight for a bunch of particles.
        dens_ion : astropy.units.quantity.Quantity
            Ions number density in (cm**-3) for a bunch of particles.
        """
        
        mu = np.ones(self.N)
        dens_ion = np.ones(self.N)
        for i in range(5):
            temp = (5./3 - 1) * mu * ct.m_p * self.u / ct.k_B
            mu = self.get_mean_weight(temp)
            HII = self.get_fraction_ionized_H(temp)
        self.temp = temp.decompose().to("K")
        self.mu = mu
        self.HII = HII
        self.get_dens_ion()

    def get_mean_weight(self, temp):
        """
        Calculate the average molecular weight of a bunch of particles,
        as a first aproximation for input archives for which there is no
        electron abundance information stored.
        
        Parameters
        ----------
        temp : astropy.units.quantity.Quantity
            Temperature in (K) for a bunch of particles.

        Returns
        -------
        mu : ndarray
            Average molecular weight of ionized hydrogen for a bunch of particles.
        """
        
        mu = np.ones(len(temp))
        mu[np.where(temp >= 1e4*unit.K)[0]] = 0.63
        mu[np.where(temp < 1e4*unit.K)[0]] = 1.22
        return mu
    
    def get_fraction_ionized_H(self, temp):
        """
        Calculate fraction of ionized hydrogen of a bunch of particles, 
        as a first aproximation for input archives for which there is
        no electron abundance information stored.
        
        Parameters
        ----------
        temp : astropy.units.quantity.Quantity
            Temperature in (K) for a bunch of particles.

        Returns
        -------
        HII : ndarray
            Fraction of ionized hydrogen for a bunch of particles.
        """
        
        HII = np.ones(len(temp))
        HII[np.where(temp >= 1e4*unit.K)[0]] = 1.
        HII[np.where(temp < 1e4*unit.K)[0]] = 0.
        return HII

    def get_HII(self):
        """
        Calculate the fraction of ionized hydrogen of each particle, using the values
        stored in the simulation.

        Returns
        -------
        HII : ndarray
            Fraction of ionized hydrogen for a bunch of particles.
        """
    
        HII = self.data["HII"]
        self.HII = HII
    
    def get_Rahmati_HII(self):
        """
        Calculate the fraction of ionized hydrogen of each particle, using the procedure
        exposed in (Rahmati et al 2012).

        Returns
        -------
        HII : ndarray
            Fraction of ionized hydrogen for a bunch of particles.
        """

        a = rahmati.Rahmati_HII(self.redshift)
        HII = 1 - a.neutral_fraction(ct.Xh * (self.dens.to("g cm**-3")/ct.m_p.to("g")).value, self.temp.value)
        self.HII = HII

    def get_mu(self):
        """
        Calculate the average molecular weight of a bunch of particles, using
        the hydrogen cosmological fraction to establish the relationship between
        the helium and hydrogen spices in each particle and asuming ionization
        only for hydrogen.

        Returns
        -------
        mu : ndarray
            Average molecular weight for a bunch of particles.
        """
        
        mu = 4. / (3*ct.Xh + 1 + 4*self.HII*ct.Xh)
        self.mu = mu
    
    def get_temp(self):
        """
        Calculate the temperature of each particle in (K), using the internal
        energy and the average molecular weight.

        Returns
        -------
        temp : astropy.units.quantity.Quantity
            Temperature in (K) for a bunch of particles.
        """
        
        temp = (5./3 - 1) * self.mu * ct.m_p * self.u / ct.k_B
        self.temp = temp.decompose().to("K")  
    
    def get_dens_ion(self):
        """
        Calculate the ions number density of a bunch of particles in (cm**-3) using
        the hydrogen cosmological fraction as a constant, the fraction of ionized 
        hydrogen and the density of each particle.

        Returns
        -------
        dens_ion : astropy.units.quantity.Quantity
            Ions number density in (cm**-3) for a bunch of particles.
        """
        
        self.dens_ion = (self.dens * self.HII * ct.Xh / ct.m_p)

    def get_alphaH(self):
        """
        Calculate the case-B Hydrogen effective recombination rate coefficient,
        taking in to account the temperature of each particle - Osterbrock & 
        Ferland (2006).
        
        Returns
        -------
        alphaH : astropy.units.quantity.Quantity
            Hydrogen effective recombination rate coefficient in (cm3/s) for a 
            bunch of particles.
        """
        
        self.alphaH = ct.alphaH.to("cm3/s")*(self.temp.to("K").value / 1.0e4)**-0.845

    def get_vect_lines(self, n_ch):
        """
        Array the velocity in Z axis as the line center, the velocity dispersion
        as the line broadening and the H-alpha emission as the line flux, each one
        stored in a matrix where each row is a particle, and columns will serve to
        store fluxes at each of the cube spectral channels.
        
        Parameters
        ----------
        n_ch : int
            Number of spectral channels.        
        
        Returns
        -------
        line_center : astropy.units.quantity.Quantity
            Line center of the emision process of each particle, determined by the
            velocity in Z axis.
        line_sigma : astropy.units.quantity.Quantity
            Broadening of the line in the emision process of each particle,
            determined by the velocity dispersion.
        line_flux : astropy.units.quantity.Quantity
            Total flux produced in the emision process of each particle, determined
            by the H-alpha emission.
        
        Examples
        --------
        With n particles centered at l1, l2 ..., ln, line_center is:
        
        [ l1 l1 l1 ... l1
          l2 l2 l2 ... l2
          .  .  .  ...
          .  .  .  ...
          ln ln ln ... ln]
        """
        
        line_center = np.transpose(np.tile(self.vz, (n_ch, 1)))
        line_sigma = np.transpose(np.tile(self.dispersion, (n_ch, 1)))
        line_flux = np.transpose(np.tile(self.Halpha_lum, (n_ch, 1)))
        return line_center, line_sigma, line_flux

    def get_vect_channels(self, channels, width, n_ch):
        """
        Array the center and the width of each spectral channel, saving each of 
        these properties in a matrix where each row is a particle, and columns
        will serve to store the respective spectral channel information.
        
        Parameters
        ----------
        channels: astropy.units.quantity.Quantity
            Central values for the spectral channels.
        width: astropy.units.quantity.Quantity
            Constant width for the spectral channels.
        n_ch : int
            Number of spectral channels.        
        
        Returns
        -------
        channel_center : astropy.units.quantity.Quantity
            Spectral channel center for each particle.
        channel_width : astropy.units.quantity.Quantity
            Spectral channel width for each particle and for each spectral 
            channel.
        
        Examples
        --------
        With n particles, m spectral channels and X constant width channel
        value, channel_width is:
        
        [ (X)11 (X)12 (X)13 ... (X)1m
          (X)21 (X)22 (X)23 ... (X)2m
            .     .     .   ...
            .     .     .   ...
          (X)n1 (X)n2 (X)n3 ... (X)nm]
        """
        
        channel_center = np.tile(channels, (self.N, 1))
        channel_width = np.tile(width, (self.N, n_ch))
        return channel_center, channel_width

    def int_gaussian(self, x, dx, mean, sigma):
        """
        Compute the integral of a normalized gaussian inside some limits, using
        parameters without units.
    
        Parameters
        ----------
        x : float, ndarray
            Central position of the interval.
        dx : float, ndarray
            Width of the interval.
        mean : float, ndarray
            Mean of the gaussian.
        sigma : float, ndarray
            Standard deviation.
        
        Returns
        -------
        value : float, ndarray
            Value of the normalized gaussian integral inside the limits.
        """

        A = special.erf((x+dx/2-mean)/np.sqrt(2)/sigma)
        B = special.erf((x-dx/2-mean)/np.sqrt(2)/sigma)
        value = np.abs((A-B)/2)
        return value

    # Notice that, because *int_gaussian* integrates a normalized gaussian,
    # the units of the input parameters do not affect the result as far as all
    # of them are the same
    def int_gaussian_with_units(self, x, dx, mean, sigma):
        """
        Compute the integral of a normalized gaussian inside some limits, using
        parameters with units.
    
        Parameters
        ----------
        x : float, ndarray
            Central position of the interval.
        dx : float, ndarray
            Width of the interval.
        mean : float, ndarray
            Mean of the gaussian.
        sigma : float, ndarray
            Standard deviation.
        
        Returns
        -------
        inte : float, ndarray
            Value of the normalized gaussian integral inside the limits.
        """
        
        dx = dx.to(x.unit)
        mean = mean.to(x.unit)
        sigma = sigma.to(x.unit)
        inte = self.int_gaussian(x.value, dx.value, mean.value, sigma.value)
        return inte