# -*- coding: utf-8 -*-
"""Module for finding the neutral hydrogen in a halo. Each class has a function get_reproc_rhoHI, which returns
the neutral hydrogen density in *physical* atoms / cm^3
    Contains:
        GasProperties: converts reported code outputs to useful physical quantities.
                  get_temp: gets temperature from internal energy
                  get_code_rhoH: gets the neutral density from the code
                  get_reproc_HI - Gets a corrected neutral hydrogen density,
                  so that we are neutral even for star forming gas.
"""

import numpy as np
import logging
import scipy.interpolate.interpolate as intp
#import unitsystem


class GasProperties(object):
    """Class implementing the neutral fraction ala Rahmati 2012.
    Arguments:
        redshift - redshift at which the data was drawn, to compute the self-shielding correction.
        absnap - AbstractSnapshot instance from which to get the particle data.
        hubble - Hubble parameter.
        fbar - Baryon fraction
        units - UnitSystem instance
        sf_neutral - If True (the default) then gas on the star-forming equation of state is assumed to be neutral.
        Should only be true if used with a Springel-Hernquist star formation model in a version of Gadget/Arepo which incorrectly
        sets the neutral fraction in the star forming gas to less than unity.
        """
#    def __init__(self, redshift, absnap, hubble = 0.71, fbar=0.17, units=None, sf_neutral=True):

    def __init__(self, redshift, fbar=0.17):
        #        if units is not None:
        #            self.units = units
        #        else:
        #            self.units = unitsystem.UnitSystem()
        #        self.absnap = absnap
        self.f_bar = fbar
        self.redshift = redshift
#        self.sf_neutral = sf_neutral
        # Interpolate for opacity and gamma_UVB
        # Opacities for the FG09 UVB from Rahmati 2012.
        # IMPORTANT: The values given for z > 5 are calculated by fitting a power law and extrapolating.
        # Gray power law was: -1.12e-19*(zz-3.5)+2.1e-18 fit to z > 2.
        # gamma_UVB was: -8.66e-14*(zz-3.5)+4.84e-13
        # This is clearly wrong, but this model is equally a poor choice at these redshifts anyway.
        gray_opac = [2.59e-18, 2.37e-18, 2.27e-18, 2.15e-18,
                     2.02e-18, 1.94e-18, 1.82e-18, 1.71e-18, 1.60e-18]
        gamma_UVB = [3.99e-14, 3.03e-13, 6e-13, 5.53e-13,
                     4.31e-13, 3.52e-13, 2.678e-13,  1.81e-13, 9.43e-14]
        zz = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.redshift_coverage = True
        if redshift > zz[-1]:
            self.redshift_coverage = False
            logging.warning("no self-shielding at z=", redshift)
        else:
            gamma_inter = intp.interp1d(zz, gamma_UVB)
            gray_inter = intp.interp1d(zz, gray_opac)
            self.gray_opac = gray_inter(redshift)
            self.gamma_UVB = gamma_inter(redshift)
        # self.hy_mass = 0.76 # Hydrogen massfrac
        self.gamma = 5./3
        # Boltzmann constant (cgs)
        self.boltzmann = 1.38066e-16
#        self.hubble = hubble
        # Physical density threshold for star formation in H atoms / cm^3
#        self.PhysDensThresh = self._get_rho_thresh(hubble)

    def _photo_rate(self, nH, temp):
        """Photoionisation rate as  a function of density from Rahmati 2012, eq. 14.
        Calculates Gamma_{Phot}.
        Inputs: hydrogen density, temperature
            n_H
        The coefficients are their best-fit from appendix A."""
        nSSh = self._self_shield_dens(temp)
        photUVBratio = 0.98*(1+(nH/nSSh)**1.64)**-2.28+0.02*(1+nH/nSSh)**-0.84
        return photUVBratio * self.gamma_UVB

    def _self_shield_dens(self, temp):
        """Calculate the critical self-shielding density. Rahmati 202 eq. 13.
        gray_opac and gamma_UVB are parameters of the UVB used.
        gray_opac is in cm^2 (2.49e-18 is HM01 at z=3)
        gamma_UVB in 1/s (1.16e-12 is HM01 at z=3)
        temp is particle temperature in K
        f_bar is the baryon fraction. 0.17 is roughly 0.045/0.265
        Returns density in atoms/cm^3"""
        T4 = temp/1e4
        G12 = self.gamma_UVB/1e-12
        return 6.73e-3 * (self.gray_opac / 2.49e-18)**(-2./3)*(T4)**0.17*(G12)**(2./3)*(self.f_bar/0.17)**(-1./3)

    def _recomb_rate(self, temp):
        """The recombination rate from Rahmati eq A3, also Hui Gnedin 1997.
        Takes temperature in K, returns rate in cm^3 / s"""
        lamb = 315614./temp
        return 1.269e-13*lamb**1.503 / (1+(lamb/0.522)**0.47)**1.923

    def _neutral_fraction(self, nH, temp):
        """The neutral fraction from Rahmati 2012 eq. A8"""
        alpha_A = self._recomb_rate(temp)
        # A6 from Theuns 98
        LambdaT = 1.17e-10*temp**0.5 * \
            np.exp(-157809./temp)/(1+np.sqrt(temp/1e5))
        A = alpha_A + LambdaT
        B = 2*alpha_A + self._photo_rate(nH, temp)/nH + LambdaT
        return (B - np.sqrt(B**2-4*A*alpha_A))/(2*A)
