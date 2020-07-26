"""
=======
rahmati
=======

Aurora module that contains methods in charge of building the objects of the
class Rahmati_HII responsible for calculating the neutral hydrogen fraction 
using the self-shielding correction presented in (Rahmati et al 2013).  
"""

import logging
import numpy as np
import scipy.interpolate.interpolate as intp

from . import constants as ct

class Rahmati_HII:
    """
    Class implementing the self-shielding correction to calculate hydrogen 
    neutral fraction using the procedure exposed in (Rahmati et al 2013).  
        """

    def __init__(self, redshift, f_bar=0.17):
        """
        Sets the necessary parameters for the correction, by interpolation 
        according to the input redshift.
        
        Notes
        -----
        * The values of sigma_vHI and gamma_UVB < 5 in FG09 UVB from 
          (Rahmati et al 2013).
        * The values of sigma_vHI and gamma_UVB > 5 are calculated 
          by fitting a power law and extrapolating:
            > Power law for sigma_vHI: -1.12e-19*(z-3.5)+2.1e-18
            > Power law for gamma_UVB: -8.66e-14*(z-3.5)+4.84e-13
        
        Parameters
        ----------
        redshift : int or float
            redshift at which the data was drawn, to compute the 
            self-shielding correction.
        f_bar : float, optional
            Baryon fraction. The default value corresponds to the LambdaCDM cosmology. 
            We adopt fiducial cosmological parameters consistent with the most recent 
            WMAP 7-year results:  Omega_m = 0.272 y Omega_b = 0.0455. 
            f_bar = Omega_b / Omega_m.    
        """    
        
        # Code flow:
        # =====================
        # > Assign the necessary parameters.   
        # > Interpolates the values of sigma_vHI and gamma_UVB for the input redshift.  
        self.f_bar = f_bar
        self.redshift = redshift
        
        # Boltzmann constant in (erg/K)
        self.boltzmann = ct.k_B.value
        
        # The gray absorption cross-section in (cm**2)
        sigma_vHI = [2.59e-18, 2.37e-18, 2.27e-18, 2.15e-18,
                     2.02e-18, 1.94e-18, 1.82e-18, 1.71e-18, 1.60e-18] 
        
        # The hydrogen photoionization rate by the metagalactic ultra-violet
        # background radiation in (s**-1) .
        gamma_UVB = [3.99e-14, 3.03e-13, 6e-13, 5.53e-13,
                     4.31e-13, 3.52e-13, 2.678e-13,  1.81e-13, 9.43e-14]
        # Redshift range where the correction can be applied
        z = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.redshift_coverage = True
        
        if redshift > z[-1]:
            self.redshift_coverage = False
            logging.warning(f"no self-shielding at z=", redshift)
        else:
            gamma_inter = intp.interp1d(z, gamma_UVB)
            sigma_inter = intp.interp1d(z, sigma_vHI)
            self.sigma_vHI = sigma_inter(redshift)
            self.gamma_UVB = gamma_inter(redshift)
            
    def self_shield_num_dens(self, temp):
        """
        Calculate the critical self-shielding number density 
        - (Rahmati et al 2013) eq. 13.
        
        gray_opac and gamma_UVB are parameters of the UVB used.
        
        gray_opac is in cm^2 (2.49e-18 is HM01 at z=3)
        gamma_UVB in 1/s (1.16e-12 is HM01 at z=3)
        
        f_bar is the baryon fraction. 0.17 is roughly 0.045/0.265
        
        Parameters
        ----------
        temp : int or float
            Temperature in (K) for a bunch of particles.
        
        Returns
        -------
        n_Hssh : float 
            Critical self-shielding number density in (atoms cm**-3)
            for a bunch of particles.
        """
        
        T4 = temp/1e4
        G12 = self.gamma_UVB/1e-12
        n_Hssh = 6.73e-3 * (self.sigma_vHI/2.49e-18)**(-2./3) * (T4)**0.17 * (G12)**(2./
                 3)* (self.f_bar/0.17)**(-1./3) 
        return n_Hssh
    
    def recombination_rate(self, temp):
        """
        Calculate the recombination rate - (Rahmati et al 2013) eq. A3.
        
        Parameters
        ----------
        temp : int or float
            Temperature in (K) for a bunch of particles.
        
        Returns
        -------
        alpha_A: float 
            Recombination rate in (cm**3 s**-1) for a bunch of particles.        
        """
        
        lamb = 315614. / temp
        alpha_A = 1.269e-13 * lamb**1.503 / (1 + (lamb/0.522)**0.47)**1.923
        return alpha_A

    def photoionization_rate(self, temp, n_H):
        """
        Calculate the total photoionization rate - (Rahmati et al 2013) eq. 13.
               
        Parameters
        ----------
        temp : int or float
            Temperature in (K) for a bunch of particles.
        n_H : float
            Hydrogen number density in (cm**-3) for a bunch of particles.
        
        Returns
        -------
        photo_rate : float 
            Total photoionization rate in (s**-1) for a bunch of particles.            
        """
        
        # Code flow:
        # =====================
        # > Calculate the critical self-shielding number density.  
        # > Calculate the photoionization ratio. 
        # > Calculate the total photoionization rate.  
        n_Hssh = self.self_shield_num_dens(temp)
        phot_ratio = 0.98 * (1 + (n_H/n_Hssh)**1.64)**-2.28 + 0.02 * (1 + n_H/n_Hssh)**-0.84
        photo_rate = phot_ratio * self.gamma_UVB
        return photo_rate

    def neutral_fraction(self, n_H, temp):
        """
        Calculate the hydrogen neutral fraction, using the procedure exposed 
        in (Rahmati et al 2013) eq. A8.
        
        Parameters
        ----------
        temp : int or float
            Temperature in (K) for a bunch of particles.
        n_H : float
            Hydrogen number density in (cm**-3) for a bunch of particles.
        
        Returns
        -------
        n : float 
            Hydrogen neutral fraction for a bunch of particles.
        """
        
        # Code flow:
        # =====================
        # > Calculate the recombination rate. 
        # > Calculate the total photoionization rate.
        # > Calculate Lambda_T - (Rahmati et al 2013) eq. A6. 
        # > Calculate the coefficients A and B in Appendix A 
        #   - (Rahmati et al 2013).
        # > Calculate the hydrogen neutral fraction.        
        alpha_A = self.recombination_rate(temp)
        photo_rate = self.photoionization_rate(n_H, temp)
        
        Lambda_T = 1.17e-10 * temp**0.5 * \
            np.exp(-157809./temp) / (1 + np.sqrt(temp/1e5))
        
        A = alpha_A + Lambda_T
        B = 2 * alpha_A + photo_rate/n_H + Lambda_T
        n = (B - np.sqrt(B**2-4 * A * alpha_A)) / (2*A)
        return n
