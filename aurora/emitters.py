import pynbody
import numpy as np
from astropy import units as unit

from . import constants as ct
from . import gasProps_sBird as bird

class Emitters:

	# Get the main physical quantities in the simulation, converting pynbody
	# instances to astropy ones, to assure compatibility across operations
	def __init__(self, data_gas,redshift=None):
		self.N = len(data_gas)
		self.data = data_gas
		self.redshift = redshift
		self.x = np.array(data_gas['x'].in_units('kpc'))*unit.kpc
		self.y = np.array(data_gas['y'].in_units('kpc'))*unit.kpc
		self.z = np.array(data_gas['z'].in_units('kpc'))*unit.kpc
		self.dens = np.array(data_gas['rho'].in_units('g cm**-3'))*unit.g/unit.cm**3
		self.vz = np.array(data_gas['vz'].in_units('cm s**-1'))*unit.cm/unit.s
		self.smooth = np.array(data_gas['smooth'].in_units('kpc'))*unit.kpc
		self.u = np.array(data_gas['u'].in_units('cm**2 s**-2'))*unit.cm**2/unit.s**2

	# Derived physical quantities
	def get_state(self):
		self.get_temp()
		self.get_HII()
		self.get_mu()
		self.get_dens_ion()

	def get_luminosity(self):
		self.get_alphaH()
		Halpha_lum = (self.smooth)**3 * (self.dens_ion)**2 * (ct.h*ct.c/ct.Halpha0) * self.alphaH 
		self.Halpha_lum = Halpha_lum.to('erg cm AA**-1 s**-1')
				
	def get_vel_dispersion(self):
		sigma = np.sqrt(ct.k_B * self.temp / (self.mu * ct.m_p))
		self.dispersion = sigma.to('cm s**-1')

	def get_temp(self):
		mu = np.ones(self.N)
		for i in range(5):
			temp = (5./3 - 1) * mu * ct.m_p * self.u / ct.k_B
			mu = self.get_mean_weight(temp)
		self.temp = temp.decompose().to('K')

	# Naive way of approximating the mean molecular weight
	# this is CUSTOMIZED for Mirage project, for which there is no
	# ElectroAbundance information stored !!!
	def get_mean_weight(self,temp):
	    mu = np.ones(len(temp))
	    mu[np.where(temp >= 1e4*unit.K)[0]] = 0.59
	    mu[np.where(temp < 1e4*unit.K)[0]] = 1.3
	    return mu

	def get_HII(self):
		if 'HII' in self.data.keys():
			HII = self.data['HII']
		else:
			a = bird.GasProperties(self.redshift)
			HII = 1 - a._neutral_fraction(ct.Xh * (self.dens.to('g cm**-3')/ct.m_p.to('g')).value, self.temp.value)
		self.HII = HII		

	def get_mu(self):
		mu = 4. / (3*ct.Xh + 1 + 4*self.HII*ct.Xh)
		self.mu = mu

	def get_dens_ion(self):
		self.dens_ion = (self.dens * self.HII * ct.Xh / ct.m_p)

    # We use case-B Hydrogen effective
    # recombination rate coefficient - Osterbrock & Ferland (2006)
    # effective recombination rate when temperature is accounted for
	def get_alphaH(self):
		self.alphaH = ct.alphaH.to('cm3/s')*(self.temp.to('K').value / 1.0e4)**-0.845







