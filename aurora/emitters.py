import pynbody
import numpy as np
from astropy import units as unit

from . import constants as ct
from . import gasProps_sBird as bird

class Emitters:

	def __init__(self, data_gas,redshift=None):
		self.N = len(data_gas)
		self.redshift = redshift
		self.x = np.array(data_gas['x'].in_units('kpc'))*unit.kpc
		self.y = np.array(data_gas['y'].in_units('kpc'))*unit.kpc
		self.z = np.array(data_gas['z'].in_units('kpc'))*unit.kpc
		self.dens = np.array(data_gas['rho'].in_units('g cm**-3'))*unit.g/unit.cm**3
		self.vz = np.array(data_gas['vz'].in_units('cm s**-1'))*unit.cm/unit.s
		self.smooth = np.array(data_gas['smooth'].in_units('kpc'))*unit.kpc
		self.u = np.array(data_gas['u'].in_units('cm**2 s**-2'))*unit.cm**2/unit.s**2
		self.temp = self.get_temp().decompose()
		self.HII = self.get_HII(data_gas)
		self.mu = self.get_mu()	
	
	def get_temp(self):
		mu = np.ones(self.N)
		for i in range(5):
			temp = (5./3 - 1) * mu * ct.m_p * self.u / ct.k_B
			mu = self.get_mean_weight(temp)
		return temp

	def get_mean_weight(self,temp):
	    mu = np.ones(len(temp))
	    mu[np.where(temp >= 1e4*unit.K)[0]] = 0.59
	    mu[np.where(temp < 1e4*unit.K)[0]] = 1.3
	    return mu

	def get_HII(self,data_gas):
		if 'HII' in data_gas.keys():
			HII = data_gas['HII']
		else:
			HII = np.zeros(self.N)
			a = bird.GasProperties(self.redshift)
			HII = 1 - a._neutral_fraction(ct.Xh * (self.dens.to('g cm**-3')/ct.m_p.to('g')).value, self.temp.value)
		return HII

	def get_mu(self):
		mu = 4. / (3*ct.Xh + 1 + 4*self.HII*ct.Xh)
		return mu

