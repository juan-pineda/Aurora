import pynbody
import numpy as np
from scipy import special
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
		self.x = np.array(data_gas["x"].in_units("kpc"))*unit.kpc
		self.y = np.array(data_gas["y"].in_units("kpc"))*unit.kpc
		self.z = np.array(data_gas["z"].in_units("kpc"))*unit.kpc
		self.dens = np.array(data_gas["rho"].in_units("g cm**-3"))*unit.g/unit.cm**3
		self.vz = np.array(data_gas["vz"].in_units("cm s**-1"))*unit.cm/unit.s
		self.smooth = np.array(data_gas["smooth"].in_units("kpc"))*unit.kpc
		self.u = np.array(data_gas["u"].in_units("cm**2 s**-2"))*unit.cm**2/unit.s**2

	def density_cut(self, density_cut="Not"):
		if density_cut=="Not":
			pass
		elif density_cut=="polytrope":
			logdens = np.log10(self.dens.to("6.77e-23 g cm**-3").value)
			logtemp = np.log10(self.temp.to("K").value)
			tokill = ((logdens > logtemp - 3.5) & (logdens > 0.7) )
			self.dens = self.dens.to("g cm**-3").value
			self.dens[tokill] = np.min(self.dens) / 10
			self.dens = self.dens*unit.g/unit.cm**3
		else:
			thresh = 10**np.float(density_cut)
			tokill = (self.dens.to("6.77e-23 g cm**-3").value > thresh)
			self.dens = self.dens.to("g cm**-3").value
			self.dens[tokill] = np.min(self.dens) / 10
			self.dens = self.dens*unit.g/unit.cm**3


	# Derived physical quantities
	def get_state(self):
		self.get_temp()
		self.get_HII()
		self.get_mu()
		self.get_dens_ion()

	def get_luminosity(self, mode, density_cut="Not"):
		self.get_alphaH()
		if mode == "square":
			Halpha_lum = (self.smooth)**3 * (self.dens_ion)**2 * (ct.h*ct.c/ct.Halpha0) * self.alphaH 
		elif mode == "linear":
			Halpha_lum = (self.smooth)**3 * (self.dens_ion)**2 * (ct.h*ct.c/ct.Halpha0) * self.alphaH / (self.dens_ion.value)
		elif mode == "root":
			Halpha_lum = (self.smooth)**3 * (self.dens_ion)**2 * (ct.h*ct.c/ct.Halpha0) * self.alphaH / (self.dens_ion.value**1.5)
		self.Halpha_lum = Halpha_lum.to("erg s**-1")

		if density_cut=="Not":
			print("Nothing to cut")
		elif density_cut=="polytrope":
			print("polytropic cut to be implemented")
			logdens = np.log10(self.dens.to("6.77e-23 g cm**-3").value)
			logtemp = np.log10(self.temp.to("K").value)
			tokill = ((logdens > logtemp - 3.5) & (logdens > 0.7) )
			self.Halpha_lum[tokill] = 0 * unit.erg * unit.s**-1
		else:
			thresh = 10**np.float(density_cut)
			print("Cutting a threshold: ",thresh)
			tokill = (self.dens.to("6.77e-23 g cm**-3").value > thresh)
			self.Halpha_lum[tokill] = 0 * unit.erg * unit.s**-1

	def get_vel_dispersion(self):
		sigma = np.sqrt(ct.k_B * self.temp / (self.mu * ct.m_p))
		self.dispersion = sigma.to("cm s**-1")

	def get_temp(self):
		mu = np.ones(self.N)
		for i in range(5):
			temp = (5./3 - 1) * mu * ct.m_p * self.u / ct.k_B
			mu = self.get_mean_weight(temp)
		self.temp = temp.decompose().to("K")

	# Naive way of approximating the mean molecular weight
	# this is CUSTOMIZED for Mirage project, for which there is no
	# ElectroAbundance information stored !!!
	def get_mean_weight(self,temp):
	    mu = np.ones(len(temp))
	    mu[np.where(temp >= 1e4*unit.K)[0]] = 0.59
	    mu[np.where(temp < 1e4*unit.K)[0]] = 1.3
	    return mu

	def get_HII(self):
		if "HII" in self.data.keys():
			HII = self.data["HII"]
		else:
			a = bird.GasProperties(self.redshift)
			HII = 1 - a._neutral_fraction(ct.Xh * (self.dens.to("g cm**-3")/ct.m_p.to("g")).value, self.temp.value)
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
		self.alphaH = ct.alphaH.to("cm3/s")*(self.temp.to("K").value / 1.0e4)**-0.845

    # Retain only line centers/broadenings for particles in this group,
    # arranged in a matrix where each row is a particle, and columns
    # will serve to store fluxes at each of the cube spectral channels, e.g,
    # with n particles centered at l1, l2 ..., ln, line_center is:
    # [ l1 l1 l1 ... l1
    #   l2 l2 l2 ... l2
    #   .  .  .  ...
    #   .  .  .  ...
    #   ln ln ln ... ln]
	def get_vect_lines(self, n_ch):
		line_center = np.transpose(np.tile(self.vz, (n_ch, 1)))
		line_sigma = np.transpose(np.tile(self.dispersion, (n_ch, 1)))
		line_flux = np.transpose(np.tile(self.Halpha_lum, (n_ch, 1)))
		return line_center, line_sigma, line_flux

	def get_vect_channels(self, channels, width, n_ch):
		channel_center = np.tile(channels, (self.N, 1))
		channel_width = np.tile(width, (self.N, n_ch))
		return channel_center, channel_width


	def int_gaussian(self, x, dx, mu, sigma):
	    """
	    Compute the integral of a normalized gaussian inside some limits.
	    The center and width
	
	    Parameters
	    ----------
	    x : float, array
	        central position of the interval.
	    dx : float, array
        width of the interval.
	    mu: float, array
	        mean of the gaussian.
	    sigma: float, array
	        standard deviation.
	    """

	    A = special.erf((x+dx/2-mu)/np.sqrt(2)/sigma)
	    B = special.erf((x-dx/2-mu)/np.sqrt(2)/sigma)
	    return np.abs((A-B)/2)

	# Notice that, because *int_gaussian* integrates a normalized gaussian,
	# the units of the input parameters do not affect the result as far as all
	# of them are the same
	def int_gaussian_with_units(self, x, dx, mu, sigma):
	    dx = dx.to(x.unit)
	    mu = mu.to(x.unit)
	    sigma = sigma.to(x.unit)
	    inte = self.int_gaussian(x.value, dx.value, mu.value, sigma.value)
	    return inte


