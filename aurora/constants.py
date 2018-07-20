import math
import numpy as np
from astropy import units as unit
from astropy import constants as const

# Halpha wavelength in rest frame [cm]
Halpha0 = 6562.78e-8 * unit.cm
# NII wavelength in rest frame [cm]
NII0 = 6584.00e-8 * unit.cm
# Effective Halpha recombination rate
eff_recomb_rate_halpha = 1.16e-16 / unit.s
# Hydrogen cosmological fraction
Xh = 0.76
# Solar mass
M_sun = const.M_sun.to("g")
# Myear [s]
myr = 1.0E6 * 365.0 * 24.0 * 3600.0 * unit.s
# Solar O/H ratio
oh_solar = 8.66
# Solar metal fraction
z_solar = 0.0126
# Obscuration coefficient computed with RV = 3.1 (MW) - See Boselli (2012)
kMW_Ha = 2.517
# Hydrogen recombination rate
alphaH = 2.59e-13 * np.power(unit.cm, 3) / unit.s
# Physical constants defined in CGS system
h = const.h.to("erg s")
c = const.c.to("cm/s")
m_p = const.m_p.to("g")
k_B = const.k_B.to("erg/K")
kpc = const.kpc.to("cm")
G = const.G.to("cm3/(g s2)")
# Factor to go from FWHM to std dev
fwhm_sigma = (2.0 * math.sqrt(2.0 * math.log(2.0)))
## --------- ##

# FOR FUTURE REFERENCES / IMPLEMENTATIONS ??
# Obscuration coefficient for important emission lines
# Computed with RV = 3.1 (MW)
# See Boselli 2012
kMW_OII = 4.751
kMW_Hd = 4.418
kMW_Hg = 4.154
kMW_OIII_4363 = 4.128
kMW_Hb = 3.588
kMW_OIII_4959 = 3.497
kMW_OIII_5007 = 3.452
kMW_OI = 2.642
kMW_NII_6548 = 2.524
kMW_Ha = 2.517
kMW_NII_6584 = 2.507
kMW_SII_6716 = 2.444
kMW_SII_6731 = 2.437
kMW_Pb = 0.832
kMW_Pa = 0.451
