# pylint: disable=C,W
import plotUtils as pu 
import numpy as np 
import matplotlib.pyplot as plt
import astroUtils as au

Myr2ns = 1e6 * 3.154e7 * 1e9
sigma_dm = 200. * au.kms2kpcMyr

m_db = np.logspace(-17,-14,100)
m22 = m_db / 1e-22
hbar_ = au.h_tilde(m22)
lam = 2*np.pi *hbar_ / sigma_dm
tau_db = lam / sigma_dm *1e6 * 3.154e7
T = 30 * 3.154e7
f_db = 1./tau_db 
dt_db = 4e-4 * (m22/1e5)**(-3) * (T/tau_db)**(1.5) #/ omega_db 
dt_sh = 5e-2 * (m22/1e5)**(-7/3.)


fo = pu.FigObj()

dT_max = np.ones(len(dt_sh))*1e-16 / (3e-19 / 2 / np.pi) / np.sqrt(3) / np.sqrt(100)
rho_curve_sh = dT_max/dt_sh
ax,im = fo.AddPlot(m_db, rho_curve_sh, color = 'r', alpha = 0)
fo.SetLogLog(m_db, rho_curve_sh)
fo.AddVertLine(1e5)

x = np.logspace(-17,-14,100)
y1 = 1e9*np.ones(len(x))
y2 = rho_curve_sh
ax.fill_between(x, y1, y2, interpolate=True, color='r', alpha = 1.0, label = r"Shapiro")

x = np.logspace(-17,-14,100)
y1 = 1e9*np.ones(len(x))
y2 = dT_max / dt_db
ax.fill_between(x, y1, y2, interpolate=True, color='b', alpha = 1.0, label = r"Redshift")

fo.SetXLim(1e-18,1e-15)
fo.SetYLim(.1,1e8)
fo.AddHorLine(1)
fo.SetXLabel(r'$\mathrm{m \, [\mathrm{eV/c^2}]}$')
fo.SetYLabel(r'$\rho / \rho_0$')
# fo.SetTitle()
fo.legend(loc = 'center right')

fo.save("density_contour")

fo.show()