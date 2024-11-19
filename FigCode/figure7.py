# pylint: disable=C,W
import plotUtils as pu 
import numpy as np 
import matplotlib.pyplot as plt
import astroUtils as au

Myr2ns = 1e6 * 3.154e7 * 1e9
sigma_dm = 200. * au.kms2kpcMyr
beta = sigma_dm / au.speed_of_light

fo = pu.FigObj(2)
f = np.logspace(-9,-6,100)
dummy = np.ones(len(f))*1e-10

### compton stuff
m_c = f/(5e-9)*1e-23
omega_c = f / (2*np.pi)
Psi_c = 2e-15 / 2 / np.sqrt(3) / (m_c/1e-23)**2
dt_c = np.sqrt(2)*Psi_c/omega_c * 1e9
print(np.interp(1e-23, m_c, dt_c))
dt_max = np.max(dt_c)
dT_max = np.ones(2)*1e-16 / (3e-19 / 2 / np.pi) / np.sqrt(3) / np.sqrt(100)

ax, im = fo.AddPlot(f, dt_c, color = 'b', label = r'$\delta t^C$')

### deBroglie stuff
m_db = np.logspace(-18,-14,100)
m22 = m_db / 1e-22
# m17 = m22*1e5
hbar_ = au.h_tilde(m22)
lam = 2*np.pi *hbar_ / sigma_dm
tau_db = lam / sigma_dm *1e6 * 3.154e7
T = 30 * 3.154e7
f_db = 1./tau_db 
omega_db = f_db / (2*np.pi)
dt_db = 4e-4 * (m22/1e5)**(-3) * (T/tau_db)**(1.5) #/ omega_db 
dt_sh = 5e-2 * (m22/1e5)**(-7/3.)
dt_dop = 7e-2 * (m22/1e5)**(-3/2.)
fo.AddLine(f_db, dt_dop, color = 'b', ls = '-.', label = r'$\delta t^d$')
fo.AddLine(f_db, dt_sh, color = 'r', ls = '-.', label = r'$\delta t^{sh}$')
fo.AddLine(f_db, dt_db, color = 'r', ls = '-', label = r'$\delta t^z$')

f_min = 1./(30*3.15e7)

xs = [f_min, np.max(f_db)]
ys = [dt_max, dt_max]
ax.fill_between(xs, ys, dT_max, interpolate=True, color='g', alpha = 0.2)#, label = r"100 pulsars, 50 ns")

fo.SetLogLog(f_db, [np.min(dt_sh), np.max(dt_c)])
fo.SetYLim(1e-7,1e5)

fo.SetXLabel(r'$\mathrm{f \, [Hz]}$')
fo.SetYLabel(r'$\delta t_\mathrm{rms} \, [\mathrm{ns}]$')
fo.legend()
fo.SetWhiteSpace(.3,0)

# fo.save("rubikov_comparison")
### make rho constraint plot
fo2 = pu.FigObj()

dT_max = np.ones(len(dt_sh))*1e-16 / (3e-19 / 2 / np.pi) / np.sqrt(3) / np.sqrt(100)
rho_curve = dT_max/dt_sh
fo2.AddLine(m22, rho_curve)
fo2.SetLogLog(m22, rho_curve)
fo2.AddVertLine(1e5)

### deBroglie stuff
f = np.logspace(-10,0,100)
### compton stuff
m_c = f/(5e-9)*1e-23
omega_c = f / (2*np.pi)
Psi_c = 2e-15 / 2 / np.sqrt(3) / (m_c/1e-23)**2
dt_c = np.sqrt(2)*Psi_c/omega_c * 1e9

m_db = np.logspace(-24,-14,100)
m22 = m_db / 1e-22
hbar_ = au.h_tilde(m22)
lam = 2*np.pi *hbar_ / sigma_dm
tau_db = lam / sigma_dm *1e6 * 3.154e7
f_db = 1./tau_db 
dt_db = 4e-4 * (m22/1e5)**(-3) * (T/tau_db)**(1.5) #/ omega_db 
dt_sh = 5e-2 * (m22/1e5)**(-7/3.)
dt_dop = 7e-2 * (m22/1e5)**(-3/2.)

ax, im = fo.AddPlot(m_c, dt_c, color = 'b', label = r'Compton')
fo.AddLine(m_db, dt_dop, color = 'b', ls = '-.', label = r'deBroglie, shapiro')
fo.AddLine(m_db, dt_sh, color = 'r', ls = '-.', label = r'deBroglie, shapiro')
fo.AddLine(m_db, dt_db, color = 'r', ls = '-', label = r'deBroglie, redshift')
fo.SetLogLog([1e-24,1e-14], [np.min(dt_sh), np.max(dt_db)])

fo.SetXLabel(r'$\mathrm{m \, [\mathrm{eV/c^2}]}$')
fo.SetYLabel(r'$\delta t_\mathrm{rms} \, [\mathrm{ns}]$')
# fo.legend()

fo.save("db_c_comparison")



fo.show()

