# pylint: disable=C,W
from math import tau
import plotUtils as pu 
import numpy as np 
import astroUtils as au
import DataObj as do 
import mathUtils as mu

simName = "dataSimCornerb_long_run3"
simName = "dataSim6a_vlong_run2"
pulsarInd = 0
sigma_dm = 200. * au.kms2kpcMyr
Myr2ns = 1e6 * 3.154e7 * 1e9


def LoadPulsarData(d, pulsarInd):
	dataDir = d.dataDir + f"/Pulsars/pulsar{pulsarInd}/"
	static_redshift = np.load(dataDir + "static_redshift.npy")
	static_shapiro = np.load(dataDir + "static_shapiro.npy")
	r = np.load(dataDir + "r.npy")
	t = np.load(dataDir + "t.npy")

	return static_redshift, static_shapiro, r, t


def MakeFig(d):
	hbar_ = d.hbar_[0]
	lam_db = hbar_ / (sigma_dm)
	tau_db = lam_db / sigma_dm

	# t = np.linspace(0,d.Tf,d.data_drops + 2) / tau_db
	dt = d.Tf / (d.data_drops - 1)

	fo = pu.FigObj(3)

	static_redshift, static_shapiro, r, t = LoadPulsarData(d, pulsarInd)
	t /= tau_db

	deltaOmega_d = static_redshift - np.mean(static_redshift)
	delta_t_gr_d = np.cumsum(deltaOmega_d*dt)* Myr2ns
	delta_t_gr_d -= np.mean(delta_t_gr_d)

	delta_t_sh_d = (static_shapiro - np.mean(static_shapiro)) * Myr2ns


	ax, im = fo.AddLine(t, delta_t_sh_d, color = 'r', label = r'debroglie')
	# fo.SetYLim(1e-26*Myr2ns, 1e-11*Myr2ns)
	fo.SetXLabel(r'$t/\tau_d$')
	fo.SetYLabel(r'$|\delta t| \, [\mathrm{ns}]$')
	fo.SetTitle(r'shapiro time delays')
	fo.legend()

	# print(np.mean(deltaOmega))
	# ax, im = fo.AddPlot(t, deltaOmega_d, color= 'r', label = r'debroglie' )
	ax, im = fo.AddPlot(t, deltaOmega_d, color= 'r', label = r'debroglie' )
	# fo.SetYLim(1e-19, 1e-9)
	# ax.set_yscale("log")
	fo.SetXLabel(r'$t/\tau_d$')
	# fo.SetYLabel(r'$|\delta t| \, [\mathrm{ns}]$')
	fo.SetYLabel(r'$|\Delta \Omega / \Omega_0|$')
	fo.SetTitle(r'grav redshifts')

	ax, im = fo.AddPlot(t, delta_t_gr_d, color= 'r', label = r'debroglie' )
	# fo.SetYLim(1e-19, 1e-9)
	# ax.set_yscale("log")
	fo.SetXLabel(r'$t/\tau_d$')
	fo.SetYLabel(r'$|\delta t| \, [\mathrm{ns}]$')
	# fo.SetYLabel(r'$|\Delta \Omega / \Omega_0|$')
	fo.SetTitle(r'grav redshifts')

	fo.SetWhiteSpace(0.3,0)
	fo.save(d.dataDir + "debroglieResults" + str(pulsarInd))
	fo.show()



def MakeFig2(d):
	hbar_ = d.hbar_[0]
	lam_db = hbar_ / (sigma_dm)
	tau_db = lam_db / sigma_dm

	# t = np.linspace(0,d.Tf,d.data_drops + 2) / tau_db
	dt = d.Tf / (d.data_drops - 1)

	fo = pu.FigObj(2)

	static_redshift, static_shapiro, r, t = LoadPulsarData(d, pulsarInd)

	deltaOmega_d = static_redshift - np.mean(static_redshift)
	delta_t_gr_d = np.cumsum(deltaOmega_d*dt)* Myr2ns
	delta_t_gr_d -= np.mean(delta_t_gr_d)

	delta_t_sh_d = (static_shapiro - np.mean(static_shapiro)) * Myr2ns

	ax, im = fo.AddLine(t, delta_t_sh_d, color = 'r', label = r'debroglie')
	# fo.SetYLim(1e-26*Myr2ns, 1e-11*Myr2ns)
	fo.SetXLabel(r'$t \, [\mathrm{Myr}]$')
	fo.SetYLabel(r'$\delta t^{\mathrm{sh}} (t) \, [\mathrm{ns}]$')
	fo.SetTitle(r'Shapiro delay')

	# print(np.mean(deltaOmega))
	# ax, im = fo.AddPlot(t, deltaOmega_d, color= 'r', label = r'debroglie' )
	ax, im = fo.AddPlot(t, deltaOmega_d, color= 'r', label = r'debroglie' )
	# fo.SetYLim(1e-19, 1e-9)
	# ax.set_yscale("log")
	fo.SetXLabel(r'$t \, [\mathrm{Myr}]$')
	# fo.SetYLabel(r'$|\delta t| \, [\mathrm{ns}]$')
	fo.SetYLabel(r'$z(t)$')
	fo.SetTitle(r'Redshift')

	fo.SetWhiteSpace(0.2,0)
	fo.save(d.dataDir + "debroglieResults" + str(pulsarInd))
	fo.show()



def Main(name):	
	d = do.MeshDataObj(name)
	MakeFig(d)
	MakeFig2(d)

if __name__ == "__main__":
	Main(simName)