# pylint: disable=C,W
import plotUtils as pu 
import numpy as np 
import astroUtils as au
import DataObj as do 
import statUtils as st
import mathUtils as mu
import time 
import scipy.stats as sp2
import sysUtils as su

simName = 'dataSim_super_long'

print(simName)
# Myr2ns = 1e6 * 3.154e7 * 1e9
pulsarInd = 1
sigma_dm = 200. * au.kms2kpcMyr
R_window = [0.9, 1.1]
rho0 = 0.01 * 1e9 # 0.01 Solar masses/pc^3 local dm density


def LoadPulsarData(d, pulsarInd):
	dataDir = d.dataDir + f"/Pulsars/pulsar{pulsarInd}/"
	static_redshift = np.load(dataDir + "static_redshift.npy")
	static_shapiro = np.load(dataDir + "static_shapiro.npy")
	r = np.load(dataDir + "r.npy")
	t = np.load(dataDir + "t.npy")

	return static_redshift, static_shapiro, r, t

def Get_PS_rho_predictionTotal_sh(d):
	hbar_ = d.hbar_[0]	
	lam_db = np.pi*hbar_ / (sigma_dm)
	kt_db = sigma_dm/lam_db

	N_ = d.N*32
	dx = d.L / N_
	kbins = np.arange(0, N_//2+1, .5)*(2.*np.pi/dx/N_)
	kvals = 0.5 * (kbins[1:] + kbins[:-1])
	kt = kvals*sigma_dm

	z = kt / kt_db * np.sqrt(2)

	PS = sp2.maxwell.pdf(z)

	return kt, PS


def Get_PS_rho_predictionTotal_z(d):
	hbar_ = d.hbar_[0]	
	lam_db = np.pi*hbar_ / (sigma_dm)
	kt_db = sigma_dm/lam_db

	N_ = d.N*32
	dx = d.L / N_
	kbins = np.arange(0, N_//2+1, .5)*(2.*np.pi/dx/N_)
	kvals = 0.5 * (kbins[1:] + kbins[:-1])
	kt = kvals*sigma_dm

	z = kt / kt_db * np.sqrt(2)

	PS = sp2.maxwell.pdf(z)

	return kt, PS


def MakePredictionPlot(d,fo):
	# fo = pu.FigObj()

	r_earth = np.array([0,0,0])
	r_E = LoadExtras(d)
	if not(r_E is None):
		r_earth = r_E

	static_redshift, static_shapiro, r, t = LoadPulsarData(d, 0)
	dt = np.max(t) / len(t)
	Tf = np.max(t)

	P_shapiro_avg = np.zeros(len(t))
	kt = np.fft.fftfreq(len(t),d = np.max(t) / len(t))

	count = 0

	for i in range(d.n_pulsars):
		static_redshift, static_shapiro, r, t = LoadPulsarData(d, i)

		r_ = r[0] - r_earth
		R = np.sqrt(np.sum(np.abs(r_)**2))

		if R < R_window[1] and R > R_window[0]:
			dt_sh = static_shapiro - np.mean(static_shapiro)
			# P_shapiro = np.abs(np.fft.fft(dt_sh))**2 * dt
			P_shapiro = np.abs(np.fft.fft(dt_sh))**2 / len(static_shapiro)
			kt = np.fft.fftfreq(len(t),d = np.max(t) / len(t))

			P_shapiro_avg += P_shapiro 

			count += 1

	P_shapiro_avg /= count


	lam = 2*np.pi*d.hbar_ / sigma_dm
	tau = lam / sigma_dm
	m_eff = rho0*lam**3
	N_enc_factor = (1./lam)**(2/3.)
	dt_avg = 1.5 * au.G * m_eff * N_enc_factor / au.speed_of_light**3
	kt_ = np.linspace(np.min(kt[kt>0]) / 10,np.max(kt[kt>0]) * 10, 256)
	omega, signal_predict = GetSignalPSPredict_shapiro(d,kt_)

	fo.AddPlot(kt*tau, P_shapiro_avg / dt_avg**2, ls= ' ', mk = '.', color = 'k', label = r'simulated data')
	# fo.AddLine(omega, signal_predict/dt_avg**2, color = 'r', label = r'$\propto f^{-2} e^{-\tau \, f / \sqrt{2}}$')

	kt_rho, PS_rho = Get_PS_rho_predictionTotal_sh(d)
	lam = 1. / (kt_rho / sigma_dm)
	N_enc_factor = (1./lam)**(2/3.)
	Amp =  lam**3 * N_enc_factor * rho0 * au.G / au.speed_of_light**3 / 1.9
	PS_prediction = PS_rho * Amp**2
	fo.AddLine(kt_rho*tau, PS_prediction/dt_avg[0]**2, color = 'r', label = r'$\propto f^{-8/3} e^{-\tau \, f / \sqrt{2}}$')

	fo.SetLogLog(kt[kt>0], P_shapiro_avg[kt>0] / dt_avg**2)
	fo.SetXLim(0.1, 2.)
	fo.SetYLim(1e-3,10)

	lam_ = 2*np.pi*d.hbar_/sigma_dm
	tau_ = lam_ / sigma_dm
	t1kpc = 1. / sigma_dm
	print(lam_)

	# fo.AddVertLine(1./Tf)
	# fo.AddVertLine(1./tau_)
	# fo.AddVertLine(1./t1kpc)

	fo.SetXLabel(r'$\tau f$')
	# fo.SetYLabel(r'$P^\mathrm{sh}(f) / \langle \tau { \Phi_\mathrm{rms}} \rangle^2$')
	fo.SetYLabel(r'$P^\mathrm{sh}(f) / \delta t_\mathrm{rms}^2$')
	fo.SetTitle(r"Shapiro delay")
	fo.legend()

	# fo.Save(d.dataDir + "prediction_vs_data_shapiro")
	# fo.show()

def GetSignalPSPredict(d,kt):
	# dt_z = (4e-4 * (d.m22/1e5)**(3/2))
	t_mode = 1./kt
	lam = 2*np.pi*d.hbar_ / sigma_dm
	m_eff = rho0*lam**3
	dt_z = t_mode * au.G*m_eff / lam / au.speed_of_light**2 / 2.5
	tau = lam / sigma_dm

	return kt, np.exp(-tau/t_mode / np.sqrt(2.) )*(dt_z)**2

def GetSignalPSPredict_shapiro(d,kt):
	# dt_z = (4e-4 * (d.m22/1e5)**(3/2))
	t_mode = 1./kt
	lam_mode = t_mode*sigma_dm
	lam = 2*np.pi*d.hbar_ / sigma_dm
	m_eff = rho0*lam_mode**3
	N_enc_factor = (1./lam_mode)**(2/3.)
	dt_z = 1.5 * au.G * m_eff * N_enc_factor / au.speed_of_light**3 / lam_mode
	tau = lam / sigma_dm

	return kt, np.exp(-tau/t_mode / np.sqrt(2.) )*(dt_z)**2

def MakePredictionPlot_z(d, fo):
	r_earth = np.array([0,0,0])
	r_E = LoadExtras(d)
	if not(r_E is None):
		r_earth = r_E


	static_redshift, static_shapiro, r, t = LoadPulsarData(d, 0)
	dt = np.max(t) / len(t)
	Tf = np.max(t)

	P_redshift_avg = np.zeros(len(t))
	kt = np.fft.fftfreq(len(t),d = np.max(t) / len(t))

	count = 0

	for i in range(d.n_pulsars):
		static_redshift, static_shapiro, r, t = LoadPulsarData(d, i)

		r_ = r[0] - r_earth
		R = np.sqrt(np.sum(np.abs(r_)**2))

		if R < R_window[1] and R > R_window[0]:
			delta_z = static_redshift - np.mean(static_redshift)
			# P_redshift = np.abs(np.fft.fft(static_redshift - np.mean(static_redshift)))**2 * dt
			P_redshift = np.abs(np.fft.fft(delta_z))**2 / len(static_redshift)
			kt = np.fft.fftfreq(len(t),d = np.max(t) / len(t))

			P_redshift_avg += P_redshift 

			count += 1

	P_redshift_avg /= count

	lam = 2*np.pi*d.hbar_ / sigma_dm
	tau = lam / sigma_dm
	m_eff = rho0*lam**3
	dt_z = tau* au.G*m_eff / lam / au.speed_of_light**2
	kt_ = np.linspace(np.min(kt[kt>0]) / 10,np.max(kt[kt>0]) * 10, 256)
	omega, signal_predict = GetSignalPSPredict(d,kt_)

	fo.AddPlot(kt, P_redshift_avg /dt_z**2, ls = '', mk = '.', color = 'k', label = r'simulated data')
	fo.AddLine(omega, signal_predict/dt_z**2, color = 'r', label = r'$\propto f^{-2} e^{-\tau \, f / \sqrt{2}}$')
	fo.SetLogLog(kt[kt>0], P_redshift_avg [kt>0])
	fo.SetXLim(0.1, 2)
	# fo.SetXLim(0.1,10)
	fo.SetYLim(1e-3,1e1)
	fo.SetXLabel(r'$\tau f$')
	fo.SetYLabel(r'$P^z(f) / \langle \tau { \Phi_\mathrm{rms}} \rangle^2$')
	fo.SetTitle(r'Redshift delay')
	fo.legend()
	# fo.Save("PS_redshift")
	# fo.legend()

	# fo.Save(d.dataDir + "prediction_vs_data_redshift")


def LoadExtras(d):
	d.LoadExtraParams(('r_earth'))
	r_earth = d.extraParams['r_earth']
	return r_earth


def Main(name):	
	d = do.MeshDataObj(name)
	LoadExtras(d)

	fo = pu.FigObj(2)
	MakePredictionPlot(d, fo)
	MakePredictionPlot_z(d, fo)

	fo.SetWhiteSpace(0.3,0)
	fo.Save(d.dataDir + "bothPS")
	fo.show()


if __name__ == "__main__":
	Main(simName)