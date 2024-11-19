# pylint: disable=C,W
import DataObj as do 
import astroUtils as au
import plotUtils as pu
import mathUtils as mu
import matplotlib.pyplot as plt 
import math
import statUtils as st
import scipy.stats as sp2
import numpy as np 

figName = "values"
simNames = [ 
"dataSim6a_vlong"
# ,"dataSim6a" 
# ,"dataSim6a_run2" 
# ,"dataSim6a_disk" 
,"dataSim6a_disk_long" 
# ,"dataSim6a_long"
,"dataSim6a_vlong_run2"
,"dataSim6c_long"
,"dataSim6c_run2"
,"dataSim6c_ball"
,"dataSim6a_ball_long"
,'dataSim6b'
,'dataSim6b_ball'
,"dataSimCornerd_long_run3"
# ,'dataSim6b_long'
,'dataSimCornerb_long_run3'
,'dataSimCornerf_long'
,'dataSim7_run1'
,'dataSimRand_run1'
,'dataSimRand_run2'
,'dataSimRand_run3'
,'dataSimRand_run5'
,'dataSimRand_run6'
,'dataSimCornere_long'
,'dataSimRand_run8'
,'dataSimRand_run9'
,'dataSimRand_run7'
,'dataSimRand_run10'
,'dataSimRand_run4'
,'dataSimRand_run11'
,'dataSimRand_run12'
,'dataSimRand_run13'
,'dataSimRand_run14'
,'dataSimRand_run15'
,'dataSimRand_run16'
# ,'dataSimRand_run17'
,'dataSimRand_run18'
,'dataSimRand_run19'
,'dataSimRand_run20'
,'dataSimRand_run21'
,'dataSimRand_run22'
,'dataSimRand_run23'
,'dataSimRand_run27'
,'dataSimRand_run28'
,'dataSimRand_run29'
,'dataSimRand_run30'
,'dataSimRand_run31'
,'dataSimRand_run32'
,'dataSim_super_long'
### new data
,'dataSim_1m22'
,'dataSim_05m22'
,'dataSim_2m22'
,'dataSim_4m22'
# ,'dataSim_8m22'
,'dataSim_center_1m22'
,'dataSim_center_2m22'
,'dataSim_center_2_5m22'
,'dataSim_center_05m22'
,'dataSim_center_02m22'
,'dataSim_corner_3m22'
,'dataSim_corner_4m22'
,'dataSim_corner_5m22'
,'dataSim_corner_6m22'
,'dataSim_corner_7m22'
,'dataSim_corner_8m22'
# ,'dataSim_corner_9m22'
# ,'dataSim_corner_10m22'
]

Myr2ns = 1e6 * 3.154e7 * 1e9
R_window = [0.9, 1.1]
sigma_dm = 200. * au.kms2kpcMyr
digits_result = 2
digits_error = 1
T_true_int = 30e-6
m22_thresh = .01
times = [0]
rho0 = 0.01 * 1e9 # 0.01 Solar masses/pc^3 local dm density

def LoadExtras(d):
	d.LoadExtraParams(('r_earth'))
	r_earth = d.extraParams['r_earth']
	return r_earth

def LoadPulsarData(d, pulsarInd, time = 0):
	dataDir = d.dataDir + f"/Pulsars/pulsar{pulsarInd}/"
	static_redshift = np.load(dataDir + "static_redshift.npy")
	static_shapiro = np.load(dataDir + "static_shapiro.npy")
	r = np.load(dataDir + "r.npy")
	t = np.load(dataDir + "t.npy")

	return static_redshift[time], static_shapiro[time], r, t

# def GetApprox(m22):
# 	prediction = np.zeros(m22.shape)
# 	v = np.linspace(.02, 2.4, 1024)*sigma_dm
# 	dv = (np.max(v) - np.min(v))/1024.

# 	P = 0
# 	for i in range(len(v)):	
# 		hbar_ = au.h_tilde(m22)
# 		v_ = v[i]
# 		lam = np.pi*hbar_ / v_ 
# 		D = 1.
# 		M_eff = rho0 * lam**3
# 		N_enc = D/lam
# 		P_ = sp2.maxwell.pdf(v_, scale = sigma_dm)*dv
# 		P += P_
# 		prediction += P_*2*au.G*M_eff / au.speed_of_light**3 * np.sqrt(N_enc) 
# 	return prediction * 1./P

def GetApprox(m22):
	hbar_ = au.h_tilde(m22)
	lam_d = hbar_ * 2*np.pi / sigma_dm
	m_eff = rho0*lam_d**3
	D = 1.
	delta_t = au.G*m_eff / au.speed_of_light**3 *(D/lam_d)**(2./3)
	return delta_t * 1.5
print( GetApprox( np.array([1e5]) ) * Myr2ns )

# def GetApproxRed(m22):
# 	prediction = np.zeros(m22.shape)
# 	v = np.linspace(.02, 2.4, 1024)*sigma_dm
# 	dv = (np.max(v) - np.min(v))/1024.

# 	P = 0
# 	for i in range(len(v)):	
# 		hbar_ = au.h_tilde(m22)
# 		v_ = v[i]
# 		lam = np.pi*hbar_ / v_
# 		M_eff = rho0 * lam**3
# 		P_ = sp2.maxwell.pdf(v_, scale = sigma_dm)*dv
# 		P += P_	
# 		prediction += P_*np.sqrt(2)*au.G*M_eff / lam / au.speed_of_light**2
# 	return prediction * 1./P

def GetApproxRed(m22):
	hbar_ = au.h_tilde(m22)
	lam_d = hbar_ * 2*np.pi / sigma_dm
	m_eff = rho0*lam_d**3
	phi_rms = au.G*m_eff / lam_d / au.speed_of_light**2
	return phi_rms / 1.5

def PlotStuff():

	shapiros = []
	redshifts = []
	masses = []

	T_max = 0

	for i in range(len(simNames)):
		name_ = simNames[i]
		d = do.MeshDataObj(name_)
		tau_c = d.hbar_ / sigma_dm**2
		print(name_, d.Tf / tau_c)

		_, _, _, t = LoadPulsarData(d, 0)
		if i == 0:
			T_max = np.max(t)
			# print(name_, np.max(t))
		else:
			T_max = np.min([T_max, np.max(t)])
			# print(name_, np.max(t))
	
	print(T_max)

	print("data to include\n")

	for time_ in times:
		for i in range(len(simNames)):
			name_ = simNames[i]
			d = do.MeshDataObj(name_)
			lam_d = d.hbar_[0] / sigma_dm
			if d.L > 2 and lam_d < 0.125: # orignal setting d.L > 2, lam_d < 0.1
				tau_c = d.hbar_ / sigma_dm**2
				print(name_, T_max / tau_c)


				r_earth = np.array([0,0,0])
				r_E = LoadExtras(d)
				if not(r_E is None):
					r_earth = r_E

				rms_sh_1kpc = []
				rms_gr_1kpc = []
				for j in range(d.n_pulsars):
					static_redshift, static_shapiro, r, t = LoadPulsarData(d, j, time_)

					r_ = r[0] - r_earth
					R = np.sqrt(np.sum(np.abs(r_)**2))

					sh_ = np.sqrt(static_shapiro**2)
					gr_ = np.sqrt(static_redshift**2)

					if R < R_window[1] and R > R_window[0]:
						rms_sh_1kpc.append(sh_)
						rms_gr_1kpc.append(gr_)

				sh_1kps = np.mean(rms_sh_1kpc)*Myr2ns
				gr_1kps = np.mean(rms_gr_1kpc)#*Myr2ns
				mass_ = d.m22[0]

				shapiros.append(sh_1kps)
				redshifts.append(gr_1kps)
				masses.append(mass_)

	shapiros = np.array(shapiros)
	redshifts = np.array(redshifts)
	masses = np.array(masses)
	prediction = GetApprox(masses)*Myr2ns

	fo = pu.FigObj(2)

	fo.AddPlot(masses, shapiros, ls = ' ', mk = 'o')
	fo.AddLine(masses, prediction, color = 'r', label = 'quasi-particles calc')

	log_dist = np.log10(masses[masses > m22_thresh])
	log_delay = np.log10(shapiros[masses > m22_thresh])
	xhat, yhat, mhat, bhat, m_se, b_se = st.fitLine(log_dist,log_delay, True)
	m_se_string = str(m_se)[2:4]
	fo.AddLine(10**xhat, 10**yhat, label = r'$\propto x^{%.2f (%s)}$'%(mhat, m_se_string) )
	fo.legend()
	print("shapiro slope:", mhat - m_se, mhat + m_se)

	fo.SetLogLog(masses, shapiros)
	fo.SetYLabel(r'$\delta t_\mathrm{rms} \, [\mathrm{ns}]$')
	fo.SetTitle(r'shapiros')
	fo.SetXLim(np.min(masses)/1.1 , np.max(masses)*1.1)
	fo.SetYLim(np.min(shapiros)/1.1 , np.max(shapiros)*1.1)
	fo.SetXLabel(r'$m_{22}$')

	masses = np.array(masses)
	prediction = GetApproxRed(masses)

	fo.AddPlot(masses, redshifts, ls = ' ', mk = 'o')
	fo.AddLine(masses, prediction, color = 'r', label = 'quasi-particles calc')
	
	log_dist = np.log10(masses[masses > m22_thresh])
	log_delay = np.log10(redshifts[masses > m22_thresh])
	xhat, yhat, mhat, bhat, m_se, b_se = st.fitLine(log_dist,log_delay, True)
	m_se_string = str(m_se)[2:4]
	fo.AddLine(10**xhat, 10**yhat, label = r'$\propto x^{%.2f (%s)}$'%(mhat, m_se_string) )
	fo.legend()
	print("redshift slope:", mhat - m_se, mhat + m_se)

	fo.SetXLabel(r'$m_{22}$')
	fo.SetYLabel(r'$\delta \Omega_\mathrm{rms} /\Omega_0 $')
	fo.SetTitle(r'redshifts')
	fo.SetLogLog(masses, redshifts)
	fo.SetXLim(np.min(masses)/1.1 , np.max(masses)*1.1)
	fo.SetYLim(np.min(redshifts)/1.1 , np.max(redshifts)*1.1)

	fo.Save(figName)
	fo.SavePng(figName)

	fo = pu.FigObj()

	fo.AddPlot(masses, shapiros, ls = ' ', mk = 'o', label = r'data')
	prediction = GetApprox(masses)*Myr2ns
	fo.AddLine(masses, prediction, color = 'r', label = r'quasi-particles calc')
	fo.SetXLim(np.min(masses)/1.1 , np.max(masses)*1.1)
	fo.SetYLim(np.min(shapiros)/1.1 , np.max(shapiros)*1.1)
	fo.SetLogLog(masses, shapiros)
	fo.SetXLabel(r'$m_{22}$')
	fo.SetYLabel(r'$\delta t_\mathrm{rms} \, [\mathrm{ns}]$')
	fo.legend()

	np.save("masses_shapiro.npy", masses)
	np.save("shapiros.npy", shapiros)
	np.save("prediction_shapiro.npy", prediction)

	fo.Save(figName + "shapiroOnly")
	fo.SavePng(figName+ "shapiroOnly")

	fo.show()


if __name__ == "__main__":
	PlotStuff()