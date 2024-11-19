# pylint: disable=C,W
import plotUtils as pu 
import numpy as np 
import astroUtils as au
import DataObj as do 
import statUtils as st
import mathUtils as mu
import os
import time 
import sysUtils as su

figName = "Phi_rms"

simNames = ["dataSim6a_long"
,"dataSim6a_res1"
,"dataSim_super_long"
,"dataSim6a_disk_long"
,"dataSim6a_res2"
,"dataSim6a_run2"
,"dataSim6a_vlong"
,"dataSim6a_vlong_run2"
,"dataSim6c_long"
,"dataSim6a_ball_long"
,"dataSim6b_ball"
,"dataSimCornerb_long_run3"
,"dataSimCornerd_long_run3"
,"dataSimCornere_long"
,'dataSimCornerf_long'
,'dataSim7_run1'
,'dataSimRand_run1'
,'dataSimRand_run2'
,'dataSimRand_run3'
,'dataSimRand_run5'
,'dataSimRand_run6'
,'dataSimRand_run8'
,'dataSimRand_run9'
,'dataSimRand_run7' # questionable
,'dataSimRand_run10'
,'dataSimRand_run4'
,'dataSimRand_run11'
,'dataSimRand_run12'
,'dataSimRand_run13'
,'dataSimRand_run16'
,'dataSimRand_run20'
,'dataSimRand_run21'
,'dataSimRand_run22'
,'dataSimRand_run23'
,'dataSimRand_run25'
,'dataSimRand_run26'
,'dataSimRand_run27'
,'dataSimRand_run28'
,'dataSimRand_run29'
,'dataSimRand_run30'
,'dataSimRand_run31'
,'dataSimRand_run32'
]

Myr2ns = 1e6 * 3.154e7 * 1e9
R_window = [0.9, 1.1]
tau_min = 5
tau_max = 15
dropMax = 0
sigma_dm = 200. * au.kms2kpcMyr
overwrite = False
rho0 = 0.01 * 1e9 # 0.01 Solar masses/pc^3 local dm density


def LoadPulsarData(d, pulsarInd,drop_max):
	dataDir = d.dataDir + f"/Pulsars/pulsar{pulsarInd}/"
	static_redshift = np.load(dataDir + "static_redshift.npy")
	static_shapiro = np.load(dataDir + "static_shapiro.npy")
	r = np.load(dataDir + "r.npy")
	t = np.load(dataDir + "t.npy")

	# static_redshift -= np.mean(static_redshift)
	# static_shapiro -= np.mean(static_shapiro)

	if drop_max > 0:
		static_redshift = static_redshift[0:drop_max+1]
		static_shapiro = static_shapiro[0:drop_max+1]
		r = r[0:drop_max+1]
		t = t[0:drop_max+1]

	# print(np.max(t) - np.min(t))
	dt = (np.max(t) - np.min(t)) / len(t)
	static_redshift -= np.mean(static_redshift)
	static_redshift = mu.cumInt(static_redshift, t, dt)
	# static_redshift -= np.mean(static_redshift)

	static_shapiro -= np.mean(static_shapiro)

	return static_redshift, static_shapiro, r, t


def LoadExtras(d):
	d.LoadExtraParams(('r_earth'))
	r_earth = d.extraParams['r_earth']
	return r_earth


def AnalyzeData(d):
	# check if data already exists or if I should overwrite it
	path1 = d.dataDir + "std_value_shapiro.npy"
	path2 = d.dataDir + "std_value_redshift.npy"
	# if it does then return else
	if os.path.isfile(path1) and os.path.isfile(path2):
		return
	# produce data analysis

	dropMax = d.data_drops+1

	distance_from_start_shapiro = np.zeros(dropMax-1)
	distance_from_start_redshift = np.zeros(dropMax-1)
	std_value_shapiro = np.zeros(dropMax-1)
	std_value_redshift = np.zeros(dropMax-1)
	mean_value_shapiro = np.zeros(dropMax - 1)
	mean_value_redshift = np.zeros(dropMax - 1)
	T = np.zeros(dropMax-1)

	L = d.L 
	r_earth = np.array([0,0,0])
	r_E = LoadExtras(d)
	if not(r_E is None):
		r_earth = r_E

	time0 = time.time()
	for j in range(3,dropMax):

		dfssh = []
		dfsrs = []
		svsh = []
		svrs = []
		mvsh = []
		mvrs = []


		for i in range(d.n_pulsars):
			pulsarInd = i
			static_redshift, static_shapiro, r, t = LoadPulsarData(d, pulsarInd,j)

			T[j-1] = np.max(t)

			r_ = r[0] - r_earth
			R = np.sqrt(np.sum(np.abs(r_)**2))

			if R < R_window[1] and R > R_window[0]:
				dfssh.append(static_shapiro - static_shapiro[0])
				dfsrs.append(static_redshift - static_redshift[0])
				svsh.append(np.std(static_shapiro))
				svrs.append(np.std(static_redshift))
				mvsh.append(np.mean(static_shapiro))
				mvrs.append(np.mean(static_redshift))

		distance_from_start_shapiro[j-1] = np.mean(dfssh)
		distance_from_start_redshift[j-1] = np.mean(dfsrs)
		std_value_shapiro[j-1] = np.mean(svsh)
		std_value_redshift[j-1] = np.mean(svrs)
		mean_value_shapiro[j-1] = np.mean(mvsh)
		mean_value_redshift[j-1] = np.mean(mvrs)

		su.PrintTimeUpdate(j + 1, dropMax, time0)

	su.PrintCompletedTime(time0)
	np.save(d.dataDir + "t.npy", T)
	np.save(d.dataDir + "distance_from_start_shapiro.npy", distance_from_start_shapiro)
	np.save(d.dataDir + "distance_from_start_redshift.npy", distance_from_start_redshift)
	np.save(d.dataDir + "std_value_shapiro.npy", std_value_shapiro)
	np.save(d.dataDir + "std_value_redshift.npy", std_value_redshift)
	np.save(d.dataDir + "mean_value_shapiro.npy", mean_value_shapiro)
	np.save(d.dataDir + "mean_value_redshift.npy", mean_value_redshift)	


def AnalyzeAllData():
	# loop through all the data and produce the integrated std
	numSims = len(simNames)
	for i in range(numSims):
		name = simNames[i]
		d = do.MeshDataObj(name)
		AnalyzeData(d)


def GetApprox(m22):
	hbar_ = au.h_tilde(m22)
	lam_d = hbar_ * 2*np.pi / sigma_dm
	m_eff = rho0*lam_d**3
	D = 1.
	delta_t = au.G*m_eff / au.speed_of_light**3 *(D/lam_d)**(2./3)
	return delta_t * 1.5
def GetApproxRed(m22):
	hbar_ = au.h_tilde(m22)
	lam_d = hbar_ * 2*np.pi / sigma_dm
	tau_d = lam_d / sigma_dm
	m_eff = rho0*lam_d**3
	phi_rms = au.G*m_eff / lam_d / au.speed_of_light**2
	return phi_rms / 1.5 * tau_d / (2*np.pi) / 4
	# m17 = m22 / 1e5
	# print(m22, m17)
	# return 1e-3 * m17**(-3) / Myr2ns
print( GetApproxRed(np.array([1e5])) * Myr2ns  )
print(1./( au.h_tilde(np.array([1e-1])) *2 *np.pi / au.speed_of_light**2 * Myr2ns / 1e9 ) )

def ScaleAndConcatenateData(d, t_matrix,
 shapiro_data_matrix, redshift_data_matrix):
	t = np.load(d.dataDir + "t.npy")[1:]
	std_value_shapiro = np.load(d.dataDir + "std_value_shapiro.npy")[1:]
	std_value_redshift = np.load(d.dataDir + "std_value_redshift.npy")[1:]

	tau = 2*np.pi * d.hbar_ / sigma_dm**2
	t_scaled = t / tau
	t_matrix = np.concatenate((t_matrix, t_scaled))

	shapiro_scale = GetApprox(d.m22)[0] 
	# print(d.m22, shapiro_scale)
	redshift_scale = GetApproxRed(d.m22)[0]
	std_value_shapiro /= shapiro_scale
	std_value_redshift /= redshift_scale

	shapiro_data_matrix = \
		np.concatenate((shapiro_data_matrix, std_value_shapiro))  

	redshift_data_matrix = \
		np.concatenate((redshift_data_matrix, std_value_redshift)) 

	return t_matrix, shapiro_data_matrix, redshift_data_matrix



def PlotStuff():
	# plot
	### - loop through data and create one large data matrix
	### - plot that
	### - fit best fit line to that
	t_matrix = np.zeros(0)
	shapiro_data_matrix = np.zeros(0)
	redshift_data_matrix = np.zeros(0)

	numSims = len(simNames)
	for i in range(numSims):
		name = simNames[i]
		d = do.MeshDataObj(name)
		r_E = LoadExtras(d)

		t_matrix, shapiro_data_matrix, redshift_data_matrix = \
			ScaleAndConcatenateData(
				d, t_matrix, shapiro_data_matrix, redshift_data_matrix)

	fo = pu.FigObj(2)
	t_max_sh = 1
	t_max_z = 12.5
	fo.AddPlot(t_matrix, shapiro_data_matrix, ls = '', mk = '.',color = 'k', alpha = 0.1)
	fo.AddLine([0],[0], ls = '', mk = '.',color = 'k', alpha = 0.5, label = r'simulated data')
	shapiro_data_matrix = shapiro_data_matrix[t_matrix > 0]
	redshift_data_matrix = redshift_data_matrix[t_matrix > 0]
	t_matrix = t_matrix[t_matrix>0]
	logt = np.log10(t_matrix[t_matrix<t_max_sh])
	logsh = np.log10(shapiro_data_matrix[t_matrix<t_max_sh])
	xhat, yhat, mhat, bhat, m_se, b_se = st.fitLine(logt,logsh, True)
	print(mhat)
	m_se_string = str(m_se)[2:4]
	# fo.AddLine(10**xhat, 10**yhat, label = r'$\propto x^{%.2f (%s)}$'%(mhat, m_se_string) )
	fo.AddHorLine(1,ls= '-', color = 'r', label = r'prediction')
	fo.SetYLabel(r'$\delta t_\mathrm{rms}^\mathrm{sh} / \delta \hat t_\mathrm{rms}^\mathrm{sh}$')
	fo.SetXLabel(r'$T/\tau$')
	fo.SetXLim(0,12.5)
	fo.SetYLim(0,2)
	fo.SetTitle(r'Shapiro delay')
	fo.legend()

	fo.AddPlot(t_matrix, redshift_data_matrix, ls = '', mk = '.', color = 'k', alpha = 0.1)
	logt = np.log10(t_matrix[t_matrix<t_max_z])
	logz = np.log10(redshift_data_matrix[t_matrix<t_max_z])
	xhat, yhat, mhat, bhat, m_se, b_se = st.fitLine(logt,logz, True)
	print(mhat)
	m_se_string = str(m_se)[2:4]
	# fo.AddLine(10**xhat, 10**yhat, label = r'$\propto x^{%.2f (%s)}$'%(mhat, m_se_string) )
	tau = np.linspace(0,20,128)
	fo.AddLine(tau, tau**1.5, color = 'r')
	fo.SetXLim(0,12.5)
	fo.SetYLim(0,50)
	# fo.legend()
	fo.SetYLabel(r'$\delta t_\mathrm{rms}^\mathrm{z} / \delta \hat t_\mathrm{rms}^\mathrm{z}$')
	fo.SetXLabel(r'$T/\tau$')
	fo.SetTitle(r'Redshift delay')
	
	fo.save("timeScaling")

	fo.show()

def Main():
	AnalyzeAllData()
	PlotStuff()


if __name__ == "__main__":
	Main()