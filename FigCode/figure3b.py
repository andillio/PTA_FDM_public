# pylint: disable=C,W
import DataObj as do 
import astroUtils as au
import plotUtils as pu
import mathUtils as mu
import matplotlib.pyplot as plt 
import math
import statUtils as st
import numpy as np 

figName = "Phi_rms"
simNames = ["dataSim6a_long"
,"dataSim6a_res1"
# ,"dataSim6a_disk"
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
,'dataSimCornere_long'
,'dataSimRand_run8'
,'dataSimRand_run9'
,'dataSimRand_run7'
,'dataSimRand_run10'
,'dataSimRand_run4'
,'dataSimRand_run11'
,'dataSimRand_run12'
,'dataSimRand_run13'
]

rho0 = 0.01 * 1e9
sigma_dm = 200. * au.kms2kpcMyr

def GetPrediction(d):
	lam_d = d.hbar_ * 2*np.pi / sigma_dm
	m_eff = rho0*lam_d**3
	phi_rms = au.G*m_eff / lam_d
	prediction = phi_rms / 1.5 / au.speed_of_light**2
	return prediction


def PlotStuff():

	fo = pu.FigObj()

	potential = []
	prediction = []
	masses = []

	for i in range(len(simNames)):
		name_ = simNames[i]
		print(name_)
		d = do.MeshDataObj(name_)
		d.LoadData(0)
		phi = d.compute_phi() / au.speed_of_light**2
		phi -= np.mean(phi)
		rms_ = mu.rms(phi)
		potential.append(rms_)
		masses.append(d.m22[0])
		prediction.append(GetPrediction(d))

	potential = np.array(potential)
	prediction = np.array(prediction)
	masses = np.array(masses)

	fo.AddPlot(masses, potential, ls = ' ', mk = 'o', label = r'data')

	log_dist = np.log10(masses)
	log_delay = np.log10(potential)
	xhat, yhat, mhat, bhat, m_se, b_se = st.fitLine(log_dist,log_delay, True)
	m_se_string = str(m_se)[3:5]
	fo.AddLine(10**xhat, 10**yhat, label = r'$\propto m^{%.3f (%s)}$'%(mhat, m_se_string) )
	print("shapiro slope:", mhat - m_se, mhat + m_se)

	fo.AddLine(masses, prediction, color = 'r', label = r'quasi particle approx.')
	fo.legend()

	fo.SetLogLog(masses, potential)
	fo.SetYLabel(r'$\delta \Phi_\mathrm{rms} / c^2$')
	fo.SetXLim(np.min(masses)/1.1 , np.max(masses)*1.1)
	fo.SetYLim(np.min(potential)/1.1 , np.max(potential)*1.1)
	fo.SetXLabel(r'$m_{22}$')

	np.save("masses_z.npy", masses)
	np.save("potentials.npy", potential)
	np.save("prediction_z.npy", prediction)

	fo.Save(figName)
	fo.show()

if __name__ == "__main__":
	PlotStuff()