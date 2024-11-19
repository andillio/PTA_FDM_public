# pylint: disable=C,W
# finds distance scaling
from math import dist
import astroUtils as au
import numpy as np_ 
from mathUtils import interpAtPos3D
import numpy as np 
import time 
import gridUtils as gu
import mathUtils as mu 
import sysUtils as su
import numpy.linalg as npl
import plotUtils as pu 
import scipy.stats as sp2
seed_ = int(time.time())
gpu = False
if gpu:
	try:
		import cupy as np
	except ImportError:
		pass 

figName = "distanceScalingPlot_randE_alt"
nf = 1
N  = 256
N_pulsars = int(1e5)
# Tf = 30e-6 # 30 yrs
# m22_ = 2.5 # run1
m22_ = 7.5  # run2
m22 = np_.array([m22_])
hbar_ = au.h_tilde(m22)
if gpu:
	m22 = np.asarray(m22) 
L = 2.2 # kpc box
dx = L / N
C = au.G * 4. * np.pi
rho0 = 0.01 * 1e9
Mtot = rho0 * L**3 # 0.01 Solar masses/pc^3 local dm density
Myr2ns = 1e6 * 3.154e7 * 1e9

sigma_dm = 200. * au.kms2kpcMyr
lam_ = np.pi * hbar_[0] / sigma_dm
print(lam_, dx)
# print(sigma_dm**2 / au.speed_of_light**2)
n_streams = 1024
N_path = 1024
tag = "big"

# returns a random variable in a ball
def randomInBall(Npoints):  
	"""
	returns random variables in a 3 ball

	:Npoints: int, number of points to return
	"""  
	x = np.random.normal(0,1,Npoints)
	y = np.random.normal(0,1,Npoints)
	z = np.random.normal(0,1,Npoints)

	points = np.zeros((Npoints, 3))
	points[:,0] = x
	points[:,1] = y    
	points[:,2] = z

	norm  = 1./npl.norm(points, axis = 1)
	points = np.einsum("ij,i->ij",points, norm )
	#mag = np.random.exponential(size = Npoints)
	mag = sp2.maxwell.rvs(size = Npoints )
	points = np.einsum("ij,i->ij",points, mag )

	return points


# returns a random variable in a ball
def RandomOnSphere(Npoints, R, noise = 0):  
    """
    returns random variables in a 3 ball

    :Npoints: int, number of points to return
    """  
    x = np.random.normal(0,1,Npoints)
    y = np.random.normal(0,1,Npoints)
    z = np.random.normal(0,1,Npoints)

    points = np.zeros((Npoints, 3))
    points[:,0] = x
    points[:,1] = y    
    points[:,2] = z

    norm  = 1./npl.norm(points, axis = 1)
    points = np.einsum("ij,i->ij",points, norm )
    # mag = np.random.exponential(size = Npoints)
    # mag = sp2.maxwell.rvs(size = Npoints )

    mag1 = np.random.uniform(0,R, Npoints)
    mag2 = np.random.uniform(0,R, Npoints)
    mag3 = np.random.uniform(0,R, Npoints)
    mag = np.maximum(mag1, mag2)
    mag = np.maximum(mag,mag3)
    # mag = np.random.normal(R,noise, Npoints)

    points = np.einsum("ij,i->ij",points, mag )

    return points

def SetFieldICs():
	X,Y,Z = gu.grid((N,N,N),L,gpu=gpu)
	psi = np.zeros((nf, N, N, N)) + 0j

	stream_velocities = np.zeros((nf, n_streams,3))
	time0 = time.time()
	np.random.seed(seed_)

	for j in range(nf):

		v_streams = randomInBall(n_streams)*sigma_dm
		stream_velocities[j,:] = v_streams

		for i in range(n_streams):
		
			v_ = v_streams[i]
			k_ = v_ / hbar_[j]
			S_ = np.rint(k_*L / 2 / np.pi)
			v_ = S_*2*np.pi * hbar_[j] / L
			v_mag = np.sqrt(np.sum(np.abs(v_)**2))
			w_ = 1.#np.exp(-.5*(v_mag/sigma_dm)**2)
			arg = -1j*(v_[0]*X + v_[1]*Y + v_[2]*Z) / hbar_[j]
			phi = np.random.uniform(0,2*np.pi)
			psi[j,:,:,:] += np.sqrt(w_)*np.exp(arg)*np.exp(1j*phi)
			done = i + j*len(v_streams) + 1

			su.PrintTimeUpdate(done, nf*n_streams, time0)

		psi[j,:,:,:] /= np.sqrt(np.sum(np.abs(psi[j,:,:,:])**2)*dx**3)
		psi[j,:,:,:] *= np.sqrt(Mtot)

	rho = np.sum(np.abs(psi)**2, axis = 0)

	return rho


def SetPulsarICs(r_earth):
	r = RandomOnSphere(N_pulsars, 1.)
	rval = np.zeros((N_pulsars, 3))

	for i in range(N_pulsars):
		r_ = np.zeros(3)
		r_[0] = np.abs(r[i,0]) + r_earth[0]
		r_[1] = np.abs(r[i,1]) + r_earth[1]
		r_[2] = np.abs(r[i,2]) + r_earth[2]
		rval[i] = r_

	return rval


def Analyze(rho):
	Phi = au.compute_phi(rho, L)
	Phi -= np.mean(Phi)
	Phi /= au.speed_of_light**2

	distances = np.zeros(N_pulsars)
	shapiro_delays = np.zeros(N_pulsars)
	redshifts = np.zeros(N_pulsars)

	time0 = time.time()

	i = 0
	while i < N_pulsars:
		r_pos = np.random.uniform(-L/2., L/2., 3) # pulsar position
		earth_pos = np.random.uniform(-L/2., L/2., 3) # earth position
		pathlength = np.sqrt(np.sum(np.abs((r_pos - earth_pos))**2))

		if pathlength < L/2.:
			distances[i] = pathlength

			vec2Center = np.zeros((N_path,3))
			affine = np.linspace(0,1.0,N_path)*pathlength
			for j in range(N_path):
				vec2Center[j] = r_pos - (r_pos -earth_pos) * float(j) / (N_path-1)

			Phi_ = interpAtPos3D(vec2Center, dx, Phi, leftEdge= -L/2.)
			shapiro_delays[i] = (pathlength / len(affine) / au.speed_of_light)*np.sum(2*Phi_)
			redshifts[i] = Phi_[0] - Phi_[-1]

			su.PrintTimeUpdate(i+1,N_pulsars, time0)
			i += 1


	np.save(f"../Data/distancesToPulsars{tag}.npy", distances)
	np.save(f"../Data/shapiro_delaysToPulsars{tag}.npy", shapiro_delays)
	np.save(f"../Data/redshiftsToPulsars{tag}.npy", redshifts)

def GetAvg(a_initial, a_final):
	bins = 100
	a_max = L/2.
	da = a_max / bins

	a_bins = da*(np.arange(bins) + .5)
	a_counts = np.zeros(bins)
	a_final_avg = np.zeros(bins)

	for i in range(len(a_initial)):
		a_ = a_initial[i]
		af_ = a_final[i]

		ind_ = int(a_ / da)
		if a_ > 1e-10 and ind_ < bins:
			a_final_avg[ind_] += af_ 
			a_counts[ind_] += 1

	a_final_avg = a_final_avg / a_counts
	mu.RemoveNans(a_final_avg)

	return a_bins, a_final_avg


def GetApprox(D):
	lam_d = hbar_ * 2*np.pi / sigma_dm
	m_eff = rho0*lam_d**3
	delta_t = au.G*m_eff / au.speed_of_light**3 * (D/lam_d)**(2/3.)
	# delta_t = au.G*m_eff / au.speed_of_light**3 * (D/lam_d)**.5
	return delta_t*1.5

def GetApproxRed():
	lam_d = hbar_ * 2*np.pi / sigma_dm
	m_eff = rho0*lam_d**3
	phi_rms = au.G*m_eff / lam_d / au.speed_of_light**2
	return phi_rms / 1.5


def PlotDistance():
	distances = np.load(f"../Data/distancesToPulsars{tag}.npy")
	shapiro_delays = np.abs(np.load(f"../Data/shapiro_delaysToPulsars{tag}.npy"))
	redshifts = np.abs(np.load(f"../Data/redshiftsToPulsars{tag}.npy"))

	fo = pu.FigObj(2)

	x, dt_avg = GetAvg(distances, shapiro_delays)
	print(np.mean(shapiro_delays))
	dt_est = GetApprox(x)
	x, z_avg = GetAvg(distances, redshifts)
	z_est = GetApproxRed()

	fo.AddLine(distances[::10], shapiro_delays[::10] * Myr2ns, ls = '', mk = '.'
		,color = 'k', alpha = 0.1)
	fo.AddLine([0],[0],color = 'k', ls = '', mk = '.', label = r"simulated data")
	fo.AddLine(x, dt_avg * Myr2ns, label = r"data average")
	fo.AddLine(x, dt_est * Myr2ns, color = 'r', label = r'prediction')
	fo.SetYLim(0,8e8)
	fo.SetXLim(0.01,1.1)
	fo.AddVertLine(2*lam_, label = r'$2\pi \hbar/m\sigma$')
	# fo.SetLogLog(distances, shapiro_delays * Myr2ns)
	fo.SetXLabel(r'$x \, \mathrm{[kpc]}$')
	fo.SetYLabel(r'$|\delta t^\mathrm{sh}| \, [\mathrm{ns}]$')
	fo.SetTitle(r'Shapiro delay')
	fo.legend()

	# fo.save(figName + "shapiro")
	# fo.SavePng(figName+ "shapiro")

	# fo = pu.FigObj()

	fo.AddPlot(distances[::10], redshifts[::10], ls = '', mk = '.', color = 'k', alpha = 0.1)
	fo.AddLine([0],[0],color = 'C0', ls = '', mk = '.', label = r"simulated data")
	fo.AddLine(x, z_avg, label = r"data average")
	fo.AddHorLine(z_est, label = r'quasi-particle approx.', color = 'r', ls = '-')
	# fo.SetLogLog(distances, redshifts)
	fo.SetYLim(0,1e-11)
	fo.SetXLim(0.01,1.1)
	fo.AddVertLine(lam_, label = r'$\hbar/m\sigma$')
	fo.SetXLabel(r'$x \, \mathrm{[kpc]}$')
	fo.SetYLabel(r'$|z|$')
	fo.SetTitle(r'Redshift')
	# fo.legend()

	# fo.save(figName + "redshift")
	# fo.SavePng(figName+ "redshift")

	fo.save(figName)

	fo.show()



def PlotDensity(rho):
	fo = pu.FigObj()

	rho_proj = np.sum(rho, axis = 0)
	x = [L/-2., L/2.]

	fo.AddDens2d(x, rho_proj)
	fo.SetXLabel(r'$x \, \mathrm{[kpc]}$')
	fo.SetYLabel(r'$y \, \mathrm{[kpc]}$')

	fo.show()

def Main():
	# rho = SetFieldICs()
	# PlotDensity(rho)
	# Analyze(rho)
	PlotDistance()


if __name__ == "__main__":
	Main()