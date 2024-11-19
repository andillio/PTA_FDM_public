# pylint: disable=C,W
# simulates pta fdm signals

import astroUtils as au
import numpy as np 
import time 
import Solvers.pulsarFDM_solver as ms
import Solvers.pulsarObjects as pulse
import gridUtils as gu
import sysUtils as su
import numpy.linalg as npl
import plotUtils as pu 
import scipy.stats as sp2

gpu = False 
if gpu:
	try:
		import cupy as np
	except ImportError:
		pass 

simName = "dataSim_center_1m22"
nf = 1
N  = 512
N_pulsars = 2048*2
data_drops = 100.
L = 5. # kpc box
dx = L / N
# Tf = 30e-6 # 30 yrs
m22 = np.array([1.000001])
C = au.G * 4. * np.pi
cf = 0.1
Mtot = 0.01 * 1e9 * L**3 # 0.01 Solar masses/pc^3 local dm density

sigma_dm = 200. * au.kms2kpcMyr
# print(sigma_dm**2 / au.speed_of_light**2)
n_streams = 2048

hbar_ = au.h_tilde(m22[0])
lam_db = hbar_ / (sigma_dm)
tau_db = lam_db / sigma_dm
Tf = tau_db * 20.

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


def SetFieldICs():
	X,Y,Z = gu.grid((N,N,N),L)
	psi = np.zeros((nf, N, N, N)) + 0j

	s = ms.Solver(simName= simName, N = N, data_drops=data_drops, padded = False,
		cf = 0.1, L = L, m22 = m22, Tf = Tf, gpu = gpu, C = C)
	s.D = 3
	s.hbar_ = au.h_tilde(m22)
	s.psi = psi
	s.dx = L/N
	s.dt_max = Tf / data_drops / 2.

	stream_velocities = np.zeros((nf, n_streams,3))
	time0 = time.time()
	np.random.seed(1)

	for j in range(nf):

		v_streams = randomInBall(n_streams)*sigma_dm
		stream_velocities[j,:] = v_streams

		for i in range(n_streams):
		
			v_ = v_streams[i]
			k_ = v_ / s.hbar_[j]
			S_ = np.rint(k_*L / 2 / np.pi)
			v_ = S_*2*np.pi * s.hbar_[j] / L
			v_mag = np.sqrt(np.sum(np.abs(v_)**2))
			w_ = 1.#np.exp(-.5*(v_mag/sigma_dm)**2)
			arg = -1j*(v_[0]*X + v_[1]*Y + v_[2]*Z) / s.hbar_[j]
			phi = np.random.uniform(0,2*np.pi)
			s.psi[j,:,:,:] += np.sqrt(w_)*np.exp(arg)*np.exp(1j*phi)
			done = i + j*len(v_streams) + 1

			su.PrintTimeUpdate(done, nf*n_streams, time0)

		s.psi[j,:,:,:] /= np.sqrt(np.sum(np.abs(s.psi[j,:,:,:])**2)*s.dx**3)
		s.psi[j,:,:,:] *= np.sqrt(Mtot)

	s.set_K()

	s.include_potential = False
	s.oldUpdateRule = True
	s.savePsi = False

	extras = {}
	extras['sigma_dm'] = sigma_dm
	s.extras = extras

	return s

def SetPulsarICs(s):
	r_earth = np.array([0,0,0])
	s.extras['r_earth'] = r_earth
	np.random.seed(1)

	for i in range(N_pulsars):
		pulsar_ = pulse.Pulsar()

		xPos = np.random.uniform(-L/2., L/2.)
		yPos = np.random.uniform(-L/2., L/2.)
		zPos = np.random.uniform(-L/2., L/2.)

		if i == 0:
			pulsar_.r = np.array([L/4., L/4., L/4.])
		else:
			pulsar_.r = np.array([xPos, yPos, zPos])
		pulsar_.v = np.array([0,0,0])

		pulsar_.simObj = s

		pulsar_.r_earth = r_earth

		pulsar_.name = "pulsar" + str(i)

		s.Pulsars.append(pulsar_)


def PlotStuff(d):
	fo = pu.FigObj()

	pulsar_ = d.Pulsars[0]
	print(pulsar_.r)
	print(pulsar_.r_earth)
	print(pulsar_.name)

	Phi = pulsar_.GetPotentialAlongPath()
	Phi_c = pulsar_.GetPhiOsc()
	pathlength = np.sqrt(np.sum(np.abs((pulsar_.r - pulsar_.r_earth))**2))
	affine = np.linspace(0,1,len(Phi_c)) * pathlength

	# fo.AddPlot(affine, Phi_c)
	# TODO: 
	# check summed potential so that we can check amplitude is logical

	# fo.AddLine(affine, (sigma_dm/au.speed_of_light)**2 + Phi)
	# fo.AddLine(affine, Phi + Phi_c, color = 'r')
	# fo.AddLine(affine, Phi, color = 'k')
	fo.AddLine(affine, Phi_c)
	print(np.sum(Phi_c))
	print(np.sum(Phi))
	# fo.AddPlot(np.abs(d.psi[0,0,:,:])**2)

	# print(np.sum(np.abs(d.psi)**2)*s.dx**3)

	fo.show()

if __name__ == "__main__":
	s = SetFieldICs()
	SetPulsarICs(s)
	# PlotStuff(s)
	s.RunSim()