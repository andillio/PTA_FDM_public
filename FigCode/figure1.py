# pylint: disable=C,W
import numpy as np 
import astroUtils as au
from mathUtils import interpAtPos3D
import time 
import gridUtils as gu
import meshSolver as ms
import numpy.linalg as npl
import statUtils as st
import sysUtils as su
import plotUtils as pu 
import scipy.stats as sp2
gpu = False 
if gpu:
	try:
		import cupy as np
	except ImportError:
		pass 

simName = "1m22"
nf = 1
N  = 512
N_path = 2024
sample_points = 2048*4
data_drops = 10.
L = 1.0 # kpc box
Tf = 1.0 # 30e-6 # 30 yrs
m22 = np.array([1.0])
C = au.G * 4. * np.pi
cf = 0.1
Myr2s = 3.14e7 * 1e6
Mtot = 0.01 * 1e9 * L**3 # 0.01 Solar masses/pc^3 local dm density

c = au.speed_of_light
sigma_dm = 200. * au.kms2kpcMyr
n_streams = 512 * 2

hbar_ = au.h_tilde(m22[0])
kmax = np.pi * N / 2.
dk = 2 * kmax / N
dt_kinetic = 2. * np.pi / kmax / dk / hbar_
dt_kinetic_old =  4 *np.pi / kmax**2 / hbar_

lam_db = hbar_ / (sigma_dm)
tau_db = lam_db / sigma_dm
tau_obs = 30e-6
tau_propogate = 3e-3
tau_c = tau_db*1e-6
omega_c = c**2 / hbar_
omega_d = sigma_dm**2 / hbar_
tau_c = 1. / omega_c


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



def SetICs():
	X,Y,Z = gu.grid((N,N,N),L)
	psi = np.zeros((nf, N, N, N)) + 0j

	s = ms.Solver(simName= simName, N = N, data_drops=data_drops, padded = False,
		cf = 0.1, L = L, m22 = m22, Tf = Tf, gpu = gpu)
	s.D = 3
	s.hbar_ = au.h_tilde(m22)
	s.psi = psi
	s.dx = L/N
	s.mp = 0
	s.C = au.G * 4* np.pi

	stream_velocities = np.zeros((nf, n_streams,3))
	time0 = time.time()

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
			psi_ = np.sqrt(w_)*np.exp(arg)*np.exp(1j*phi)
			psi_ /= np.sqrt(np.sum(np.abs(psi_)**2)*s.dx**3)
			s.psi[j,:,:,:] += psi_
			done = i + j*len(v_streams) + 1

			su.PrintTimeUpdate(done, nf*n_streams, time0)

		s.psi[j,:,:,:] /= np.sqrt(np.sum(np.abs(s.psi[j,:,:,:])**2)*s.dx**3)
		s.psi[j,:,:,:] *= np.sqrt(Mtot)
 
	s.set_K()
	s.R,_,_ = gu.sphrGrid(N,L,gpu = gpu)
	s.oldUpdateRule = True
	return s


def PlotDensity(d):
	fo = pu.FigObj(2)

	rho = np.sum(np.abs(d.psi)**2, axis = 0)
	rho_proj = np.sum(rho, axis = 2)*d.dx
	rho_slice = rho[d.N//2, :, :]
	x = [L/-2., L/2.]

	fo.AddDens2d(x,rho_proj)	
	fo.SetXLabel(r'$x \,[\mathrm{kpc}]$')
	fo.SetYLabel(r'$y \,[\mathrm{kpc}]$')
	fo.SetTitle(r'Density')
	# fo.RemoveXLabels()

	# fo.save(simName + "density")
	# fo.show()
	return fo

def PlotLineInt(s,fo):
	Phi = s.compute_phi(False, False) 
	Phi -= np.mean(Phi)
	Phi /= au.speed_of_light**2

	r_pos = np.array([L/2.5, L/2.5, L/2.5]) # pulsar position
	xPos = r_pos[0]
	yPos = r_pos[1]
	earth_pos = np.array([L/-2.5, L/-2.5, L/-2.5]) # earth position
	xEarth = earth_pos[0]
	yEarth = earth_pos[1]

	fo.AddLine([xPos, xEarth], [yPos, yEarth], mk = 'o', color = 'r')
	fo.SetXLim(-L/2,L/2)
	fo.SetYLim(-L/2,L/2)

	pathlength = np.sqrt(np.sum(np.abs((r_pos - earth_pos))**2))
	vec2Center = np.zeros((N_path,3))
	affine = np.linspace(0,1.0,N_path)*pathlength
	for j in range(N_path):
		vec2Center[j] = r_pos - (r_pos -earth_pos) * float(j) / (N_path-1)

	Phi_ = interpAtPos3D(vec2Center, s.dx, Phi, leftEdge= -L/2.)
	fo.AddPlot(affine, Phi_, color = 'r')
	fo.SetXLabel(r'$r \,[\mathrm{kpc}]$')
	fo.SetYLabel(r'$\Phi (r) / c^2$')
	fo.SetTitle(r'Potential')


if __name__ == "__main__":
	s = SetICs()
	fo = PlotDensity(s)
	PlotLineInt(s,fo)
	fo.SetWhiteSpace(0.25,0)
	fo.save("simExamplePlot")

	fo.show()