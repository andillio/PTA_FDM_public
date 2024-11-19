# pylint: disable=C,W
import numpy as np_ 
import astroUtils as au
from mathUtils import interpAtPos3D
import numpy as np_
CUPY_IMPORTED = True
import os
import warnings as warn 
try:
	import cupy as cp 
except ImportError:
	CUPY_IMPORTED = False


class Pulsar(object):

	def __init__(self):
		# sim stuff
		self.r = None # position of pulsar 
		self.v = None # velocity of pulsar

		self.simObj = None # pointer to main solver object
		self.r_earth = None # position of earth

		self.N_needed_cap = 0

		self.gpu = False

		# data stuff
		self.name = "" # file name of this pulsar object
		self.data_dir = ""
		self.static_shapiro = []
		self.osc_shapiro = []
		self.static_redshift = []
		self.osc_redshift = []
		self.r_over_time = []
		self.v_over_time = []
		self.time = []

	def InitializeFiles(self, phi):
		"""
		makes the data directory, outputs the toml file and initial data drop
		"""
		if self.name == None:
			raise Exception("simName has not been set.\n"+\
				"set simName before initializing files.")

		if not(os.path.isdir(f"Data/{self.simObj.simName}/Pulsars")):
			os.mkdir(f"Data/{self.simObj.simName}/Pulsars")

		if not(os.path.isdir(f"Data/{self.simObj.simName}/Pulsars/{self.name}")):
			os.mkdir(f"Data/{self.simObj.simName}/Pulsars/{self.name}")

		self.data_dir = f"Data/{self.simObj.simName}/Pulsars/{self.name}"

		self.OutputICs(phi)

	def OutputICs(self, phi):
		"""
		outputs the initial conditions
		"""
		self.DataDrop(0, phi)


	def Update(self, dt):
		self.Drift(dt)


	def Drift(self, dt):
		"""
		update the dynamic variable positions

		:dt: float, timestep
		"""
		self.r += self.v * dt

	def GetPotentialAlongPath(self, Phi):
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		# get potential along path to earth
		lam_c = self.simObj.hbar_[0] / au.speed_of_light
		pathlength = np.sqrt(np.sum(np.abs((self.r - self.r_earth))**2))
		dx = self.simObj.dx

		N_needed = pathlength / lam_c	
		if (self.N_needed_cap > 0):
			N_geusses = np.zeros(2)
			N_geusses[0] = N_needed
			N_geusses[1] = self.N_needed_cap
			N_needed = np.min(N_geusses)

		N_path = self.simObj.N * 2
		if N_needed > self.simObj.N:
			N_path = int(N_needed * 4)

		if len(Phi) == 0:
			Phi = self.simObj.compute_phi() / au.speed_of_light**2
		Phi -= np.mean(Phi)
		# rms = np.sqrt(np.mean(np.abs(Phi)**2))
		# print(rms)
		vec2Earth = np.zeros((N_path,self.simObj.D))
		for j in range(N_path):
			vec2Earth[j] = self.r - (self.r - self.r_earth) * float(j) / (N_path-1)
		Phi_ = interpAtPos3D(vec2Earth, self.simObj.dx, Phi, leftEdge= -1*self.simObj.L/2., gpu = self.gpu)	

		return Phi_

	def GetPhiOsc(self):
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		lam_c = self.simObj.hbar_[0] / au.speed_of_light
		pathlength = np.sqrt(np.sum(np.abs((self.r - self.r_earth))**2))
		dx = self.simObj.dx

		N_needed = pathlength / lam_c	
		if (self.N_needed_cap > 0):
			N_geusses = np.zeros(2)
			N_geusses[0] = N_needed
			N_geusses[1] = self.N_needed_cap
			N_needed = np.min(N_geusses)

		N_path = self.simObj.N * 2
		if N_needed > self.simObj.N:
			N_path = int(N_needed * 4)

		affine = np.linspace(0,1,N_path) * pathlength # kpc
		vec2Earth = np.zeros((N_path,self.simObj.D))
		for j in range(N_path):
			vec2Earth[j] = self.r - (self.r - self.r_earth) * float(j) / (N_path-1)

		psi_real = interpAtPos3D(vec2Earth, dx, np.real(self.simObj.psi[0])
			, leftEdge= -1*self.simObj.L/2., gpu = self.gpu)
		psi_imag = interpAtPos3D(vec2Earth, dx, np.imag(self.simObj.psi[0])
			, leftEdge= -1*self.simObj.L/2., gpu = self.gpu)
		psi_ = psi_real + psi_imag*1j

		rho_ = np.abs(psi_)**2
		alpha_ = np.angle(psi_)
		t_ = affine / au.speed_of_light + self.simObj.T_sim
		omega = 2 * au.speed_of_light**2 / self.simObj.hbar_

		amp = np.pi*au.G * self.simObj.hbar_[0]**2 * rho_ / au.speed_of_light**2

		arg = t_*omega + 2*alpha_

		Phi_c = amp*np.cos(arg) / au.speed_of_light**2
		# rms = np.sqrt(np.mean(np.abs(Phi_c)**2))
		# print(rms)
		return Phi_c



	def DataDrop(self, drop, phi = []):
		"""
		record the pulsars position, velocity, time
		shapiro time delays and gravitational redshifts
		from oscillating and static component
		"""
		t_ = (self.simObj.Tf*drop) / self.simObj.data_drops
		self.time.append(t_)

		self.r_over_time.append(self.r)
		self.v_over_time.append(self.v)

		Phi_ = self.GetPotentialAlongPath(phi)
		Phi_c = self.GetPhiOsc()

		# static shapiro
		shapiro_static = self.compute_shapiro_time_delay(Phi_)
		self.static_shapiro.append(shapiro_static)
		# osc shapiro
		shapiro_osc = self.compute_shapiro_time_delay(Phi_c)
		self.osc_shapiro.append(shapiro_osc)
		# static redshift
		redshift_static = self.compute_grav_redshift(Phi_)
		self.static_redshift.append(redshift_static)
		# osc redshift
		redshift_osc = self.compute_grav_redshift(Phi_c)
		self.osc_redshift.append(redshift_osc)

		if drop == self.simObj.data_drops // 2 or drop == 0 or drop == self.simObj.data_drops // 3 or drop == self.simObj.data_drops:
			self.DataOutput()
		

	def DataOutput(self):
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		# output all the data files
		static_shapiro = np.array(self.static_shapiro)
		np.save(self.data_dir + "/static_shapiro.npy", static_shapiro)

		osc_shapiro = np.array(self.osc_shapiro)
		np.save(self.data_dir + "/osc_shapiro.npy", osc_shapiro)

		static_redshift = np.array(self.static_redshift)
		np.save(self.data_dir + "/static_redshift.npy", static_redshift)

		osc_redshift = np.array(self.osc_redshift)
		np.save(self.data_dir + "/osc_redshift.npy", osc_redshift)

		r_over_time = np.array(self.r_over_time)
		np.save(self.data_dir + "/r.npy", r_over_time)

		v_over_time = np.array(self.v_over_time)
		np.save(self.data_dir + "/v.npy", v_over_time)

		time = np.array(self.time)
		np.save(self.data_dir + "/t.npy", time)


	def compute_shapiro_time_delay(self, Phi_):
		"""
		computes and returns the shapiro time delay due to the granular 
		overdensity due to FDM
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		pathlength = np.sqrt(np.sum(np.abs((self.r - self.r_earth))**2))

		shapiro_FDM = (pathlength / len(Phi_) / au.speed_of_light)*np.sum(2*Phi_)

		return shapiro_FDM

	def compute_grav_redshift(self, Phi_):
		"""
		computes and returns the gravitational redshift between the first
		and last entries of Phi
		"""
		return Phi_[0] - Phi_[-1]

