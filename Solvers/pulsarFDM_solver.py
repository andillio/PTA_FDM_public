# pylint: disable=C,W
import numpy as np_
import os
import time
import sysUtils as su
import mathUtils as mu
import astroUtils as au
import plotUtils as pu
import baseSolver as BS
CUPY_IMPORTED = True
import warnings as warn 
try:
	import cupy as cp 
except ImportError:
	CUPY_IMPORTED = False

class Solver(BS.Solver):

	def __init__(self, simName = None, N = None, np = None, data_drops = None, padded = False,
		cf = None, L = None, dx = None, nf = None, m22 = [], hbar_ = [], 
		D = None, C = None, Tf = None, r = [], v = [], dt = None, gpu = False, 
		psi = [], K = [], mp = None):
		"""
		initialize solver object

		:simName: string, simulation data directory name
		:N: int, sim resolution
		:np: int, number of corpuscular particles
		:mp: float, mass of corpuscular particles
		:data_drops: int, number of data outputs
		:padded: bool, pad the density
		:cf: float, courant factor 
		:L: float, box length
		:dx: float, pixel size
		:nf: int, number of fields
		:m22: array-like, [nf] dark matter mass [1e-22 eV/C^2]
		:hbar_: array-like, [nf] hbar/m
		:D: int, number of spatial dimensions
		:C: float, Poisson's constant
		:Tf: float, final sim time
		:r: array-like, [np, D] particle positions
		:v: array-like, [np, D] particle velocities
		:dt: float, initial timestep
		:gpu: bool, run on the gpu
		:psi: array-like, [nf,N^D] spatial field
		:K: array-like, [N^3] kinetic update operator argument, \n
			i.e. e^{-1j*hbar_*K*dt} is the op

		:return: solver object
		"""
		super().__init__(simName = simName, N = N, np = np, data_drops = data_drops,
		padded = padded, cf = cf, L = L, dx = dx, nf = nf, D = D, C = C, Tf = Tf,
		r = r, v = v, dt = dt, mp = mp)

		### simulation parameters
		self.oldUpdateRule = True
		self.include_potential = False
		self.savePsi = True

		### physics parameters
		self.m22 = m22 # float, particle mass [1e-22 eV/c^2]
		self.hbar_ = hbar_ # float, hbar / m [kpc^2 / Myr] 

		### dynamical paramters
		self.psi = psi # array-like, [nf,N^D] spatial field
		self.K = K # array-like, [N^3] kinetic update operator argument, kx x kx x kx
		self.R = [] # array-like, [N^3] radial position each cell is defined at
		self.Pulsars = [] # array-like, pulsars in simulation
		self.n_pulsars = 0
		self.T_sim = 0.

		### diagnostic params
		self.rho_alias = 0.

	def SetParams(self, simName = None, N = None, np = None, mp = None, data_drops = None, padded = None,
		cf = None, L = None, dx = None, nf = None, m22 = [], hbar_ = [], 
		D = None, C = None, Tf = None, r = [], v = [], dt = None, gpu = None, 
		psi = [], K = [], n_pulsars = None):
		"""
		sets parameters

		:simName: string, simulation data directory name
		:N: int, sim resolution
		:np: int, number of corpusular particles
		:mp: float, mass of corpuscular particles
		:data_drops: int, number of data outputs
		:padded: bool, pad the density
		:cf: float, courant factor 
		:L: float, box length
		:dx: float, pixel size
		:nf: int, number of fields
		:m22: array-like, [nf] dark matter mass [1e-22 eV/C^2]
		:hbar_: array-like, [nf] hbar/m
		:D: int, number of spatial dimensions
		:C: float, Poisson's constant
		:Tf: float, final sim time
		:r: array-like, [np, D] particle positions
		:v: array-like, [np, D] particle velocities
		:dt: float, initial timestep
		:gpu: bool, run on the gpu
		:psi: array-like, [nf,N^D] spatial field
		:K: array-like, [N^3] kinetic update operator argument, \n
			i.e. e^{-1j*hbar_*K*dt} is the op
		"""
		super().SetParams(simName = simName, N = N, np = np, data_drops = data_drops,
		padded = padded, cf = cf, L = L, dx = dx, nf = nf, mp = mp,
		D = D, C = C, Tf = Tf, r = r, v = v, dt = dt, gpu = gpu)

		if N != None:
			self.set_N(N)
		if L != None:
			self.set_L(L)
		if len(hbar_) > 0:
			self.set_hbar_(hbar_)
		if len(m22) > 0:
			self.set_m22(m22)
		if len(hbar_) > 0:
			self.set_hbar_(hbar_)
		if len(psi) > 0:
			self.set_psi(psi)
		if len(K) > 0:
			self.set_K(K)

		if N != None:
			self.set_N_perif()
		if L != None:
			self.set_L_perif()
		if len(psi) > 0:
			self.set_psi_perif()
		if not(n_pulsars is None):
			self.set_n_pulsars(n_pulsars)

	def set_n_pulsars(self, n_pulsars):
		"""
		set the number of pulsars in the simulation

		:n_pulsars: int, number of pulsars
		"""
		self.n_pulsars = n_pulsars


	def set_N(self, N):
		"""
		set the grid resolution

		:N: int, grid resolution
		"""
		super().set_N(N)

	def set_N_perif(self):
		"""
		private function, initializes attributes that depend on N
		"""
		super().set_N_perif()
		self.set_K()

	def set_L(self, L):
		"""
		set the box size

		:L: float, box size
		"""
		super().set_L(L)

	def set_L_perif(self):
		"""
		private function, initializes attributes that depend on L
		"""
		super().set_L_perif()
		self.set_K()

	def set_psi(self, psi):
		"""
		set the field

		:psi: array-like, [nf,N^D] spatial field
		"""
		self.psi = psi
		self.set_psi_perif()

	def set_psi_perif(self):
		"""
		private function, sets attributes that depend on psi
		"""
		self.D = len(self.psi.shape) - 1
		self.nf = len(self.psi)


	def set_K(self, K = []):
		"""
		set the spectral gird

		:K: array-like, [N^3] kinetic update operator argument, \n
			i.e. e^{-1j*hbar_*K*dt} is the op \n
			default: use N,L, and D to figure K out
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		if len(K) > 0:
			self.K = K 
		else:
			if self.N != None and self.L != None and self.D != None:
				dx = self.L/self.N
				kx = 2*np.pi*np.fft.fftfreq(self.N,d = dx)
				ones = np.ones(self.N)

				self.K = self.get_K(self.N, self.L)


	def set_m22(self, m22):
		"""
		sets the mass of the field [1e-22 eV/C^2]

		:m22: array-like, [nf] dark matter mass [1e-22 eV/C^2]
		"""
		self.m22 = m22
		self.set_m22_perif()

	def set_hbar_(self, hbar_):
		"""
		sets hbar_ of the field 

		:hbar_: array-like, [nf] hbar/m
		"""		
		self.hbar_ = hbar_ 

	def set_m22_perif(self):
		"""
		private function, set attributes that depend on m22
		"""
		self.hbar_ = .01959 / self.m22
		self.nf = len(self.m22)

	def set_hbar_perif(self):
		"""
		private function, set attributes that depend on m22
		"""
		self.m22 = .01959 / self.hbar_
		self.nf = len(self.hbar_)

	def set_nf_perif(self):
		"""
		private function, initializes attributes that depend on nf
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		if len(self.m22) == 0:
			self.m22 = np.zeros(self.nf)
		if len(self.hbar_) == 0:
			self.hbar_ = np.zeros(self.nf)

	def set_psi_from_file(self, fileName):
		"""
		loads a file and sets it to be the field

		:fileName: ICs file name
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		self.set_psi(np.load("ICs/" + fileName))


	def InitializeFiles(self):
		"""
		makes the data directory, outputs the toml file and initial data drop
		"""
		if self.simName == None:
			raise Exception("simName has not been set.\n"+\
				"set simName before initializing files.")

		if not(os.path.isdir(f"Data/{self.simName}")):
			os.mkdir(f"Data/{self.simName}")
		self.OutputToml() # toml file
		self.OutputICs()

		Phi = self.compute_phi() / au.speed_of_light**2
		time0 = time.time()
		print("initializing pulsar files...\n")
		for i in range(len(self.Pulsars)):
			pulsar_ = self.Pulsars[i]
			pulsar_.InitializeFiles(Phi)
			su.PrintTimeUpdate(i+1, len(self.Pulsars), time0)


	def OutputICs(self):
		"""
		outputs the initial conditions
		"""
		if not(os.path.isdir(f"Data/{self.simName}/psi")):
			os.mkdir(f"Data/{self.simName}/psi")
		self.DataDrop(0)


	def OutputToml(self):
		"""
		outputs toml with simulation parameters
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp


		padded_ = "true" if self.padded else "false"
		text = f'''
		# all units in kpc, Msolar, Myr
		[physics]
		Tfinal                      = {self.Tf + self.T_initial} # float, final sim time
		L                           = {self.L} # float, box length
		C 							= {self.C} # float, poisson's constant
		D 							= {self.D} # int, number of spatial dimensions
		m22 						= {np.array2string(self.m22, separator=', ')} # hbar / m_field

		[simulation]
		N                           = {self.N} # int, grid size
		n_pulsars                 = {len(self.Pulsars)} # int, number of simulation particles
		drops                       = {self.data_drops+ self.initial_drop} # int, number of data drops
		c_f                         = {self.cf} # float, timestep courant factor
		'''

		f = open(f"Data/{self.simName}/meta.toml", "w")
		f.write(text)
		f.close()

		if len(self.extras) > 0:
			extras = {}
			extras['extras'] = self.extras
			print(extras)
			su.AddLines2Toml(extras, self.GetTomlString())


	def DataDrop(self, i):
		"""
		output the current status of the dynamical variables
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		super().DataDrop(i)
		if self.savePsi or i == 0 or i == self.data_drops or i == self.data_drops // 2 or i == self.data_drops // 3:
			np.save("Data/" + self.simName + f"/psi/drop{i+ self.initial_drop}.npy", self.psi)


	def compute_phi(self):
		"""
		compute the potential

		:include_particles: bool, include particle density in phi calc
		:include_external: bool, include effect or external forces
	
		:return: array-like, [N^D]
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		rval = np.sum(np.abs(self.psi)**2, axis = 0)

		rval = self.GetFFt(rval, NoFieldDimension=True)

		K = self.K
		rval = -1*self.C*rval / K
		if self.D == 3:
			rval[0,0,0] = 0.0
		elif self.D ==2:
			rval[0,0] = 0.
		elif self.D == 1:
			rval[0] = 0

		rval = self.GetFFt(rval, Forward = False, NoFieldDimension=True)
		rval = rval.real

		return rval.real


	def get_dt(self, T_remaining, Vmax = None):
		"""
		calculate the timestep

		:T_remaining: float, the time remaining until the next data drop, \n
					default: do not consider this condition
		:Vmax: float, max value of the potential, default: calculate Vmax
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		dt_base = super().get_dt(T_remaining)
		
		k_max = np.sqrt(self.D) * np.pi / self.dx
		delta_k = 2 * k_max / self.N

		dt_kinetic = self.cf * 2. * np.pi / k_max / delta_k / np.max(self.hbar_)
		dt_kinetic_old = self.cf * 4 *np.pi / k_max**2 / np.max(self.hbar_)
		dt_kinetic = dt_kinetic if not(self.oldUpdateRule) else dt_kinetic_old
		
		dt_array = np.zeros(2)
		dt_array[0] = dt_base
		dt_array[1] = dt_kinetic
		return np.min(dt_array)


	def Drift(self, dt):
		"""
		update the dynamic variable positions

		:dt: float, timestep
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp

		k2max = (np.pi*self.N/self.L)**2

		self.psi = self.GetFFt(self.psi)

		self.psi *= np.exp(-1j*dt*\
            np.einsum("i,jkl->ijkl",self.hbar_, self.K)/(2.))

		rhoOverThresh = np.sum(np.abs(self.psi[self.K[np.newaxis,:,:,:] > (k2max*.9)])**2)
		
		rhoToT = np.sum(np.abs(self.psi)**2)
		self.rho_alias = rhoOverThresh / rhoToT

		self.psi = self.GetFFt(self.psi, Forward=False)


	def Kick(self, dt):
		"""
		update the dynamic variable momenta

		:dt: float, timestep
		"""
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		phi = self.compute_phi()

		self.psi *= np.exp(-1j*dt*\
            np.einsum("i,jkl->ijkl",1./self.hbar_, phi))

		Vmax = np.max(np.abs(phi))

		if not(self.oldUpdateRule):
			Vmax = self.GetMaxPhiGrad(phi)

		return Vmax


	def GetMaxPhiGrad(self,phi):
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		Vmax = np.abs(mu.gradient_1D(phi, self.dx, axis_ = 0, gpu = self.gpu)[2:self.N-2,2:self.N-2,2:self.N-2])**2
		Vmax += np.abs(mu.gradient_1D(phi, self.dx, axis_ = 1, gpu = self.gpu)[2:self.N-2,2:self.N-2,2:self.N-2])**2
		Vmax += np.abs(mu.gradient_1D(phi, self.dx, axis_ = 2, gpu = self.gpu)[2:self.N-2,2:self.N-2,2:self.N-2])**2
		Vmax = np.sqrt(np.max(Vmax))*self.dx

		return Vmax


	def Update(self, dt):
		"""
		updates the dynamic variables using a drift-kick-drift scheme

		:dt: float, the timestep
		"""
		Vmax = 0.
		if not(self.include_potential):
			self.Drift(dt)
		else:
			self.Drift(dt/2.)
			Vmax = self.Kick(dt)
			self.Drift(dt/2.)
		return Vmax


	def PrintDiagnostics(self, T, time0):
		"""
		prints diagnostic information

		:T: float, sim time completed
		:time0: float, real time the sim started
		"""
		str_ = self.BaseDiagnostics(T,time0)
		str_ += " %.2f alias fraction"%(self.rho_alias)
		su.repeat_print(str_)


	def RunSim(self):
		"""
		run the simulation
		"""
		self.checkPotentialConsistent()
		self.InitializeFiles()
		print("\nrunning simulation " + self.simName + "..." )
		time0 = time.time()
		tNext = float(self.Tf)/self.data_drops
		drop = 1
		T = 0.
		self.T_sim = T + self.T_initial
		while(T < self.Tf):
			T_remaining = tNext - T
			dt = self.get_dt(T_remaining)
			self.Update(dt)
			for i in range(len(self.Pulsars)):
				pulsar_ = self.Pulsars[i]
				pulsar_.Update(dt)
			T += dt 
			self.T_sim = T + self.T_initial
	
			self.PrintDiagnostics(T, time0)
			if T_remaining <= 0:
				# su.PrintTimeUpdate(drop,self.data_drops,time0)
				self.DataDrop(drop)
				Phi = self.compute_phi() / au.speed_of_light**2
				for i in range(len(self.Pulsars)):
					pulsar_ = self.Pulsars[i]
					pulsar_.DataDrop(drop, Phi)
				drop += 1
				tNext = float(self.Tf*drop)/self.data_drops

		if (drop == self.data_drops):
			self.DataDrop(drop)
			Phi = self.compute_phi() / au.speed_of_light**2
			for i in range(len(self.Pulsars)):
				pulsar_ = self.Pulsars[i]
				pulsar_.DataDrop(drop, Phi)

		for i in range(len(self.Pulsars)):
			pulsar_ = self.Pulsars[i]
			pulsar_.DataOutput()

		su.PrintCompletedTime(time0, "simulation")


	def GetFieldDensity(self):
		'''
		returns field density
		'''
		np = np_
		if CUPY_IMPORTED and self.gpu:
			np = cp
		return np.sum(np.abs(self.psi)**2, axis = 0)