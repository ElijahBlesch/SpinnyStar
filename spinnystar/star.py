import os, glob
import numpy as np
from .utils import prepare_interp, fast_interp
from . import wind_laws, coupling_laws, disk_locking_laws
from . import constants as const

class SpinnyStar:

    ################
    #Structure Data#
    ################

    def __init__(self, mass, data_dir, min_log_moi= -10, interpolate=True):

        """
        Loads dtrk structure data from Dartmouth Stellar Evolution Code (Convective Envelope -> Radiative Shell -> Convective Core)

        Parameters
        ---------------------------------------------------------------------------------------------------------------------------
        mass: float           mass in solar units
        data_dir: str         data directory path containing dtrk files
        min_log_moi: float    dtrk momentum of inertia values below this are floored to this value to prevent divide by zero errors
        interpolate: bool     interpolate along the dtrk data

        Attributes
        ---------------------------------------------------------------------------------------------------------------------------
        mass : float                                       total mass [solar units]
        mass_g : float                                     total mass [g] 
        
        age : ndarray                                      stellar age [yr]
        
        moi_env, moi_shell, moi_core : ndarray             zone moment of inertia [g cm^2]
        mass_env, mass_shell, mass_core : ndarray          zone mass [g]
        radius_env, radius_shell, radius_core : ndarray    radius to outer boundry of zone [cm]
        
        T_eff : ndarray                                    effective temperature [K]
        T_bcz : ndarray                                    temperature at the base of the convective envelope [K]
        T_brz : ndarray                                    temperature at the base of the radiative shell [K]
        
        rho_bcz : ndarray                                  density at the base of the convective envelope [g/m^3]
        rho_brz : ndarray                                  density at the base of the radiative shell [g/m^3]
        
        g : ndarray                                        surface gravity [cm/s^2]
        P : ndarray                                        surface pressure [dyn/cm^2] 
        B :  ndarray                                       surface magnetic field strength [G]
        L: ndarray                                         luminosity [erg/s]
        convec_turnover: ndarray                           convective cell turnover time [s]
        
        """


        #inputed parameters
        self.mass = mass
        self.mass_g = self.mass * const.M_SUN

        # Resolve data directory: use provided path, or fall back to current working directory
        if data_dir is None:
            data_dir = os.getcwd()
        self.data_dir = data_dir 

        #load in the needed files
        star_mass = f"{int(mass * 1000):04d}"
        star_data = glob.glob(os.path.join(data_dir, f"m{star_mass}*.dtrk"))
        star = np.genfromtxt(star_data[0], skip_header=12, names=True)

    ### ARRAYS FROM DATASET ###
        # Age (yr)
        self.age = star['age_Gyr'] * 1.0e9
        
        # Moment of Inertia (g cm^2)
        self.moi_shell = 10**(star['log10_I_rad'])
        if 'log10_I_cc' in star.dtype.names:
            self.moi_core = 10**(star['log10_I_cc'])
        else:
            self.moi_core = np.zeros(len(self.age))
        #Replace with min moi
        threshold = 10**(min_log_moi)
        index_shell = np.where(self.moi_shell > threshold)
        replace_shell = self.moi_shell[index_shell[0][0]] if (len(index_shell[0]) > 0) else 0
        index_core = np.where(self.moi_core > threshold)
        replace_core = self.moi_core[index_core[0][0]] if (len(index_core[0]) > 0) else 0
        
        #Moments of Inertia (g cm^2)
        self.moi_env = 10**(star['log10_I_ce'])
        self.moi_shell[self.moi_shell < threshold] = replace_shell
        self.moi_core[self.moi_core < threshold] = replace_core
        self.replace_core = replace_core
        self.replace_shell = replace_shell
        
        # Mass (g)
        if 'M_CCore_Msun' in star.dtype.names:
            self.mass_core = star['M_CCore_Msun'] * const.M_SUN
        else:
            self.mass_core = np.zeros(len(self.age))
        self.mass_env = star['M_CEnv_Msun'] * const.M_SUN
        self.mass_shell = self.mass_g - self.mass_env - self.mass_core
        
        # Radius (cm)
        self.radius_env   = 10**star['log10_R_Rsun'] * const.R_SUN
        self.radius_shell = self.radius_env * star['R_bcz_Rstar']
        self.radius_core  = self.radius_env * star['R_ccz_Rstar']
        
        #Convective Turnover
        self.convec_turnover = 10**star['log10_tau'] * const.SEC_PER_DAY
        
        #Luminosity
        self.L = 10**star['log10_L_Lsun'] * const.L_SUN
        
        #Temperature
        self.T_eff = 10**star['log10_Teff']
        self.T_bcz = 10**star['log10_T_bcz']
        self.T_brz = 10**star['log10_T_ccz']

        #Density
        self.rho_bcz = 10**star["log10_rho_bcz"]
        self.rho_brz = 10**star["log10_rho_ccz"]       
 
        #Surface gravity
        self.g = 10**star['log10_g']
        
        #Surface Pressure
        self.P = 10**star['log10_P_surf']   
        
        #Surface Magnetic Field
        self.B = 10**star['log10_B_surf']


    
    ### INTERPOLATED ARRAYS ###
        ms_expansion_cutoff = np.where(self.radius_env > 1.5 * min(self.radius_env))[0]
        ms_expansion_cutoff = ms_expansion_cutoff[ms_expansion_cutoff > np.argmin(self.radius_env)]
        if ms_expansion_cutoff.size > 0:
            max_age = self.age[ms_expansion_cutoff[0]]
        else:
            max_age = self.age[-1]
        
        if interpolate:
            #Set up fast interp
            t_sorted, idx, t_finite = prepare_interp(self.age)
            
            #Age [yr]
            self.time_points = np.geomspace(1_000_000, max_age, num=15_000)
            # Moment of Inertia
            self.moi_core = fast_interp(self.time_points, t_sorted, idx, t_finite, self.moi_core)
            self.moi_env = fast_interp(self.time_points, t_sorted, idx, t_finite, self.moi_env)
            self.moi_shell = fast_interp(self.time_points, t_sorted, idx, t_finite, self.moi_shell)
            # Mass
            self.mass_core = fast_interp(self.time_points, t_sorted, idx, t_finite, self.mass_core)
            self.mass_shell = fast_interp(self.time_points, t_sorted, idx, t_finite, self.mass_shell)
            self.mass_env = fast_interp(self.time_points, t_sorted, idx, t_finite, self.mass_env)
            # Radius
            self.radius_core = fast_interp(self.time_points, t_sorted, idx, t_finite, self.radius_core)
            self.radius_shell = fast_interp(self.time_points, t_sorted, idx, t_finite, self.radius_shell)
            self.radius_env = fast_interp(self.time_points, t_sorted, idx, t_finite, self.radius_env)
            #Convective Turnover
            self.convec_turnover = fast_interp(self.time_points, t_sorted, idx, t_finite, self.convec_turnover)
            #Luminosity
            self.L = fast_interp(self.time_points, t_sorted, idx, t_finite, self.L)
            #Temperature
            self.T_eff = fast_interp(self.time_points, t_sorted, idx, t_finite, self.T_eff)
            self.T_bcz = fast_interp(self.time_points, t_sorted, idx, t_finite, self.T_bcz)
            self.T_brz = fast_interp(self.time_points, t_sorted, idx, t_finite, self.T_brz)
            #Density
            self.rho_bcz = fast_interp(self.time_points, t_sorted, idx, t_finite, self.rho_bcz)
            self.rho_brz = fast_interp(self.time_points, t_sorted, idx, t_finite, self.rho_brz)
            #Surface Gravity
            self.g = fast_interp(self.time_points, t_sorted, idx, t_finite, self.g)
            #Surface Pressure
            self.P = fast_interp(self.time_points, t_sorted, idx, t_finite, self.P) 
            #Surface Magnetic Field
            self.B = fast_interp(self.time_points, t_sorted, idx, t_finite, self.B)

            self.age = self.time_points

        self._mass_core_raw = self.mass_core.copy()
        self._moi_core_raw = self.moi_core.copy()
        self._radius_core_raw = self.radius_core.copy()

    ###############################                    
    #Two Zone Rotational Evolution#                     
    ###############################
    
    def two_zone_evolution(self, initial_period, wind_model, coupling_model, disk_locking_model,
                         wind_const=4.5e47, coupling_const=3.5e-5, coupling_power=4.5,
                         rossby_saturation_threshold= 0.1, solar_rotation= 'avg',
                         use_mass_rate=True, use_moi_change=True, use_momentum_exchange=True, use_wind_braking=True,
                         coupling_data_path=None):

        """
        Rotational Evolution of a two-zone stellar model (Convective Envelope -> Radiative Core)

        Parameters
        -----------------------------------------------------------------------------------------------------------------------------------------------
        initial_period: float              starting rotation period [days]

        wind_model: str                    wind model of choice (models found in wind_laws.py) 
        coupling_model: str                coupling model of choice (models found in coupling_laws.py)
        disk_locking_model: float / str    disk Locking model of choice (models found in disk_locking_laws.py) or float for constant lifetime [Myr]

        wind_const: float                  wind model free parameter (set to current best fit)
        coupling_const: float              coupling model free parameter (set to current best fit)
        coupling_power: float              coupling mass dependence (set to current best fit)

        use_*: bool                        angular velocity component toggles
 
        coupling_data_path: str            data directory path containing coupling timescales

        Attributes
        -----------------------------------------------------------------------------------------------------------------------------------------------
        omega_env, omega_core: ndarray                angular velocity of each zone [rad/s]
        period_env, period_core: ndarray              rotation period of each zone [days]
        delta_omega_env, delta_omega_core: ndarray    difference in angular from current step and previous [rad/s]
        rossby_number: ndarray                        ratio of envelope rotation period to convective turnover time [dimensionless]
        J_star, J_wind, J_total: ndarray              angular momentum budget [g cm^2/s]
        J_lost: float                                 conservation error
        term_*: ndarray                               per step angular velocity contribution

        NOTE: The convective core and radiative shell properties are combined into one core for two zone evolution
        NOTE: Free parameter presets are set to the "test" wind_model and "Frictional_Torque" coupling_law
        """
        
        #ID for Tools
        self.model = 'two_zone_evolution'
        
        #Inputs
        self.initial_period = initial_period * const.SEC_PER_DAY
        self.rossby_saturation_threshold = rossby_saturation_threshold
        self.wind_model = wind_model
        self.coupling_model = coupling_model
        self.disk_locking_model = disk_locking_model
        self.solar_rotation = solar_rotation

        #Localize Stellar Parameters
        age = self.age
        moi_core, moi_env = (self._moi_core_raw + self.moi_shell), self.moi_env
        mass_core, mass_env = (self._mass_core_raw + self.mass_shell), self.mass_env
        radius_core, radius_env = self.radius_shell, self.radius_env
        convec_turnover = self.convec_turnover
        L, T_eff,T_bcz, g, M, P, B = self.L, self.T_eff, self.T_bcz, self.g, self.mass_g, self.P, self.B
        rho_bcz, rho_brz = self.rho_bcz, self.rho_brz
        
        #Empty Arrays to be Written#
        N = len(self.age)
        #Output Arrays
        self.J_dot = np.zeros(N)
        self.tau_wb = np.zeros(N)
        #Loop-Only Arrays
        J_env, J_core = np.zeros(N), np.zeros(N)
        J_m_env, J_m_core = np.zeros(N), np.zeros(N)
        dJ = np.zeros(N)
        J_fric = np.zeros(N)
        omega_env, omega_core = np.zeros(N), np.zeros(N)
        Omega_m_env, Omega_m_core = np.zeros(N), np.zeros(N)
        Omega_moi_env, Omega_moi_core = np.zeros(N), np.zeros(N)
        Omega_env_coupling, Omega_core_coupling = np.zeros(N), np.zeros(N)
        Omega_env_fric, Omega_core_fric = np.zeros(N), np.zeros(N)
        Omega_wb = np.zeros(N)
        env_momentum_exchange, core_momentum_exchange = np.zeros(N), np.zeros(N)
        

        #Time Step
        dt = np.diff(self.age) * const.SEC_PER_YR

        #Set Initial Period
        omega_env[0]  = 2*np.pi / self.initial_period
        omega_core[0] = 2*np.pi / self.initial_period

        #Mass Exchange
        dM = np.diff(mass_core)
        
        #Check for Core Existence
        core_threshold = 2e-3 * M
        core_exists = (mass_core > core_threshold) & (moi_core > 2 * self.replace_core)

        #Coupling and Disk Locking Models
        self.disk_locking = disk_locking_laws.disk_locking(disk_locking_model, self.mass)
        if coupling_model != 'Frictional_Torque':
            self.coupling = coupling_laws.coupling(
                model=self.model, coupling_model=coupling_model, data_path=coupling_data_path, mass=M)
        elif coupling_model == 'Frictional_Torque':
            self.coupling = 0
        elif coupling_model == 'Turbulence':
            self.coupling = 0
        
        
        for i in range(N-1):
            
        ### Component Momentum ###
            if core_exists[i]:
                J_env[i] = moi_env[i] * omega_env[i]
                J_core[i] = moi_core[i] * omega_core[i]  
            elif not core_exists[i]:
                J_env[i] = moi_env[i] * omega_env[i]
                J_core[i] = 0

        ### Mass Exchange Term ###
            
            #Mass Exchange Momentum Contribution
            if dM[i] > 0  and core_exists[i]:
                J_m_env[i] = -(2/3) * dM[i] * radius_core[i]**2 * omega_env[i]
                J_m_core[i] = (2/3) * dM[i] * radius_core[i]**2 * omega_env[i]
            elif dM[i] < 0  and core_exists[i]:
                J_m_env[i] = -(2/3) * dM[i] * radius_core[i]**2 * omega_core[i]
                J_m_core[i] = (2/3) * dM[i] * radius_core[i]**2 * omega_core[i]
            elif dM[i] < 0  and not core_exists[i]:
                J_m_env[i] = -(2/5) * dM[i] * radius_core[i]**2 * omega_core[i]
                J_m_core[i] = 0
            else:
                J_m_env[i] = 0
                J_m_core[i] = 0

            #Mass Exchange Omega Contribution
            Omega_m_env[i] = J_m_env[i] / moi_env[i+1]
            Omega_m_core[i] = (J_m_core[i] / moi_core[i+1]) if core_exists[i+1] else 0

        ### MOI Change Term ###
            if core_exists[i]:
                Omega_moi_env[i] = -(omega_env[i]/moi_env[i+1]) * (moi_env[i+1]-moi_env[i])
                Omega_moi_core[i] = -(omega_core[i]/moi_core[i+1]) * (moi_core[i+1]-moi_core[i])
            else:
                Omega_moi_env[i] = -(omega_env[i]/moi_env[i+1]) * (moi_env[i+1]-moi_env[i])
                Omega_moi_core[i] = 0 

        ### Coupling Term ###
            if coupling_model != 'Frictional_Torque' and coupling_model != 'Turbulence':    
                if core_exists[i]:
                    dJ[i] = (moi_env[i] * J_core[i] - moi_core[i] * J_env[i]) / (moi_core[i] + moi_env[i])
                    Omega_env_coupling[i] = (dJ[i] / self.coupling) * (dt[i] / moi_env[i+1])
                    Omega_core_coupling[i] = -(dJ[i] / self.coupling) * (dt[i] / moi_core[i+1])
                else: 
                    Omega_env_coupling[i], Omega_core_coupling[i] = 0, 0

            elif coupling_model == 'Frictional_Torque' or coupling_model == 'Turbulence': 
                if core_exists[i]:
                    J_fric[i] = coupling_laws.coupling(
                                    model=self.model, coupling_model=coupling_model, data_path=coupling_data_path, mass=M, 
                                    coupling_const=coupling_const, coupling_power=coupling_power, 
                                    R_bcz=radius_core[i], T_bcz=T_bcz[i], rho_bcz=rho_bcz[i],
                                    omega_env=omega_env[i], omega_core=omega_core[i])
                    Omega_env_fric[i] = J_fric[i] * dt[i] / moi_env[i+1]
                    Omega_core_fric[i] = - J_fric[i] * dt[i] / moi_core[i+1]
                else:
                    Omega_env_fric[i], Omega_core_fric[i] = 0, 0
            
        ### Windbraking Law ###
            self.J_dot[i], Omega_wb[i] = wind_laws.wind_braking(wind_model, coupling_model, use_momentum_exchange, solar_rotation, wind_const,
                                    moi_core[i], moi_env[i], moi_env[i+1], omega_env[i],
                                    convec_turnover[i], rossby_saturation_threshold,
                                    dt[i], L[i], radius_env[i], M, T_eff[i], g[i], P[i], B[i])
    
            #Windbraking Efficiency
            self.tau_wb[i] = (self.J_dot[i] / J_env[i])**(-1)

        ### Component Toggles ###
            env_mass_rate_i = Omega_m_env[i] if use_mass_rate else 0.0
            core_mass_rate_i = Omega_m_core[i] if (use_mass_rate and core_exists[i]) else 0.0
            env_moi_change_i = Omega_moi_env[i]  if use_moi_change else 0.0
            core_moi_change_i = Omega_moi_core[i] if use_moi_change else 0.0
            env_coupling_i = Omega_env_coupling[i]  if use_momentum_exchange else 0.0
            core_coupling_i = Omega_core_coupling[i] if use_momentum_exchange else 0.0
            env_fric_i = Omega_env_fric[i]  if use_momentum_exchange else 0.0
            core_fric_i = Omega_core_fric[i] if use_momentum_exchange else 0.0
            wind_braking_i = Omega_wb[i] if use_wind_braking else 0.0

            if coupling_model != 'Frictional_Torque' and coupling_model != 'Turbulence':
                env_momentum_exchange_i = env_coupling_i
                core_momentum_exchange_i = core_coupling_i
            elif coupling_model == 'Frictional_Torque' or coupling_model == 'Turbulence':
                env_momentum_exchange_i = env_fric_i
                core_momentum_exchange_i = core_fric_i

            #Terms used in plots
            env_momentum_exchange[i] = env_momentum_exchange_i
            core_momentum_exchange[i] = core_momentum_exchange_i 

        ### Spinny Evolution ###
            if (self.age[i] > self.disk_locking):
                omega_env[i+1]  = omega_env[i] + env_mass_rate_i + env_momentum_exchange_i + env_moi_change_i  + wind_braking_i
                omega_core[i+1] = omega_core[i] + core_mass_rate_i + core_momentum_exchange_i + core_moi_change_i
            else:
                omega_env[i+1] = omega_env[i]
                omega_core[i+1] = omega_core[i] + core_mass_rate_i + core_moi_change_i
 
            #Restart Core Rotation if it pops back into existence
            if core_exists[i+1] and (i == 0 or not core_exists[i]):
                omega_core[i+1] = omega_env[i+1]


        #Spinny Evolution Outputs
        
        self.omega_env = omega_env
        self.omega_core = omega_core
        self.omega_core[~core_exists] = np.nan
        self.period_env = 2*np.pi / omega_env / const.SEC_PER_DAY
        self.period_core = np.where(core_exists, 2*np.pi / omega_core / const.SEC_PER_DAY, np.nan)

        self.delta_omega_env, self.delta_omega_core = np.diff(self.omega_env), np.diff(self.omega_core)
        self.rossby_number = (self.period_env * const.SEC_PER_DAY) / self.convec_turnover

        #Set Core Terms in Two zone to be the Core and Shell
        self.mass_core, self.moi_core, self.radius_core = mass_core, moi_core, radius_core

        #Angular Momentum Check
        J_star = J_env + J_core
        self.J_star = J_star[:-1]
        self.J_wind = np.cumsum(self.J_dot[:-1] * dt)
        self.J_total = self.J_star - self.J_wind
        self.J_lost = self.J_total[0]-self.J_total[-1]
       

        #Term Contributions Plot
        self.term_env_mass_rate = Omega_m_env[:-1]
        self.term_core_mass_rate = Omega_m_core[:-1]
        self.term_env_moi_change = Omega_moi_env[:-1]
        self.term_core_moi_change = Omega_moi_core[:-1]
        self.term_env_momentum_exchange = env_momentum_exchange[:-1]
        self.term_core_momentum_exchange = core_momentum_exchange[:-1]
        self.term_wind_braking = Omega_wb[:-1]      
        
    #################################                  
    #Three Zone Rotational Evolution#                   
    #################################                  

    def three_zone_evolution(self, initial_period, wind_model, coupling_model, disk_locking_model,
                         wind_const=4.5e47, coupling_const=3.5e-5, coupling_power=4.5,
                         rossby_saturation_threshold= 0.1, solar_rotation= 'avg',
                         use_mass_rate=True, use_moi_change=True, use_momentum_exchange=True, use_wind_braking=True,
                         coupling_data_path=None):

        """
        Rotational Evolution of a three-zone stellar model (Convective Envelope -> Radiative Shell -> Convective Core)

        Parameters
        -----------------------------------------------------------------------------------------------------------------------------------------------
        initial_period: float              starting rotation period [days]

        wind_model: str                    wind model of choice (models found in wind_laws.py) 
        coupling_model: str                coupling model of choice (models found in coupling_laws.py)
        disk_locking_model: float / str    disk Locking model of choice (models found in disk_locking_laws.py) or float for constant lifetime [Myr]

        use_*: bool                        angular velocity component toggles
 
        coupling_data_path: str            data directory path containing coupling timescales

        Attributes
        -----------------------------------------------------------------------------------------------------------------------------------------------
        omega_env, omega_shell, omega_core: ndarray                      angular velocity of each zone [rad/s]
        period_env, period_shell, period_core: ndarray                   rotation period of each zone [days]
        delta_omega_env, delta_omega_shell, delta_omega_core: ndarray    difference in angular from current step and previous [rad/s]
        rossby_number: ndarray                                           ratio of envelope rotation period to convective turnover time [dimensionless]
        J_star, J_wind, J_total: ndarray                                 angular momentum budget [g cm^2/s]
        J_lost: float                                                    conservation error
        term_*: ndarray                                                  per step angular velocity contribution
        """

        #ID for Tools
        self.model = 'three_zone_evolution'

        #Inputs
        self.initial_period = initial_period * const.SEC_PER_DAY
        self.rossby_saturation_threshold = rossby_saturation_threshold
        self.wind_model = wind_model
        self.coupling_model = coupling_model
        self.disk_locking_model = disk_locking_model
        self.solar_rotation = solar_rotation

        #Localize Stellar Parameters
        age = self.age
        moi_env, moi_shell, moi_core = self.moi_env, self.moi_shell, self._moi_core_raw
        mass_env, mass_shell, mass_core = self.mass_env, self.mass_shell, self._mass_core_raw
        radius_env, radius_shell, radius_core = self.radius_env, self.radius_shell, self._radius_core_raw
        convec_turnover = self.convec_turnover
        L, T_eff, T_bcz, T_brz, g, M, P, B = self.L, self.T_eff, self.T_bcz, self.T_brz, self.g, self.mass_g, self.P, self.B
        moi_core_shell = moi_core + moi_shell

        #Empty Arrays to be Written#
        N = len(self.age)
        #Output Arrays
        self.J_dot = np.zeros(N)
        self.tau_wb = np.zeros(N)
        #Loop-Only Arrays
        J_env, J_shell, J_core = np.zeros(N), np.zeros(N), np.zeros(N)
        J_m_env, J_m_shell_env, J_m_shell_core, J_m_core = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
        dJ_shell_env = np.zeros(N)
        J_shell_env_fric, J_shell_core_fric = np.zeros(N), np.zeros(N)
        omega_env, omega_shell, omega_core = np.zeros(N), np.zeros(N), np.zeros(N)
        Omega_m_env, Omega_m_shell_env, Omega_m_shell_core, Omega_m_core = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
        Omega_moi_env, Omega_moi_shell, Omega_moi_core = np.zeros(N), np.zeros(N), np.zeros(N)
        Omega_env_coupling, Omega_shell_coupling = np.zeros(N), np.zeros(N)
        Omega_env_fric, Omega_shell_env_fric, Omega_shell_core_fric, Omega_core_fric = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
        Omega_wb = np.zeros(N)
        env_momentum_exchange, shell_env_momentum_exchange, shell_core_momentum_exchange, core_momentum_exchange = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
        
        #Time Step
        dt = np.diff(self.age) * const.SEC_PER_YR

        #Set Initial Period
        omega_env[0]  = 2*np.pi / self.initial_period
        omega_shell[0] = 2*np.pi / self.initial_period
        omega_core[0] = 2*np.pi / self.initial_period
        
        #Mass Exchange
        dM_shell_core= np.diff(mass_core)
        dM_shell_env = -np.diff(mass_env) #Minus to keep same sign notation as the Two Zone

        #Check for Existence
        shell_threshold = 1e-2 * M
        core_threshold = 1e-4 * M
        shell_exists = (mass_shell > shell_threshold) & (moi_shell > 2 * self.replace_shell)
        core_exists = (mass_core > core_threshold) & (moi_core > 2 * self.replace_core)


        #Coupling and Disk Locking Models
        self.disk_locking = disk_locking_laws.disk_locking(disk_locking_model, self.mass)
        if coupling_model != 'Frictional_Torque':
            self.shell_env_coupling  = coupling_laws.coupling(
                model=self.model, coupling_model=coupling_model, data_path=coupling_data_path, mass=M)


        for i in range(N-1):

        ### Component Momentum ###
            if shell_exists[i] and core_exists[i]:
                J_env[i] = moi_env[i] * omega_env[i]
                J_shell[i] = moi_shell[i] * omega_shell[i]   
                J_core[i] = moi_core[i] * omega_core[i]
            elif shell_exists[i] and not core_exists[i]:
                J_env[i] = moi_env[i] * omega_env[i]
                J_shell[i] = moi_shell[i] * omega_shell[i]   
                J_core[i] = 0
            elif not shell_exists[i] and not core_exists[i]:
                J_env[i] = moi_env[i] * omega_env[i]
                J_shell[i] = 0  
                J_core[i] = 0

        ###  Mass Exchange Term ###
            
            #Mass Exchange Momentum Contribution
            if dM_shell_env[i] > 0 and shell_exists[i]:
                J_m_env[i] = -(2/3) * dM_shell_env[i] * radius_shell[i]**2 * omega_env[i]
                J_m_shell_env[i] = (2/3) * dM_shell_env[i] * radius_shell[i]**2 * omega_env[i]
            elif dM_shell_env[i] < 0 and shell_exists[i]:
                J_m_env[i] = -(2/3) * dM_shell_env[i] * radius_shell[i]**2 * omega_shell[i]
                J_m_shell_env[i] = (2/3) * dM_shell_env[i] * radius_shell[i]**2 * omega_shell[i]
            if dM_shell_core[i] > 0 and core_exists[i]:
                J_m_shell_core[i] = -(2/3) * dM_shell_core[i] * radius_core[i]**2 * omega_shell[i]
                J_m_core[i] = (2/3) * dM_shell_core[i] * radius_core[i]**2 * omega_shell[i]
            elif dM_shell_core[i] < 0 and core_exists[i]:
                J_m_shell_core[i] = -(2/3) * dM_shell_core[i] * radius_core[i]**2 * omega_core[i]
                J_m_core[i] = (2/3) * dM_shell_core[i] * radius_core[i]**2 * omega_core[i]
        

            #Do the Spherical Mass Rate for when the core finally disappears?

            #Mass Exchange Omega Contribution
            Omega_m_env[i] = J_m_env[i] / moi_env[i+1]
            Omega_m_shell_env[i] = (J_m_shell_env[i] / moi_shell[i+1]) if shell_exists[i+1] else 0
            Omega_m_shell_core[i] = (J_m_shell_core[i] / moi_shell[i+1]) if shell_exists[i+1] else 0
            Omega_m_core[i] = (J_m_core[i] / moi_core[i+1]) if core_exists[i+1] else 0

        ### MOI Change Term ###
            if shell_exists[i] and core_exists[i]:
                Omega_moi_env[i] = -(omega_env[i]/moi_env[i+1]) * (moi_env[i+1]-moi_env[i])
                Omega_moi_shell[i] = -(omega_shell[i]/moi_shell[i+1]) * (moi_shell[i+1]-moi_shell[i])
                Omega_moi_core[i] = -(omega_core[i]/moi_core[i+1]) * (moi_core[i+1]-moi_core[i])
            elif shell_exists[i] and not core_exists[i]:
                Omega_moi_env[i] = -(omega_env[i]/moi_env[i+1]) * (moi_env[i+1]-moi_env[i])
                Omega_moi_shell[i] = -(omega_shell[i]/moi_shell[i+1]) * (moi_shell[i+1]-moi_shell[i])
                Omega_moi_core[i] = 0
            elif not shell_exists[i] and not core_exists[i]:
                Omega_moi_env[i] = -(omega_env[i]/moi_env[i+1]) * (moi_env[i+1]-moi_env[i])
                Omega_moi_shell[i] = 0
                Omega_moi_core[i] = 0

        ### Coupling Term ###
            if coupling_model != 'Frictional_Torque':
                if shell_exists[i]:
                    dJ_shell_env[i] = (moi_env[i] * J_shell[i] - moi_shell[i] * J_env[i]) / (moi_shell[i] + moi_env[i])
                    Omega_env_coupling[i] = (dJ_shell_env[i]/self.shell_env_coupling) * (dt[i] / moi_env[i+1])
                    Omega_shell_coupling[i] = -(dJ_shell_env[i]/self.shell_env_coupling) * (dt[i] / moi_shell[i+1])
                else: 
                    Omega_env_coupling[i], Omega_shell_coupling[i] = 0, 0
            
            elif coupling_model == 'Frictional_Torque':
                if shell_exists[i] and core_exists[i] :
                    J_shell_env_fric[i], J_shell_core_fric[i] = coupling_laws.coupling(self.model, coupling_model, data_path= coupling_data_path, mass=M,
                                                    coupling_const=coupling_const, coupling_power=coupling_power, 
                                                    R_bcz=radius_shell[i], R_brz=radius_core[i], T_bcz=T_bcz[i], T_brz=T_brz[i], 
                                                    omega_env=omega_env[i], omega_shell=omega_shell[i], omega_core=omega_core[i])
                    
                    Omega_env_fric[i] = J_shell_env_fric[i] * dt[i] / moi_env[i+1] 
                    Omega_shell_env_fric[i] = -J_shell_env_fric[i] * dt[i] / moi_shell[i+1] 
                    Omega_shell_core_fric[i] = J_shell_core_fric[i] * dt[i] / moi_shell[i+1]  
                    Omega_core_fric[i] = -J_shell_core_fric[i] * dt[i] / moi_core[i+1]
                
                elif shell_exists[i] and not core_exists[i]:
                    J_shell_env_fric[i], J_shell_core_fric[i] = coupling_laws.coupling(self.model, coupling_model, data_path= coupling_data_path, mass=M,
                                                    coupling_const=coupling_const, coupling_power=coupling_power, 
                                                    R_bcz=radius_shell[i], T_bcz=T_bcz[i], T_brz=T_brz[i], omega_env=omega_env[i], omega_shell=omega_shell[i])
                    
                    Omega_env_fric[i] = J_shell_env_fric[i] * dt[i] / moi_env[i+1] 
                    Omega_shell_env_fric[i] = -J_shell_env_fric[i] * dt[i] / moi_shell[i+1] 
                    Omega_shell_core_fric[i], Omega_core_fric[i] = 0,0
                
                elif not shell_exists[i] and not core_exists[i]:
                    Omega_env_fric[i], Omega_shell_env_fric[i], Omega_shell_core_fric[i], Omega_core_fric[i] = 0, 0, 0, 0

        ### Windbraking Law ###
            self.J_dot[i], Omega_wb[i] = wind_laws.wind_braking(wind_model, coupling_model, use_momentum_exchange, solar_rotation, wind_const, 
                                                                moi_core_shell[i], moi_env[i], moi_env[i+1], omega_env[i],
                                                                convec_turnover[i], rossby_saturation_threshold,
                                                                dt[i], L[i], radius_env[i], M, T_eff[i], g[i], P[i], B[i])
            #Windbraking Efficiency
            self.tau_wb[i] = (self.J_dot[i] / J_env[i])**(-1)

        ### Component Toggles ###
            env_mass_rate_i = Omega_m_env[i] if use_mass_rate else 0.0
            shell_env_mass_rate_i = Omega_m_shell_env[i] if (use_mass_rate and shell_exists[i]) else 0.0
            shell_core_mass_rate_i = Omega_m_shell_core[i] if (use_mass_rate and core_exists[i] and shell_exists[i]) else 0.0
            core_mass_rate_i = Omega_m_core[i] if (use_mass_rate and core_exists[i]) else 0.0
            env_moi_change_i = Omega_moi_env[i]  if use_moi_change else 0.0
            shell_moi_change_i = Omega_moi_shell[i] if use_moi_change else 0.0
            core_moi_change_i = Omega_moi_core[i] if use_moi_change else 0.0
            env_coupling_i = Omega_env_coupling[i]  if use_momentum_exchange else 0.0
            shell_coupling_i = Omega_shell_coupling[i] if use_momentum_exchange else 0.0
            env_fric_i = Omega_env_fric[i] if use_momentum_exchange else 0.0
            shell_env_fric_i = Omega_shell_env_fric[i] if use_momentum_exchange else 0.0
            shell_core_fric_i = Omega_shell_core_fric[i] if use_momentum_exchange else 0.0
            core_fric_i = Omega_core_fric[i] if use_momentum_exchange else 0.0   
            wind_braking_i = Omega_wb[i] if use_wind_braking else 0.0

            if coupling_model != 'Frictional_Torque':
                env_momentum_exchange_i = env_coupling_i
                shell_env_momentum_exchange_i = shell_coupling_i
                shell_core_momentum_exchange_i = 0
                core_momentum_exchange_i = 0
            elif coupling_model == 'Frictional_Torque':
                env_momentum_exchange_i = env_fric_i
                shell_env_momentum_exchange_i = shell_env_fric_i
                shell_core_momentum_exchange_i = shell_core_fric_i
                core_momentum_exchange_i = core_fric_i

            #Terms used in plots
            env_momentum_exchange[i] = env_momentum_exchange_i
            shell_env_momentum_exchange[i] = shell_env_momentum_exchange_i
            shell_core_momentum_exchange[i] = shell_core_momentum_exchange_i
            core_momentum_exchange[i] = core_momentum_exchange_i 

        ### Spinny Evolution ###
            if (self.age[i] > self.disk_locking):
                omega_env[i+1]  = omega_env[i]  + env_mass_rate_i  + env_momentum_exchange_i + env_moi_change_i  + wind_braking_i
                omega_shell[i+1] = omega_shell[i] + shell_env_mass_rate_i + shell_core_mass_rate_i + shell_env_momentum_exchange_i + shell_core_momentum_exchange_i + shell_moi_change_i
                omega_core[i+1] = omega_core[i] + core_mass_rate_i + core_moi_change_i + core_momentum_exchange_i
            else:
                omega_env[i+1] = omega_env[i]
                omega_shell[i+1] = omega_shell[i] + shell_env_mass_rate_i + shell_core_mass_rate_i + shell_moi_change_i
                omega_core[i+1] = omega_core[i] + core_mass_rate_i + core_moi_change_i
 
            #Restart Core Rotation if it pops back into existence
            if core_exists[i+1] and (i == 0 or not core_exists[i]):
                omega_core[i+1] = omega_shell[i+1]
            if shell_exists[i+1] and (i == 0 or not shell_exists[i]):
                omega_shell[i+1] = omega_env[i+1]

        #Spinny Evolution Outputs
    
        self.omega_env = omega_env
        self.omega_shell = omega_shell
        self.omega_core = omega_core
        self.omega_shell[~shell_exists] = np.nan
        self.omega_core[~core_exists] = np.nan

        self.delta_omega_env, self.delta_omega_shell, self.delta_omega_core = np.diff(self.omega_env), np.diff(self.omega_shell), np.diff(self.omega_core)
        self.period_env = 2*np.pi / omega_env / const.SEC_PER_DAY
        self.period_shell = np.where(shell_exists, 2*np.pi / omega_shell / const.SEC_PER_DAY, np.nan)
        self.period_core = np.where(core_exists, 2*np.pi / omega_core / const.SEC_PER_DAY, np.nan)
        self.rossby_number = (self.period_env * const.SEC_PER_DAY) / self.convec_turnover

        #Set Core Terms to the zones this model used (convective core; plots stay model-agnostic)
        self.mass_core, self.moi_core, self.radius_core = mass_core, moi_core, radius_core

        #Angular Momentum Check
        J_star = J_env + J_shell + J_core
        self.J_star = J_star[:-1]
        self.J_wind = np.cumsum(self.J_dot[:-1] * dt)
        self.J_total = self.J_star - self.J_wind
        self.J_lost = self.J_total[0]-self.J_total[-1]
       

        #Omega Contributions 
        self.term_env_moi_change = Omega_moi_env[:-1]
        self.term_shell_moi_change = Omega_moi_shell[:-1]
        self.term_core_moi_change = Omega_moi_core[:-1]
        self.term_env_mass_rate = Omega_m_env[:-1]
        self.term_shell_env_mass_rate = Omega_m_shell_env[:-1]
        self.term_shell_core_mass_rate = Omega_m_shell_core[:-1]
        self.term_core_mass_rate = Omega_m_core[:-1]
        self.term_wind_braking = Omega_wb[:-1]
        self.term_env_momentum_exchange = env_momentum_exchange[:-1]
        self.term_shell_env_momentum_exchange = shell_env_momentum_exchange[:-1]
        self.term_shell_core_momentum_exchange = shell_core_momentum_exchange[:-1]
        self.term_core_momentum_exchange = core_momentum_exchange[:-1] 

    ####################
    #Save to RTRK Files# <------ Broken
    ####################

    def save_data(self, save_path=None):

        """Write this star's evolution to a .rtrk file. See utils.save_rtrk."""

        from .utils import save_rtrk
        return save_rtrk(self, save_path)

                
