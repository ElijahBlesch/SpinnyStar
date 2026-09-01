"""
coupling_laws.py -- apply angular momentum transport formulation of choice

Models
-------------------------------------------------------------------------------------------------------------
'SpadaPL':              Spada and Lanzafame 2020 power law
'SpadaBPL':             Spada and Lanzafame 2025 split power law
'EnvGrowth':            Tracks the timescale in which the envelope is growing down into the core
'CoreTemp':             Tracks the core temperature instead of radius to determine envelope growth timescale
'test':                 simple coupling timescale play model
'Frictional_Torque':    momentum transfer through molecular friction
'Turbulence':           momentum transported through turbulent motions

Returns
-------------------------------------------------------------------------------------------------------------
NOTE: Returns are dependent on the model. Some need to be calculated once while others are per step

timescale:    (SpadaPL, SpadaBPL, EnvGrowth, CoreTemp, test) returns a timescale in [s], called once per star
torque:       (Frictional_Torque, Turbulence) returns a torque [g cm^2 / s^2], called per step

"""

import numpy as np
from scipy.interpolate import interp1d
import os
from . import constants as const

data_dir = os.getcwd()

def coupling(model, coupling_model, data_path, mass,
             R_bcz=0, R_brz=0, T_bcz=0, rho_bcz=0, T_brz=0, rho_brz=0, 
             omega_env=0, omega_shell=0, omega_core=0):
    solar_coupling = 22 * const.SEC_PER_MYR
    mass_frac = np.round((mass / const.M_SUN), 2)
    
    if coupling_model == 'SpadaPL':
        a = 5.6
        coupling = solar_coupling*(mass_frac)**-a
        return coupling
    
    elif coupling_model == 'SpadaBPL':
        tau_0 = 565 * const.SEC_PER_MYR
        a = 3.5 #Calibrated to Unbroken Power Law
        b = 6.22
        if mass_frac < 0.6:
            coupling = tau_0 * (mass_frac / 0.6)**-a 
        elif mass_frac >= 0.6:
            coupling = tau_0 * (mass_frac / 0.6)**-b 
        return coupling
    
    elif coupling_model == 'EnvGrowth':
        if data_path is None:
            data_path = os.path.join(data_dir, "conv_growth_gs98.csv")
        stall = np.genfromtxt(data_path, delimiter=",", skip_header=1)
        stall_int_curve = interp1d(stall[:,0], stall[:,1:4], axis=0, kind="linear", bounds_error=False)
        stall_epoch_est = stall_int_curve(mass_frac)
        conv_growth_timescale = (stall_epoch_est[1] - stall_epoch_est[0]) * const.SEC_PER_GYR 
        convec_growth_end = (stall_epoch_est[1]) * const.SEC_PER_GYR
        return conv_growth_timescale
    
    elif coupling_model == 'CoreTemp':
        if data_path is None:
            data_path = os.path.join(data_dir, "core_temp_minimum.csv")
        temp = np.genfromtxt(data_path, delimiter=",", skip_header=1)
        temp_int_curve = interp1d(temp[:,0], temp[:,1:4], axis=0, kind="linear", bounds_error=False)
        temp_epoch_est = temp_int_curve(mass_frac)
        min_core_temp = temp_epoch_est[0] * const.SEC_PER_GYR
        return min_core_temp
        
    elif coupling_model == 'test':
        coupling = 28*(mass_frac)**-6.2 #33.05, 6.2
        return coupling * const.SEC_PER_MYR
    
    elif coupling_model == 'Frictional_Torque':
        if model == 'two_zone_evolution':
            mu = 1.6        #Mean Molecular Weight, Placeholder For now
            
            f = (mass_frac)**4.5
            eta = 3.50e-5 * f     # adjusted for mean molecular weight change
            
            #Constants
            C = 9.0 * eta * (const.k_boltzmann**2 / const.e_esu**4) * ((mu * const.m_p) / 3.0)**0.5

            #Torque
            J_fric = C * T_bcz**(2.5) * R_bcz**3 * (omega_core - omega_env)

            return J_fric
        
        elif model == 'three_zone_evolution':
            f = (mass_frac)**4.5

        ### Shell-Envelope ###
            mu_es = 1.6           # Mean Molecular Weight, Placeholder For now
            eta_es = 3.5e-5 * f   # Tacholine Thickness Ratio
            #Constants
            C_es = 9.0 * eta_es * (const.k_boltzmann**2 / const.e_esu**4) * ((mu_es * const.m_p) / 3.0)**0.5
            #Torque
            J_shell_env_fric = C_es * T_bcz**(2.5) * R_bcz**3 * (omega_shell - omega_env)

        ### Shell-Core ###
            mu_sc = 1.6           # Mean Molecular Weight, Placeholder For now
            eta_sc = 2.5e-5 * f   # Tacholine Thickness Ratio
            #Constants
            C_sc = 9.0 * eta_sc * (const.k_boltzmann**2 / const.e_esu**4) * ((mu_sc * const.m_p) / 3.0)**0.5
            #Torque
            J_shell_core_fric = C_sc * T_brz**(2.5) * R_brz**3 * (omega_core - omega_shell)

            return J_shell_env_fric, J_shell_core_fric

    elif coupling_model == 'Turbulence':
        if model == 'two_zone_evolution':
            K = 2.0e-11

            J_fric = 4.0 * np.pi * K * rho_bcz * R_bcz**5 * (omega_core - omega_env)**2
            
            return J_fric
        else:
            K_env   = 1.0
            K_shell = 1.0
            
            J_shell_env_fric  = 4.0 * np.pi * K_env   * rho_bcz * R_bcz**5 * (omega_shell - omega_env)**2
            J_shell_core_fric = 4.0 * np.pi * K_shell * rho_brz * R_brz**5 * (omega_core - omega_shell)**2
            return J_shell_env_fric, J_shell_core_fric 

    else: 
        raise ValueError(f"Unknown coupling model: '{coupling_model}'. ")
