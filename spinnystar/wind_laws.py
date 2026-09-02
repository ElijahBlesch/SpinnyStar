"""
wind_laws.py -- apply the angular momentum loss formulation of choice

Models
---------------------------------------------------------------------------------------------------------
'Spada':          Spada and Lanzafame 2020 
'SimpleSat':     'Spada' model modified with saturation
'vanSaders':      van Saders and Pinsonneault 2013
'Reiners':        Reiners and Mohanty 2012
'SpadaS2':        Spada and Lanzafame 2025 [S2]
'vanSadersS2':    S2 simplification of van Saders and Pinsonneault 2013
'MattS2':         S2 simplification of Matt 2015
'Garraffo':       Garraffo et. al. 2018
'GarraffoSat':    'Garraffo' model modified with saturation

'test':           play model. calibrated against frictional_torque coupling

Returns
---------------------------------------------------------------------------------------------------------
J_dot:       wind torque [g cm^2 / s^2]
Omega_wb:    the per step omega contribution [rad_s]

NOTE: The free paramaters in each model are uncalibrated or outdated. Calibrate against the Sun or reference gyrochrones 

"""

import numpy as np
import math
from . import constants as const

def wind_braking(wind_model, coupling_model, use_momentum_exchange, solar_rot,
                 moi_core, moi_env, moi_env_next, omega,
                 convec_turnover, R_0_sat, 
                 dt, L, R, M, T_eff, g, P, B):
    

    omega_crit = (2*np.pi) / (R_0_sat*convec_turnover)
    I_star = moi_core + moi_env
    R_0 = (2*np.pi) / (omega*convec_turnover)
    #solar rot is the solar calibration number, so 'eq' == equatorial (25.67) and 'avg' == average (27.28)
    
    if wind_model == 'Spada':                                                    
        k_w = 1
        k_0 = 1.81e47  # g cm^2 s

        J_dot = -k_w * k_0 * ((I_star * convec_turnover) / (const.TOTAL_MOI_SUN * const.TAU_CONV_SUN)) * (omega)**3
        Omega_wb = J_dot * dt / moi_env_next
        return J_dot, Omega_wb

    elif wind_model == 'SimpleSat':                                                
        K = 4.42e47
       
        if omega_crit <= omega:
            J_dot = -K  * omega * omega_crit**2
        else:
            J_dot = -K  * omega**3
        
        Omega_wb = J_dot * dt / moi_env_next
        return J_dot, Omega_wb
    
    elif wind_model == 'vanSaders':                                                
        #Calibration Constant
        if use_momentum_exchange == True:
            if solar_rot == 'eq':
                f = 1.51 * 1e-11
            if solar_rot == 'avg':
                f = 1.72 * 1e-11
        elif use_momentum_exchange == False:
            if solar_rot == 'eq':
                f = 1.48 * 1e-12
            if solar_rot == 'avg':
                f = 1.67 * 1e-12
        #Centrifugal Correction
        c = 1
        #Structure Dependence
        K_sun = (const.R_SUN**3.1) * (const.M_SUN**-0.22) * (const.L_SUN**0.56) * (const.P_SURF_SUN**0.44)
        K = -c*((R/const.R_SUN)**3.1) * ((M/const.M_SUN)**-0.22) * ((L/const.L_SUN)**0.56) * ((P/const.P_SURF_SUN)**0.44) * K_sun
        #Wind Braking
        if omega_crit <= omega * (convec_turnover/const.TAU_CONV_SUN):
            J_dot = f*K*omega*(omega_crit/const.OMEGA_SUN)**2
        else:
            J_dot = f*K*omega * ((omega*convec_turnover)/(const.OMEGA_SUN*const.TAU_CONV_SUN))**2

        Omega_wb = J_dot * dt / moi_env_next
        return J_dot, Omega_wb

    elif wind_model == 'Reiners':
        #Calibration Constant (2/3 [(B_crit**8) / (G**2 * K**4 * M_dot)]**1/3)
        C = 4.89 * 1e4
        if omega >= omega_crit:
            J_dot = -C * omega * (R**16 / M**2)**(1/3)
        elif omega < omega_crit:
            J_dot = -C * (omega / omega_crit)**4 * omega * (R**16 / M**2)**(1/3)
            
        Omega_wb = J_dot * dt / moi_env_next
        return J_dot, Omega_wb

    elif wind_model == 'SpadaS2':
        K = .813 * 1E31
        f = moi_env / const.MOI_ENV_SUN

        J_dot = -K * f * (omega / const.OMEGA_SUN)**3
        Omega_wb = J_dot * dt / moi_env_next
        return J_dot, Omega_wb 
    
    elif wind_model == 'vanSadersS2':
        K = 0.91 * 1E31
        f = ((R/const.R_SUN)**3.1) * ((M/const.M_SUN)**-0.22) * ((L/const.L_SUN)**0.56) * ((P/const.P_SURF_SUN)**0.44)  * ((convec_turnover/const.TAU_CONV_SUN)**2)
        
        J_dot = -K * f * (omega / const.OMEGA_SUN)**3
        Omega_wb = J_dot * dt / moi_env_next
        return J_dot, Omega_wb
    
    elif wind_model == 'MattS2': 
        K = .877 * 1E31
        f = ((R/const.R_SUN)**3.1) * ((M/const.M_SUN)**0.5) * ((convec_turnover/const.TAU_CONV_SUN)**2)

        J_dot = -K * f * (omega / const.OMEGA_SUN)**3
        Omega_wb = J_dot * dt / moi_env_next
        return J_dot, Omega_wb   
    
    elif wind_model == 'Garraffo':                                                 
        if use_momentum_exchange:
            if solar_rot == 'eq':
                c = 1.45 * 1e42
            if solar_rot == 'avg':
                c = 1.89 * 1e42
        else:
            if solar_rot == 'eq':
                c = 1.07 * 1e41
            if solar_rot == 'avg':
                c = 1.37 * 1e41

        J_dip = -c * omega**3 * convec_turnover
        
        a = 0.02
        b = 2 
        n = a/R_0 + 1 + b*R_0
        
        if n > 7:
            n = 7
            Q = 4.05 * math.exp(-1.4*n)
        else:
            Q = 4.05 * math.exp(-1.4*n)
        
        J_dot = J_dip * Q
        Omega_wb = J_dot * dt / moi_env_next
        return J_dot, Omega_wb
    
    elif wind_model == 'GarraffoSat':                                              
        if use_momentum_exchange:
            if solar_rot == 'eq':
                c = 1.47 * 1e42
            if solar_rot == 'avg':
                c = 1.91 * 1e42
        else:
            if solar_rot == 'eq':
                c = 1.08 * 1e41
            if solar_rot == 'avg':
                c = 1.37 * 1e41
        
        if omega_crit <= omega:
            J_dip = -c  * omega * omega_crit**2 * convec_turnover
        else:
            J_dip = -c  * omega**3 * convec_turnover

        
        a = 0.02
        b = 2 
        n = a/R_0 + 1 + b*R_0
        if n > 7:
            n = 7
            Q = 4.05 * math.exp(-1.4*n)
        else:
            Q = 4.05 * math.exp(-1.4*n)
        
        J_dot =  J_dip * Q
        Omega_wb = J_dot * dt / moi_env_next
        return J_dot, Omega_wb
    
    elif wind_model == 'test':
        K = (4.5e47) if solar_rot=='avg' else 1 #4.42 for solar cali

        f = (convec_turnover / const.TAU_CONV_SUN)

        if omega_crit <= omega:
            J_dot = -K * f * omega * omega_crit**2
        else:
            J_dot = -K * f * omega**3

        Omega_wb = J_dot * dt / moi_env_next
        return J_dot, Omega_wb

    
    else:
        raise ValueError(f"Unknown wind model: '{wind_model}'. ")
