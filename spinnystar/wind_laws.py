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

def wind_braking(wind_model, coupling_model, use_momentum_exchange, solar_rot, wind_const,
                 moi_core, moi_env, moi_env_next, omega,
                 convec_turnover, R_0_sat, 
                 dt, L, R, M, T_eff, g, P, B):
    

    omega_crit = (2*np.pi) / (R_0_sat*convec_turnover)
    I_star = moi_core + moi_env
    R_0 = (2*np.pi) / (omega*convec_turnover)
    #solar rot is the solar calibration number, so 'eq' == equatorial (25.67) and 'avg' == average (27.28)
    
    K = wind_const
    
    if wind_model == 'Spada':                                                    
        k_w = 1

        J_dot = -k_w * K * ((I_star * convec_turnover) / (const.TOTAL_MOI_SUN * const.TAU_CONV_SUN)) * (omega)**3
        Omega_wb = J_dot * dt / moi_env_next
        return J_dot, Omega_wb

    elif wind_model == 'SimpleSat':                                                
       
        if omega_crit <= omega:
            J_dot = -K  * omega * omega_crit**2
        else:
            J_dot = -K  * omega**3
        
        Omega_wb = J_dot * dt / moi_env_next
        return J_dot, Omega_wb
    
    elif wind_model == 'vanSaders':                                                

        #Centrifugal Correction
        c = 1
        #Structure Dependence
        f_sun = (const.R_SUN**3.1) * (const.M_SUN**-0.22) * (const.L_SUN**0.56) * (const.P_SURF_SUN**0.44)
        f = -c*((R/const.R_SUN)**3.1) * ((M/const.M_SUN)**-0.22) * ((L/const.L_SUN)**0.56) * ((P/const.P_SURF_SUN)**0.44) * f_sun
        #Wind Braking
        if omega_crit <= omega * (convec_turnover/const.TAU_CONV_SUN):
            J_dot = f*K*omega*(omega_crit/const.OMEGA_SUN)**2
        else:
            J_dot = f*K*omega * ((omega*convec_turnover)/(const.OMEGA_SUN*const.TAU_CONV_SUN))**2

        Omega_wb = J_dot * dt / moi_env_next
        return J_dot, Omega_wb

    elif wind_model == 'Reiners':

        if omega >= omega_crit:
            J_dot = -K * omega * (R**16 / M**2)**(1/3)
        elif omega < omega_crit:
            J_dot = -K * (omega / omega_crit)**4 * omega * (R**16 / M**2)**(1/3)
            
        Omega_wb = J_dot * dt / moi_env_next
        return J_dot, Omega_wb

    elif wind_model == 'SpadaS2':

        f = moi_env / const.MOI_ENV_SUN

        J_dot = -K * f * (omega / const.OMEGA_SUN)**3
        Omega_wb = J_dot * dt / moi_env_next
        return J_dot, Omega_wb 
    
    elif wind_model == 'vanSadersS2':

        f = ((R/const.R_SUN)**3.1) * ((M/const.M_SUN)**-0.22) * ((L/const.L_SUN)**0.56) * ((P/const.P_SURF_SUN)**0.44)  * ((convec_turnover/const.TAU_CONV_SUN)**2)
        
        J_dot = -K * f * (omega / const.OMEGA_SUN)**3
        Omega_wb = J_dot * dt / moi_env_next
        return J_dot, Omega_wb
    
    elif wind_model == 'MattS2': 

        f = ((R/const.R_SUN)**3.1) * ((M/const.M_SUN)**0.5) * ((convec_turnover/const.TAU_CONV_SUN)**2)

        J_dot = -K * f * (omega / const.OMEGA_SUN)**3
        Omega_wb = J_dot * dt / moi_env_next
        return J_dot, Omega_wb   
    
    elif wind_model == 'Garraffo':                                                 

        J_dip = -K * omega**3 * convec_turnover
        
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
        
        if omega_crit <= omega:
            J_dip = -K  * omega * omega_crit**2 * convec_turnover
        else:
            J_dip = -K  * omega**3 * convec_turnover

        
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
        #K = (4.5e47) if solar_rot=='avg' else 1 #4.42 for solar cali
        #f = (M / const.M_SUN)**-2.0 #-- this is the power law scaling for convective overturn time!

        f = (convec_turnover / const.TAU_CONV_SUN)

        if omega_crit <= omega:
            J_dot = -K * f * omega * omega_crit**2
        else:
            J_dot = -K * f * omega**3

        Omega_wb = J_dot * dt / moi_env_next
        return J_dot, Omega_wb

    
    else:
        raise ValueError(f"Unknown wind model: '{wind_model}'. ")
