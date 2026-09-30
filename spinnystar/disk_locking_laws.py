"""
disk_locking_laws.py -- apply the disk locking lifetime of choice

Models
--------------------------------------------------------------------------------------------------------
int:              if an int is inputed to disk_locking_model it will set the disk locking to that number
'Bertout':        Bertout 2016
'exponential':    random lifetime

Returns
--------------------------------------------------------------------------------------------------------
tau:    disk_locking_model [Myr]
"""

import numpy as np

def disk_locking(disk_locking_model, mass_frac):

    #Constant Value in Myr
    if isinstance(disk_locking_model, (int, float, np.integer, np.floating)):    
        return disk_locking_model * 1e6
    
    #Mass Dependent Model Based on Taurus-Auriga T Association
    if disk_locking_model == 'Bertout':
        tau = 4*(mass_frac)**0.75
        return tau*1e6

    elif disk_locking_model == 'exponential':
        tau = np.random.exponential(scale=5.0)
        return tau
    
    else:
        raise ValueError(f"Unknown disk locking model: '{disk_locking_model}'. ")
