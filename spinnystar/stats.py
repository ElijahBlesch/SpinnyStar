"""
stats.py - helpful statistical analysis tools

Internal Helpers
------------------------------------------------------------------------------------
CLUSTER_AGES, WIND_ANCHORS,
SHAPE_ANCHORS, TAIL_ANCHORS,    chi2_staged_fit presets
SUN, TEFF_BOUNDS

snapshot                        prot and teff at target ages
sequence_interp                 model P_rot vs. Teff at a fixed age, interpolated onto observed Teffs
CHI2                            perform chi squared analysis test

External Helpers
------------------------------------------------------------------------------------
solar_wind_calibration          calibrate wind_law free parameter to the solar value
chi2_staged_fit                 fit free parameters to reference clusters

"""
import numpy as np
from . import sequence_evolution, load_reference_stars
from . import constants as const

CLUSTER_AGES = {'Pleiades_Slow': 1.20e8, 'Praesepe': 6.70e8, 'NGC6811': 1.00e9,
                '2.6 Gyr': 2.60e9, 'M67': 4.00e9}

WIND_ANCHORS  = ['2.6 Gyr', 'M67']
SHAPE_ANCHORS = ['Pleiades_Slow', 'Praesepe', 'NGC6811', '2.6 Gyr', 'M67']
TAIL_ANCHORS = ['Pleiades_Slow', 'Praesepe', 'NGC6811', '2.6 Gyr', 'M67']      

SUN = dict(teff=const.T_SUN, prot=27.28, age=const.AGE_SUN, sigma=0.5)
TEFF_BOUNDS = (4100.0, 6200.0)

##################
#Internal Helpers#
##################

def snapshot(stars, target_age):
    """return teff and prot arrays for target ages"""

    teff, prot = [],[]
    for star in stars.values():
        idx = np.argmin(np.abs(star.age-target_age))
        teff.append(star.T_eff[idx])
        prot.append(star.period_env[idx])
    return np.asarray(teff), np.asarray(prot)

def sequence_interp(stars, target_age, obs_teff, teff_bounds=TEFF_BOUNDS):
    """model P_rot vs. Teff at a fixed age, interpolated onto observed Teffs."""

    model_t, model_p = snapshot(stars, target_age)
    ok = np.isfinite(model_t) & np.isfinite(model_p)
    model_t, model_p = model_t[ok], model_p[ok]
    unique_t, u = np.unique(model_t, return_index=True)
    prot_interp = np.interp(obs_teff, unique_t, model_p[u], left=np.nan, right = np.nan)
    low, high = teff_bounds
    prot_interp[(obs_teff < low) | (obs_teff>high)] = np.nan
    return prot_interp

def CHI2(stars, refs, clusters, sigma_frac=0.10, sigma_floor=0.1, teff_bounds=TEFF_BOUNDS,
         teff_scale=None, per_cluster=True, include_sun=False):
    """chi squared analysis test"""
    
    if teff_scale is None:
        teff_scale = {'2.6 Gyr': 1.05}                 
    chi2, n = 0.0, 0
    for name in clusters:
        obs_teff, obs_prot = refs[name]
        scale = teff_scale.get(name, 1.0) if isinstance(teff_scale, dict) else teff_scale
        teff_corr = obs_teff * scale
        model = sequence_interp(stars, CLUSTER_AGES[name], teff_corr, teff_bounds)  
        m = np.isfinite(model) & np.isfinite(obs_prot)
        sigma = np.maximum(sigma_frac * obs_prot[m], sigma_floor)
        c = float(np.sum(((obs_prot[m] - model[m]) / sigma) ** 2))
        k = int(m.sum())
        if per_cluster:
            chi2 += c / max(k, 1); n += 1              
        else:
            chi2 += c; n += k                   
    if include_sun:
        model = sequence_interp(stars, SUN['age'], np.array([SUN['teff']]), teff_bounds)
        if np.isfinite(model[0]):
            chi2 += ((SUN['prot'] - model[0]) / SUN['sigma']) ** 2
            n += 1
    return chi2, n

##################  
#External Helpers#
##################

def solar_wind_calibration(data_dir, k_0, solar_rotation,
                           initial_period, wind_model, coupling_model, disk_locking_model,
                           model='two_zone_evolution', max_iter=50, tolerance=1e-3):
    """
    calibrate wind constant to align with the solar value

    Parameters
    -------------------------------------------------------------------------------------------------------------------------------------------
    data_dir: str                      data directory path containing dtrk files
    k_0: float                         wind constant initial guess
    solar_rotation: str                solar rotation measure of choice, average [27.28 d] or equitorial [25.67 d] ('avg' / 'eq') 

    initial_period: float              starting rotation period [days]
    wind_model: str                    wind model of choice (models found in wind_laws.py) 
    coupling_model: str                coupling model of choice (models found in coupling_laws.py)
    disk_locking_model: float / str    disk Locking model of choice (models found in disk_locking_laws.py) or float for constant lifetime [Myr]

    model: str                         'two_zone_evolution' or 'three_zone_evolution'
    max_iter: float                    maximum number of iterations for convergence
    tolerance: float                   tolerance to solar value

    Returns
    -------------------------------------------------------------------------------------------------------------------------------------------
    k, prot_star                       return calibrated wind constant, period of the test star
    """
    

    from . import SpinnyStar

    def star_setup( data_dir, wind_const,
                    initial_period, wind_model, coupling_model, disk_locking_model, 
                    model='two_zone_evolution'):

        star = SpinnyStar(mass=1.0, data_dir=data_dir)
        if model == 'two_zone_evolution':
            star.two_zone_evolution(initial_period=initial_period, 
                                    wind_model=wind_model, coupling_model=coupling_model, disk_locking_model=disk_locking_model,
                                    wind_const=wind_const)
        elif model == 'three_zone_evolution':
            star.three_zone_evolution(initial_period=initial_period, 
                                        wind_model=wind_model, coupling_model=coupling_model, disk_locking_model=disk_locking_model,
                                        wind_const=wind_const)  
        else:
            raise ValueError(f"Unknown model '{model}'")

        prot_star = np.interp(const.AGE_SUN, star.age, star.period_env)

        return prot_star, wind_const

    if solar_rotation == 'avg':
            prot_sun = 27.28
    elif solar_rotation == 'eq':
            prot_sun = 25.67
    else:
        raise ValueError(f"Unknown solar rotation preset: {solar_rotation}")
    
    k=k_0
    for i in range(max_iter):
        prot_star, k, = star_setup(data_dir, k,
                    initial_period, wind_model, coupling_model, disk_locking_model, model)

        err = (prot_star-prot_sun)/prot_sun
        if abs(err) <= tolerance:
             return k, prot_star

        k *= (prot_sun / prot_star) ** 2 

    raise RuntimeError(f"Solar calibration did not converge in {max_iter} iterations "
                   f"(P = {prot_star:.3f} d, target {prot_sun:.3f} d, K = {k:.4e})")

def chi2_staged_fit(data_dir, init=(4.0e47, 3.5e-5, 5.2),
               wind_bounds=(1e47, 1e48), cc_bounds=(1e-6, 1e-3), cp_bounds=(0.0, 8.0),
               tail_bounds=(3800, 6000), rsat_bounds=(0.08,0.15),
               sigma_frac=0.10, initial_period=8.0, disk_locking_model=5.0,
               max_workers=None, polish=True, verbalize=True):

    """
    fit wind_const, coupling_const, coupling_power, and rossby_saturation_threshold to align utils.sequence_evolution() to reference clusters

    Parameters
    -------------------------------------------------------------------------------------------------------------------------------------------
    data_dir: str                      data directory path containing dtrk files
    init: tuple                        initial (wind_const, coupling_const, and coupling_power)
                                       NOTE: rossby_saturation_threshold already defaults to 0.1

    *_bounds: tuple                    (lower, upper) bounds for parameters

    sigma_frac: float                  set model tolerance
    initial_period: float              starting rotation period [days]
    disk_locking_model: float / str    disk Locking model of choice (models found in disk_locking_laws.py) or float for constant lifetime [Myr]

    max_workers: int / None            multiprocessing workers, default: max_cpu_count - 2
    polish: bool                       adjust all parameters together as a final step
    verbalize: bool                    print a report of iterations

    Returns
    -------------------------------------------------------------------------------------------------------------------------------------------
    return fitted parameters, chi2, number of points, reduced chi2

    NOTE: The 2.6 Gyr reference clusters are adjusted by 5% for observational error
    NOTE: wind_model is hardcoded to 'test' for now
    """
    
    from scipy.optimize import minimize_scalar, minimize, differential_evolution

    refs = load_reference_stars()
    initial_wind_const, initial_coupling_const, initial_coupling_power = init

    run = lambda wc, cc, cp, rsat=0.1: sequence_evolution(
        data_dir,
        model='two_zone_evolution',          
        initial_period=initial_period,
        wind_model='test',
        coupling_model='Frictional_Torque',
        disk_locking_model=disk_locking_model,
        wind_const=wc, coupling_const=cc, coupling_power=cp, rossby_saturation_threshold=rsat,
        max_workers=max_workers)
    
    
    def wind_stage(log_wc):
        c2, _ = CHI2(run(10**log_wc, initial_coupling_const, initial_coupling_power), refs, WIND_ANCHORS,
                      sigma_frac=sigma_frac, include_sun=True)
        if verbalize: print(f"[A] wind_const={10**log_wc:.3e}  chi2={c2:.2f}")
        return c2

    A = minimize_scalar(wind_stage, method='bounded', options={'xatol': 0.02},
                         bounds=(np.log10(wind_bounds[0]), np.log10(wind_bounds[1])))

    wc_fit = 10**A.x

    def coupling_stage(x):
        log_cc, cp = x
        c2, _ = CHI2(run(wc_fit, 10**log_cc, cp), refs, SHAPE_ANCHORS, sigma_frac=sigma_frac)
        if verbalize: print(f"[B] coupling_const={10**log_cc:.3e} coupling_power={cp:.2f}  chi2={c2:.2f}")
        return c2
    
    B = minimize(coupling_stage, x0=[np.log10(initial_coupling_const), initial_coupling_power], method='Nelder-Mead',
                  bounds=[(np.log10(cc_bounds[0]), np.log10(cc_bounds[1])), cp_bounds], options={'xatol': 0.05, 'fatol': 0.5, 'maxiter': 60})
    cc_fit, cp_fit = 10**B.x[0], B.x[1]

    def rsat_stage(rsat):
        c2, _ = CHI2(run(wc_fit, cc_fit, cp_fit, rsat), refs, TAIL_ANCHORS,
                     sigma_frac=sigma_frac, teff_bounds=tail_bounds)   
        if verbalize: print(f"[D] rsat={rsat:.4f}  chi2={c2:.2f}")
        return c2
    D = minimize_scalar(rsat_stage, method='bounded', options={'xatol': 0.002},
                        bounds=rsat_bounds)
    rsat_fit = D.x

    # ---- Stage C: 4-parameter polish ----
    all_clusters = ['Pleiades_Slow', 'Praesepe', 'NGC6811', '2.6 Gyr', 'M67']
    if polish:
        span = 0.3
        bounds = [(np.log10(wc_fit)-span, np.log10(wc_fit)+span),
                  (np.log10(cc_fit)-span, np.log10(cc_fit)+span),
                  (max(cp_bounds[0], cp_fit-1.0), min(cp_bounds[1], cp_fit+1.0)),
                  rsat_bounds]
        def polish_fits(x):
            log_wc, log_cc, cp, rsat = x
            c2, _ = CHI2(run(10**log_wc, 10**log_cc, cp, rsat), refs, all_clusters,
                         sigma_frac=sigma_frac, include_sun=True)
            if verbalize: print(f"[C] wc={10**log_wc:.3e} cc={10**log_cc:.3e} cp={cp:.2f} rsat={rsat:.3f} chi2={c2:.2f}")
            return c2
        C = differential_evolution(polish_fits, bounds, maxiter=6, popsize=4, tol=1e-2, seed=0, polish=False)
        wc_fit, cc_fit, cp_fit, rsat_fit = 10**C.x[0], 10**C.x[1], C.x[2], C.x[3]

    # ---- final report ----
    chi2, n = CHI2(run(wc_fit, cc_fit, cp_fit, rsat_fit), refs, all_clusters,
                   sigma_frac=sigma_frac, per_cluster=False, include_sun=True)
    dof = max(n - 4, 1)                           
    return {'wind_const': wc_fit, 'coupling_const': cc_fit, 'coupling_power': cp_fit,
            'rossby_saturation_threshold': rsat_fit,
            'chi2': chi2, 'n_points': n, 'chi2_reduced': chi2 / dof}