"""
plots.py - SpinnyStar plotting helper

plot_rotation_vs_age           rotation (omega/period) vs age, one star
plot_rotational_contributions  per-process contribution terms vs age, one star
plot_gyrochrones               period vs Teff/mass for a population at fixed ages
plot_reference_gyrochrones     observed benchmark cluster gyrochrones (needs gyrointerp)

All functions are decorated with @with_style and take an optional 'ax' argument to be composed in subplots 
"""

import numpy as np
from . import constants as const
import matplotlib.pyplot as plt

##################
#Internal Helpers#
##################

def plot_signed(ax, age, values, label, color):
    """ Plots dashed lines when y<0 and solid where y>0. Allows for negative values to appear on log plots """
    pos = np.where(values >= 0, values, np.nan)
    neg = np.where(values < 0, np.abs(values), np.nan)
    ax.plot(age, pos, label=label,      color=color, linestyle='-')
    ax.plot(age, neg, color=color, linestyle='--')

def with_style(func):
    """ Applies Computer Modern mpl style to plots """
    import functools
    from importlib.resources import files
    style = str(files("spinnystar").joinpath("computermodernstyle.mplstyle"))

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        plt.style.use(style)
        return func(*args, **kwargs)

    return wrapper

##################  
#External Helpers#
##################

@with_style  
def plot_rotation_vs_age(star, primary_y= 'omega', solar_marker= True, solar_rotation= 'avg', plot_saturation= False, ax= None):
    """
    Plot rotational evolution (omega / period) for a single star

    Parameters
    -------------------------------------------------------------------------------------------------------------------
    star: SpinnyStar         rotationally evolved star
    primary_y: str           rotational measure shown on left axis ('omega' / 'period')
    solar_marker: bool       mark the Sun's age (4.603e9 yrs) and rotation of choice
    solar_rotation str       solar rotation measure of choice, average [27.28 d] or equitorial [25.67 d] ('avg' / 'eq')
    plot_saturation: bool    plot a dashed line at the age the star leaves saturation
    ax: axes or None         target ax or new figure if None
    """
    import matplotlib.pyplot as plt
    
    def zones(star):
        return ['env', 'shell', 'core'] if (star.model == 'three_zone_evolution') else ['env', 'core']

    ax1 = ax if ax is not None else plt.subplots(figsize=(8, 5), layout='constrained')[1]
    ax2 = ax1.twinx()

    solar_period= {'eq': 25.67, 'avg': 27.28 }
    zone_style = {'env': ('Envelope', 'blue'), 'shell': ('Shell', 'green'), 'core': ('Core', 'red')}
    
    for zone in zones(star):
        label, color = zone_style[zone]
        ax1.loglog(star.age / 1e6, getattr(star, f'{primary_y}_{zone}'), label=label, color=color)

    period_ax = ax2 if primary_y=='omega' else ax1
    if solar_marker and solar_rotation in solar_period:
        period_ax.plot(const.AGE_SUN/1e6, solar_period[solar_rotation], 'o', markerfacecolor='none', markeredgecolor='black')
    
    if plot_saturation:
        sat = np.argmin(np.abs(star.rossby_saturation_threshold - star.rossby_number))   
        ax1.axvline(star.age[sat]/1e6, ls='--', color='black')
        
        
    axis_label = {'omega': r'Angular Frequency [rad s$^{-1}$]', 'period': 'Period [d]'}
    ax1.set_ylabel(axis_label[primary_y])
    ax2.set_yscale('log')
    other_y = 'period' if primary_y == 'omega' else 'omega'
    ax2.set_ylabel(axis_label[other_y])
    ax2.set_ylim([(2*np.pi)/(x*const.SEC_PER_DAY) for x in ax1.get_ylim()])
    ax1.set_title(f"Rotational Evolution  ({star.mass} M$_\\odot$)", fontsize=15, fontweight="bold")
    ax1.set_xlabel('Age [Myr]')
    ax1.legend()

    return ax1

@with_style  
def plot_rotational_contributions(star, contribution='Momentum', zone='env', plot_saturation=False, ax=None):
    """
    Plot of per year rotational contributions (momentum / omega) vs age for one zone at a time

    Parameters
    ---------------------------------------------------------------------------------
    star: SpinnyStar         rotationally evolved star
    contribution: str        'Momentum' (delta-J/yr) or 'Omega' (delta-Omega/yr)
    zone: str                'env', 'shell', or 'core' (shell for three zone only)
    plot_saturation: bool    plot a dashed line at the age the star leaves saturation
    ax: axes or None         target ax or new figure if None

    """
    import matplotlib.pyplot as plt

    ax = ax if ax is not None else plt.subplots(figsize=(8, 5), layout='constrained')[1]
    dt=np.diff(star.age)
    
    def contributions(star, zone):
        if zone == 'env':
            terms = {'term_env_moi_change': ('MOI Exchange', 'green'), 
                    'term_env_mass_rate': ('Mass Exchange', 'orange'), 
                    'term_wind_braking': (f'{star.wind_model}', 'red'), 
                    'term_env_momentum_exchange': (f'{star.coupling_model}', 'purple')}
    
        if zone == 'shell':
            terms = {'term_shell_moi_change': ('MOI Exchange', 'green'), 
                    'term_shell_env_mass_rate': ('Shell-Env Mass Exchange', 'gold'), 
                    'term_shell_core_mass_rate': ('Shell-Core Mass Exchange', 'darkorange'), 
                    'term_shell_env_momentum_exchange': (f'Shell-Env {star.coupling_model}', 'mediumorchid'),
                    'term_shell_core_momentum_exchange': (f'Shell-Core {star.coupling_model}', 'indigo')}
    
        if zone == 'core':
            terms = {'term_core_moi_change': ('MOI Exchange', 'green'), 
                    'term_core_mass_rate': ('Mass Exchange', 'orange'),  
                    'term_core_momentum_exchange': (f'{star.coupling_model}', 'purple')}
        
        return terms

    terms = contributions(star, zone)
    for term in terms:
        label, color = terms[term]
        data = getattr(star, term)
        moi = getattr(star, f'moi_{zone}')
        y = data/dt if (contribution=='Omega') else (data*moi[:-1]/dt)
        plot_signed(ax, star.age[:-1]/1e6, y, label=label, color=color)

    
    if plot_saturation:
        idx = np.argmin(np.abs(star.rossby_saturation_threshold-star.rossby_number))
        ax.axvline(star.age[idx]/1e6, ls='--', color= 'black')
        
    
    ax.set_title(f"{contribution} Contribution ({star.mass} M$_\\odot$)", fontsize=15, fontweight="bold")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Age [Myr]')
    labels = {'Omega':    r'$|\Delta\Omega|$ per year [rad s$^{-1}$]',
              'Momentum': r'$|\Delta J|$ per year [g cm$^2$ s$^{-1}$]',}
    ax.set_ylabel(labels[contribution])
    ax.grid()
    ax.legend(fontsize=7)
    if contribution == 'Momentum':
        ax.set_ylim(bottom=1e35)

    return ax

@with_style    
def plot_gyrochrones(stars, x_axis='T_eff', target_ages=None,
                     add_noise=False, solar_marker= False, solar_rotation='avg', plot_saturation = False,
                     marker= 'o', alpha=0.3, ax=None, format_ax=True):


    """
    Scatter of period vs Teff (or mass) for a population at fixed cluster ages
    
    Parameters
    --------------------------------------------------------------------------------------------------------------------
    stars: dict                  {mass: SpinnyStar} compatible with cluster_evolution and sequence_evolution
    x_axis: str                  'T_eff' or 'mass'
    target_ages: dict or None    {label: (age_yr, color)}, defaults if None
    
    add_noise: bool              adds a small scatter in the x_axis
    solar_marker: bool           mark the Sun's age (4.603e9 yrs) and rotation of choice
    solar_rotation str           solar rotation measure of choice, average [27.28 d] or equitorial [25.67 d] ('avg' / 'eq')
    plot_saturation: bool        plot an open circle for the stars still in saturation
    
    marker_size, alpha:          scatter styling
    ax: axes or None             target ax or new figure if None
    format_ax: bool              if plotting mulitple calls of plot_gyrochrones on one figure, the first will set the format 
    

    """

    import matplotlib.pyplot as plt

    spt_ticks = {'F5': 6550, 'G2': 5770, 'K0': 5270, 'K5': 4440, 'M0': 3850, 'M3': 3430, 'M5': 3060}

    ax1 = ax if ax is not None else plt.subplots(figsize=(12, 8))[1]

    if target_ages == None:
        target_ages = {
            '120 Myr': (1.2e8, 'blue'),
            #'300 Myr': (3.0e8, 'firebrick'),
            '650 Myr': (6.5e8, 'orange'),
            '1.0 Gyr': (1.0e9, 'green'),
            '2.6 Gyr': (2.6e9, 'red'),
            '4.0 Gyr': (4.0e9, 'purple'),
            #'6.0 Gyr': (6.0e9, 'black'),
            #'8 Gyr':   (8e9,   'yellow'),
        }
    else:
        pass

    for label, (target_age, color) in target_ages.items():
        first_label = True
        for mass, star in stars.items():
            idx = np.argmin(np.abs(star.age - target_age))
            x_raw = getattr(star, x_axis)
            
            if add_noise:
                if x_axis=='T_eff':
                    x_raw= x_raw + np.random.normal(0,scale=50,size=len(x_raw))
                if x_axis=='mass':
                    x_raw= x_raw + np.random.normal(0,scale=0.005)

            x = x_raw if np.isscalar(x_raw) else x_raw[idx]
            
            if plot_saturation and star.rossby_number[idx] <= star.rossby_saturation_threshold:
                ax1.scatter(x, star.period_env[idx],
                        facecolors='none', edgecolors= color,
                        marker=marker, alpha=alpha,
                        label=label if first_label else None)
            else:
                ax1.scatter(x, star.period_env[idx],
                        color= color,
                        marker=marker, alpha=alpha,
                        label=label if first_label else None)

            first_label = False
        
            if solar_marker:
                solar_period = {'eq': 25.67, 'avg': 27.28}
                solar_x = {'T_eff': 5772, 'mass': 1.0}
                if solar_rotation in solar_period and x_axis in solar_x:
                    ax1.plot(solar_x[x_axis], solar_period[solar_rotation], '*', markerfacecolor='black')

    if format_ax:
        if x_axis == 'T_eff':
            ax1.set_xlabel(r'$T_{\mathrm{eff}}$ [K]')
        elif x_axis == 'mass':
            ax1.set_xlabel(f"Mass [M$_\\odot$]")
        ax1.set_ylabel('Period [d]')
        
        ax1.invert_xaxis()
        ax1.grid()
        ax1.set_ylim(1,45)
        ax1.legend(fontsize=15)
        ax1.set_yticks(np.arange(0, 45, 5))
        ax1.set_xlim(6600, 3000)

    if x_axis == 'T_eff':
        ax_top = ax1.twiny()
        ax_top.set_xlim(ax1.get_xlim())
        ax_top.set_xticks(list(spt_ticks.values()))
        ax_top.set_xticklabels(spt_ticks.keys())
        
    return ax1

@with_style
def plot_reference_gyrochrones(reference_clusters=None, ax=None, format_ax=True,
                               marker_size=100, alpha=0.7):
    """
    Plot observed benchmark-cluster gyrochrones (Prot vs Teff) from the bundled gyrointerp_references

    Parameters
    ----------
    reference_clusters: list of str or None    CSV names (without '.csv') in spinnystar/gyrointerp_references/., None sets to defaults
    ax: axes or None                           target ax or new figure if None
    format_ax: bool                            if plotting mulitple calls of plot_gyrochrones on one figure, the first will set the format
    marker_size, alpha:                        scatter styling
    """
    from importlib.resources import files
    import pandas as pd

    spt_ticks = {'F5': 6550, 'G2': 5770, 'K0': 5270, 'K5': 4440,
                 'M0': 3850, 'M3': 3430, 'M5': 3060}

    if reference_clusters is None:
        reference_clusters = ['Pleiades', 'Praesepe', 'NGC6811',
                              'NGC6819', 'Ruprecht147', 'M67']

    cluster_color = {'Pleiades': 'blue', 'Praesepe': 'orange', 'NGC6811': 'green',
                     'NGC6819': 'red', 'Ruprecht147': 'red', 'M67': 'purple',
                     'Pleiades_Slow': 'blue'}

    ax1 = ax if ax is not None else plt.subplots(figsize=(12, 8))[1]
    ref_dir = files("spinnystar").joinpath("gyrointerp_references")

    for cluster in reference_clusters:
        csv = ref_dir.joinpath(f"{cluster}.csv")
        try:
            with csv.open("r", encoding="utf-8") as fh:     
                df = pd.read_csv(fh)
        except (FileNotFoundError, OSError):
            continue                                      

        color = cluster_color.get(cluster, 'gray')

        # benchmark slow-sequence stars
        bench = df[df["flag_benchmark_period"]] if "flag_benchmark_period" in df.columns else df
        teff = pd.to_numeric(bench["Teff_Curtis20"], errors="coerce")
        prot = pd.to_numeric(bench["Prot"], errors="coerce")
        ax1.scatter(teff, prot, color=color, alpha=alpha, s=marker_size,
                    label=cluster, marker='o', edgecolors='k', linewidths=0.3)

        
    if format_ax:
        ax1.set_xlabel(r'$T_{\mathrm{eff}}$ [K]')
        ax1.set_ylabel('Period [d]')
        ax1.set_xlim(6600, 3000)               
        ax1.set_ylim(1, 45)
        ax1.set_yticks(np.arange(0, 45, 5))
        ax1.grid()
        ax1.legend(fontsize=10)

        ax_top = ax1.twiny()
        ax_top.set_xlim(ax1.get_xlim())
        ax_top.set_xticks(list(spt_ticks.values()))
        ax_top.set_xticklabels(spt_ticks.keys())

    return ax1
