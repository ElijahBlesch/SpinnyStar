"""
utils.py - helpers for building, evolving, saving, and loading reference data

Internal Utilities
-------------------------------------------------------------------------------
StarPresets / EvolutionPresets    default kwargs for SpinnyStar stages
prepare_interp / fast_interp      internal interpolation helpers
evolve_mass_batch                 evolve batch of stars, multiprocessing helper

External Utilities
-------------------------------------------------------------------------------
sequence_evolution                evolve a fixed mass grid
cluster_evolution                 evolve a synthetic cluster
load_reference_stars              load reference clusters from gyrointerp
save_rtrk                         save SpinnyStar outputs to rtrk files

"""
import numpy as np
from importlib.resources import files

StarPresets = dict(min_log_moi= -10, interpolate=True)

EvolutionPresets = dict(rossby_saturation_threshold= 0.1, solar_rotation= 'avg',
                        use_mass_rate=True, use_momentum_exchange=True, use_moi_change=True, use_wind_braking=True,
                        coupling_data_path=None)

##################
#Internal Helpers#
##################

def prepare_interp(t_old):
    """
    Sorts time axis for interpolation
    Returns (t_sorted, idx, t_finite): ascending times, sort order, finite mask

    NOTE: This was needed in previous iterations. It is still in the model, but it is not a requirement anymore.
    """
    t = np.asarray(t_old)
    idx = np.argsort(t)
    t_sorted = t[idx]
    t_finite = np.isfinite(t_sorted)
    return t_sorted, idx, t_finite

def fast_interp(t_new, t_sorted, idx, t_finite, y_old):
    """ Interpolate y_old onto t_new using outputs from the prepare_interp helper """

    y = np.asarray(y_old)[idx]
    mask = t_finite & np.isfinite(y)
    return np.interp(t_new, t_sorted[mask], y[mask])

def evolve_mass_batch(inputs):
    """
    Multiprocessing tool. Builds one SpinnyStar per mass and evolves per set of inputs

    inputs : tuple    (mass, data_dir, star_inputs, model, evolution_inputs_list, indices)
    """
    from types import SimpleNamespace
    from .star import SpinnyStar

    mass, data_dir, star_inputs, model, evolution_inputs, indices = inputs
    star = SpinnyStar(mass=mass, data_dir=data_dir, **star_inputs)
    evolve = getattr(star, model)

    batch = []
    for i, evolution_inputs in zip(indices, evolution_inputs):
        evolve(**evolution_inputs)
        results = SimpleNamespace(mass=mass, age=star.age,
                                   period_env=star.period_env, period_core=star.period_core,
                                   period_shell=getattr(star, 'period_shell', None),
                                   T_eff=star.T_eff)
        batch.append((i, results))
    return batch

##################  
#External Helpers#
##################

def sequence_evolution(data_dir, model='two_zone_evolution', max_workers=None,
                       spinnystar_inputs=None, **evolution_inputs):
    """
    Evolve one star per mass in parallel

    Parameters
    ----------------------------------------------------------------------------------------------------------------------------
    data_dir: str               data directory path containing dtrk files
    model: str                  evolution method (two_zone_evolution / three_zone_evolution)
    max_workers: int or None    multiprocessing workers, default: max_cpu_count - 2
    spinnystar_inputs: dict     overrides default StarPresets
    **evolution_inputs         initial_period, wind_model, coupling_model, disk_locking_model and overrides to EvolutionPresets

    Returns
    ----------------------------------------------------------------------------------------------------------------------------
    dict:    {grid_index: SimpleNamespace(mass, age, period_env, period_core, period_shell, T_eff)}

    """

    import multiprocessing as mp

    spinnystar_inputs = StarPresets | (spinnystar_inputs or {})
    evolution_inputs = EvolutionPresets | evolution_inputs

    if max_workers is None:
        max_workers = max(1, mp.cpu_count()-2)

    masses = np.append(np.round(np.arange(0.2, 1.01, 0.02), 2), np.round(np.arange(1.05, 1.36, 0.05), 2))
    inputs = [(mass, data_dir, spinnystar_inputs, model, [evolution_inputs], [i])
              for i, mass in enumerate(masses)]
    
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=max_workers) as pool:
        batches = pool.map(evolve_mass_batch, inputs)

    return dict(r for batch in batches for r in batch)

def cluster_evolution(data_dir, num_stars=300, model='two_zone_evolution', max_workers=None,
                       spinnystar_inputs=None, **evolution_inputs):
    """
    Evolve synthetic cluster with random mass, initial_period, disk_locking, and rossby_saturation_threshold.
    Builds one SpinnyStar per mass and reuses across evolutions

    Parameters
    ----------------------------------------------------------------------------------------------------------------------------
    data_dir: str               data directory path containing dtrk files
    num_stars: int              number of stars in cluster
    model: str                  evolution method (two_zone_evolution / three_zone_evolution)
    max_workers: int or None    multiprocessing workers, default: max_cpu_count - 2
    spinnystar_inputs: dict     overrides default StarPresets
    **evolution_inputs         initial_period, wind_model, coupling_model, disk_locking_model and overrides to EvolutionPresets

    Returns
    ----------------------------------------------------------------------------------------------------------------------------
    dict:    {grid_index: SimpleNamespace(mass, age, period_env, period_core, period_shell, T_eff)}
    
    """
    import multiprocessing as mp

    spinnystar_inputs = StarPresets | (spinnystar_inputs or {})
    evolution_inputs = EvolutionPresets | evolution_inputs

    if max_workers is None:
        max_workers = max(1, mp.cpu_count()-2)

    mass_grid = np.append(np.round(np.arange(0.3, 1.01, 0.02), 2), np.round(np.arange(1.05, 1.41, 0.05), 2))
    masses = np.random.uniform(0.2, 1.2, size=num_stars)
    masses = np.array([mass_grid[np.argmin(np.abs(mass_grid - m))] for m in masses])
    initial_periods = np.random.uniform(4,10, size=num_stars)
    #initial_periods = np.random.normal(loc=7, scale=1.5, size=num_stars)
    disk_locking_model = evolution_inputs.get('disk_locking_model', None)
    if disk_locking_model is None:
        disk_locking = np.random.uniform(1, 7, size=num_stars) 
    else:
        disk_locking = [disk_locking_model] * num_stars

    rossby_saturation_threshold = np.random.normal(loc=0.13, scale=0.025, size=num_stars)

    tasks = {}
    for i, (m, p, d, Ro) in enumerate(zip(masses, initial_periods, disk_locking, rossby_saturation_threshold)):
        kwargs = {**evolution_inputs, "initial_period": p, "disk_locking_model": d, "rossby_saturation_threshold": Ro}
        indices, kwargs_list = tasks.setdefault(m, ([], []))
        indices.append(i)
        kwargs_list.append(kwargs)

    inputs = [(mass, data_dir, spinnystar_inputs, model, kwargs_list, indices)
              for mass, (indices, kwargs_list) in tasks.items()]

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=max_workers) as pool:
        batches = pool.map(evolve_mass_batch, inputs)

    return dict(r for batch in batches for r in batch)

def load_reference_stars():
    """
    Loads bundled gyrointerp benchmark clusters as {cluster: (Teff, Prot)}

    Reads every CSV in spinnystar/gyrointerp_references/, keeps only
    flag_benchmark_period rows, and sorts by Teff. The '2.6 Gyr' entry is built
    by combining Ruprecht147 + NGC6819.
    """

    import pandas as pd

    ref_dir = files("spinnystar").joinpath("gyrointerp_references")

    out = {}
    for entry in ref_dir.iterdir():
        cluster = entry.name[:-4]

        with entry.open("r", encoding="utf-8") as f:
                df = pd.read_csv(f)

        if "flag_benchmark_period" in df.columns:
                df = df[df["flag_benchmark_period"]]        

        teff = pd.to_numeric(df["Teff_Curtis20"], errors="coerce").to_numpy()
        prot = pd.to_numeric(df["Prot"], errors="coerce").to_numpy()

        order= np.argsort(teff)
        out[cluster]= (teff[order], prot[order])

    Rup147 = out['Ruprecht147']
    NGC6819 = out['NGC6819']

    out['2.6 Gyr'] = [np.concatenate((a, b)) for a,b in zip(Rup147, NGC6819)]

    return out

def save_rtrk(star, save_path=None):
        """

        Write SpinnyStar outputs to a white space delimited .rtrk file

        Parameters
        ---------------------------------------------------------------------------
        star: SpinnyStar          a rotationally evolved star
        save_path: str or None    Output directory, defaults to dtrk data directory

        Returns
        ---------------------------------------------------------------------------
        str    filepath rtrk was saved to

        """        
        import os
        N = len(star.age) -1

        rtrk_variables = [
        ("age_yr", "age"),

        ("omega_env", "omega_env"), ("omega_shell", "omega_shell"), ("omega_core", "omega_core"),
        ("period_env", "period_env"), ("period_shell", "period_shell"), ("period_core", "period_core"),
        ("delta_omega_env", "delta_omega_env"), ("delta_omega_shell", "delta_omega_shell"), ("delta_omega_core", "delta_omega_core"),
        
        ("mass_env", "mass_env"), ("mass_shell", "mass_shell"), ("mass_core", "mass_core"),
        ("radius_env", "radius_env"), ("radius_shell", "radius_shell"),  ("radius_core", "radius_core"),
        ("moi_env", "moi_env"), ("moi_shell", "moi_shell"), ("moi_core", "moi_core"),

        ("T_eff", "T_eff"), ("L", "L"), ("P_surf", "P"), ("B_surf", "B"),
        ("convec_turnover", "convec_turnover"), ("rossby_number", "rossby_number"),

        ("term_wind_braking", "term_wind_braking"),
        ("term_env_mass_rate", "term_env_mass_rate"),
        ("term_env_moi_change", "term_env_moi_change"),
        ("term_env_momentum_exchange", "term_env_momentum_exchange"),
        ("term_shell_env_mass_rate", "term_shell_env_mass_rate"),
        ("term_shell_core_mass_rate", "term_shell_core_mass_rate"),
        ("term_shell_moi_change", "term_shell_moi_change"),
        ("term_shell_env_momentum_exchange", "term_shell_env_momentum_exchange"),
        ("term_shell_core_momentum_exchange", "term_shell_core_momentum_exchange"),
        ("term_core_mass_rate", "term_core_mass_rate"),
        ("term_core_moi_change", "term_core_moi_change"),
        ("term_core_momentum_exchange", "term_core_momentum_exchange"),
        ]

        names, cols = [], []
        for col_name, attribute in rtrk_variables:
            if not hasattr(star, attribute):
                continue
            array = np.asarray(getattr(star, attribute))
            if array.ndim != 1:
                continue
            if len(array) == N:
                cols.append(array)
            elif len(array) == N+1:
                cols.append(array[:-1])
            else:
                continue
            names.append(col_name)

        out = np.column_stack(cols)

        if save_path is None:
             save_path = star.data_dir
        os.makedirs(save_path, exist_ok=True)
            
        filename = f"m{int(star.mass * 1000):04d}_{star.wind_model}_{star.coupling_model}_P{int(star.initial_period / 86_400):02d}.rtrk"
        filepath = os.path.join(save_path, filename)

        header = [
                  "#" * 80,
                  "# SpinnyStar output and structure data",
                  f"# model:             {getattr(star, 'model', 'unknown')}",
                  f"# mass:              {star.mass} solar masses ",
                  f"# wind_model:        {star.wind_model}",
                  f"# coupling_model:    {star.coupling_model}",
                  f"# initial_period:    {star.initial_period}",
                  "#" * 80,
                ]

        col_line = "  ".join(f"{c:>22}" for c in names)

        with open(filepath, "w") as f:
            f.write("\n".join(header) + "\n")
            f.write(col_line + "\n")
            np.savetxt(f, out, fmt="%22.8E", delimiter="  ")

        return filepath
