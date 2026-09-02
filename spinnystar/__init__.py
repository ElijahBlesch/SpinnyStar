""" spinnystar: a stellar angular momentum evolution model assuming solid-body rotation """

from .star import SpinnyStar

from .utils import (
    StarPresets,
    EvolutionPresets,
    sequence_evolution,
    cluster_evolution,
    load_reference_stars,
    save_rtrk,
)

from .plots import (
    plot_rotation_vs_age,
    plot_rotational_contributions,
    plot_gyrochrones,
    plot_reference_gyrochrones,
)

__version__ = "0.1.0"

__all__ = [
    "SpinnyStar",
    "StarPresets", 
    "EvolutionPresets",
    "sequence_evolution",
    "cluster_evolution",
    "load_reference_stars",
    "plot_rotation_vs_age",
    "plot_rotational_contributions",
    "plot_gyrochrones",
    "plot_reference_gyrochrones",
    "save_rtrk",
]
