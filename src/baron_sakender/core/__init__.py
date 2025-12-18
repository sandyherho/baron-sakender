"""Core solver components for 2D Ideal MHD analysis."""

from .mhd_system import MHDSystem, MHDParams
from .integrator import MHDSolver
from .metrics import (
    compute_conservation_metrics,
    compute_stability_metrics,
    compute_turbulence_metrics,
    compute_information_metrics,
    compute_composite_metrics,
    compute_all_metrics,
    compute_diagnostics,
    compute_energy_spectrum,
    compute_structure_functions,
)

__all__ = [
    "MHDSystem",
    "MHDParams",
    "MHDSolver",
    "compute_conservation_metrics",
    "compute_stability_metrics",
    "compute_turbulence_metrics",
    "compute_information_metrics",
    "compute_composite_metrics",
    "compute_all_metrics",
    "compute_diagnostics",
    "compute_energy_spectrum",
    "compute_structure_functions",
]
