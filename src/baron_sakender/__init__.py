"""
baron-sakender: JAX-Accelerated 2D Ideal MHD Solver

A high-performance Python library for simulating and analyzing
the Orszag-Tang vortex and other 2D MHD configurations with
comprehensive turbulence and conservation metrics.

The ideal MHD equations in conservative form:
    ∂ρ/∂t + ∇·(ρv) = 0
    ∂(ρv)/∂t + ∇·(ρvv + P*I - BB) = 0
    ∂B/∂t - ∇×(v×B) = 0
    ∂E/∂t + ∇·((E+P*)v - B(v·B)) = 0

where P* = p + B²/2 is the total pressure.

Features:
    - JAX GPU/CPU acceleration for fast integration
    - HLL Riemann solver for shock capturing
    - Comprehensive conservation metrics
    - Standard and novel MHD turbulence diagnostics
    - Information-theoretic analysis tools
    - Beautiful dark-themed visualizations
    - Multiple output formats (CSV, NetCDF, PNG, GIF)


Authors: Sandy H. S. Herho, Nurjanna J. Trilaksono
License: MIT
"""

__version__ = "0.0.1"
__author__ = "Sandy H. S. Herho, Nurjanna J. Trilaksono"
__email__ = "sandy.herho@email.ucr.edu"
__license__ = "MIT"

from .core.mhd_system import MHDSystem, MHDParams
from .core.integrator import MHDSolver
from .core.metrics import (
    compute_conservation_metrics,
    compute_stability_metrics,
    compute_turbulence_metrics,
    compute_information_metrics,
    compute_composite_metrics,
    compute_all_metrics,
    compute_diagnostics,
)
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler

__all__ = [
    # Core classes
    "MHDSystem",
    "MHDParams",
    "MHDSolver",
    # Config and data
    "ConfigManager",
    "DataHandler",
    # Metrics functions
    "compute_conservation_metrics",
    "compute_stability_metrics",
    "compute_turbulence_metrics",
    "compute_information_metrics",
    "compute_composite_metrics",
    "compute_all_metrics",
    "compute_diagnostics",
]
