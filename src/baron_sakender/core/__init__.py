"""
Baron-Sakender Core Module.

2D Ideal MHD Solver with JAX acceleration.

Components:
    - MHDSystem: Physical system definition and initial conditions
    - MHDIntegrator: Time integration with HLL Riemann solver
    - metrics: Comprehensive diagnostic metrics

Example:
    >>> from baron_sakender.core import MHDSystem, MHDIntegrator
    >>> system = MHDSystem(nx=256, ny=256)
    >>> system.init_orszag_tang()
    >>> integrator = MHDIntegrator(system, cfl=0.4)
    >>> integrator.evolve(t_end=1.0)
"""

from .mhd_system import MHDSystem, MHDParams
from .integrator import MHDIntegrator, MHDSolver
from . import metrics

__all__ = [
    'MHDSystem',
    'MHDParams', 
    'MHDIntegrator',
    'MHDSolver',  # Backward compatibility
    'metrics',
]
