"""Pytest configuration and fixtures for baron-sakender tests."""

import pytest
import numpy as np


@pytest.fixture(scope="session")
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def default_params():
    """Default MHD parameters."""
    return {
        'gamma': 5.0 / 3.0,
        'nx': 64,
        'ny': 64,
        'Lx': 2 * np.pi,
        'Ly': 2 * np.pi,
        'rho_0': 1.0e-12,
        'B_0': 1.0e-9,
        'L_0': 1.0e6,
    }


@pytest.fixture
def default_simulation_params():
    """Default simulation parameters."""
    return {
        't_end': 0.5,
        'save_dt': 0.1,
        'cfl': 0.4,
    }


@pytest.fixture
def small_system():
    """Create a small MHD system for quick tests."""
    from baron_sakender import MHDSystem
    
    system = MHDSystem(nx=32, ny=32)
    system.init_orszag_tang()
    return system


@pytest.fixture
def medium_system():
    """Create a medium MHD system for standard tests."""
    from baron_sakender import MHDSystem
    
    system = MHDSystem(nx=64, ny=64)
    system.init_orszag_tang()
    return system
