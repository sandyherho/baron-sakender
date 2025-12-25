"""
Simplified Metrics for 2D Ideal MHD Analysis.

Provides essential, robust metrics for quantifying MHD dynamics:
    - Conservation metrics (mass, momentum, energy, magnetic flux)
    - Stability metrics (CFL, Mach numbers, plasma beta)
    - Turbulence metrics (enstrophy, current, cross helicity)

References:
    Biskamp, D. (2003). Magnetohydrodynamic Turbulence.
    Frisch, U. (1995). Turbulence: The Legacy of A. N. Kolmogorov.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any


def compute_diagnostics(
    U: np.ndarray,
    dx: float,
    dy: float,
    gamma: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute basic diagnostic fields from conservative variables.
    
    Args:
        U: Conservative variables [7, nx, ny]
        dx, dy: Grid spacing
        gamma: Adiabatic index
    
    Returns:
        Tuple of (density, magnetic_pressure, current_density_z, vorticity_z)
    """
    rho = U[0]
    vx = U[1] / rho
    vy = U[2] / rho
    Bx = U[3]
    By = U[4]
    
    # Magnetic pressure
    pmag = 0.5 * (Bx**2 + By**2)
    
    # Current density Jz = ∂By/∂x - ∂Bx/∂y
    dBy_dx = (np.roll(By, -1, axis=0) - np.roll(By, 1, axis=0)) / (2 * dx)
    dBx_dy = (np.roll(Bx, -1, axis=1) - np.roll(Bx, 1, axis=1)) / (2 * dy)
    Jz = dBy_dx - dBx_dy
    
    # Vorticity ωz = ∂vy/∂x - ∂vx/∂y
    dvy_dx = (np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0)) / (2 * dx)
    dvx_dy = (np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1)) / (2 * dy)
    omega_z = dvy_dx - dvx_dy
    
    return rho, pmag, Jz, omega_z


# ============================================================================
# CONSERVATION METRICS
# ============================================================================

def compute_conservation_metrics(
    U: np.ndarray,
    dx: float,
    dy: float,
    gamma: float
) -> Dict[str, float]:
    """
    Compute conservation metrics for MHD simulation.
    
    For ideal MHD, the following should be conserved:
        - Total mass: M = ∫ ρ dV
        - Total momentum: P = ∫ ρv dV
        - Total energy: E = ∫ (ρv²/2 + p/(γ-1) + B²/2) dV
        - Cross helicity: Hc = ∫ v·B dV
        - Magnetic flux: Φ = ∫ B dA
    
    Args:
        U: Conservative variables [7, nx, ny]
        dx, dy: Grid spacing
        gamma: Adiabatic index
    
    Returns:
        Dictionary with conservation metrics
    """
    dV = dx * dy
    
    rho = U[0]
    vx = U[1] / rho
    vy = U[2] / rho
    Bx = U[3]
    By = U[4]
    E_total = U[5]
    
    # Compute thermal pressure
    kinetic = 0.5 * rho * (vx**2 + vy**2)
    magnetic = 0.5 * (Bx**2 + By**2)
    p = (gamma - 1) * (E_total - kinetic - magnetic)
    p = np.maximum(p, 1e-10)
    
    # Total mass
    total_mass = np.sum(rho) * dV
    
    # Total momentum
    total_momentum_x = np.sum(rho * vx) * dV
    total_momentum_y = np.sum(rho * vy) * dV
    total_momentum = np.sqrt(total_momentum_x**2 + total_momentum_y**2)
    
    # Total energy
    total_energy = np.sum(E_total) * dV
    
    # Energy components
    kinetic_energy = np.sum(kinetic) * dV
    magnetic_energy = np.sum(magnetic) * dV
    thermal_energy = np.sum(p / (gamma - 1)) * dV
    
    # Cross helicity: Hc = ∫ v·B dV
    cross_helicity = np.sum(vx * Bx + vy * By) * dV
    
    # Magnetic flux components
    magnetic_flux_x = np.sum(Bx) * dV
    magnetic_flux_y = np.sum(By) * dV
    
    # Divergence of B (should be ~0)
    div_B = (
        (np.roll(Bx, -1, axis=0) - np.roll(Bx, 1, axis=0)) / (2 * dx) +
        (np.roll(By, -1, axis=1) - np.roll(By, 1, axis=1)) / (2 * dy)
    )
    max_div_B = float(np.max(np.abs(div_B)))
    mean_div_B = float(np.mean(np.abs(div_B)))
    
    return {
        'total_mass': float(total_mass),
        'total_momentum_x': float(total_momentum_x),
        'total_momentum_y': float(total_momentum_y),
        'total_momentum': float(total_momentum),
        'total_energy': float(total_energy),
        'kinetic_energy': float(kinetic_energy),
        'magnetic_energy': float(magnetic_energy),
        'thermal_energy': float(thermal_energy),
        'cross_helicity': float(cross_helicity),
        'magnetic_flux_x': float(magnetic_flux_x),
        'magnetic_flux_y': float(magnetic_flux_y),
        'max_div_B': max_div_B,
        'mean_div_B': mean_div_B,
    }


# ============================================================================
# STABILITY METRICS
# ============================================================================

def compute_stability_metrics(
    U: np.ndarray,
    dx: float,
    dy: float,
    gamma: float,
    dt: float,
    cfl: float
) -> Dict[str, float]:
    """
    Compute stability metrics for MHD simulation.
    
    Args:
        U: Conservative variables
        dx, dy: Grid spacing
        gamma: Adiabatic index
        dt: Current time step
        cfl: Target CFL number
    
    Returns:
        Dictionary with stability metrics
    """
    rho = U[0]
    vx = U[1] / rho
    vy = U[2] / rho
    Bx = U[3]
    By = U[4]
    
    kinetic = 0.5 * rho * (vx**2 + vy**2)
    magnetic = 0.5 * (Bx**2 + By**2)
    p = (gamma - 1) * (U[5] - kinetic - magnetic)
    p = np.maximum(p, 1e-10)
    
    # Sound speed
    cs = np.sqrt(gamma * p / rho)
    
    # Alfvén speed
    v_A = np.sqrt((Bx**2 + By**2) / rho)
    
    # Fast magnetosonic speed
    cf = np.sqrt(cs**2 + v_A**2)
    
    # Velocity magnitude
    v_mag = np.sqrt(vx**2 + vy**2)
    
    # Mach numbers
    mach_sonic = v_mag / (cs + 1e-10)
    mach_alfven = v_mag / (v_A + 1e-10)
    mach_fast = v_mag / (cf + 1e-10)
    
    # Plasma beta = 2 * thermal_pressure / magnetic_pressure
    beta = 2 * p / (Bx**2 + By**2 + 1e-10)
    
    # CFL numbers
    cfl_x = np.max(np.abs(vx) + cf) * dt / dx
    cfl_y = np.max(np.abs(vy) + cf) * dt / dy
    cfl_effective = max(cfl_x, cfl_y)
    
    return {
        'max_sonic_mach': float(np.max(mach_sonic)),
        'mean_sonic_mach': float(np.mean(mach_sonic)),
        'max_alfven_mach': float(np.max(mach_alfven)),
        'mean_alfven_mach': float(np.mean(mach_alfven)),
        'max_fast_mach': float(np.max(mach_fast)),
        'max_velocity': float(np.max(v_mag)),
        'max_sound_speed': float(np.max(cs)),
        'max_alfven_speed': float(np.max(v_A)),
        'max_fast_speed': float(np.max(cf)),
        'min_beta': float(np.min(beta)),
        'max_beta': float(np.max(beta)),
        'mean_beta': float(np.mean(beta)),
        'cfl_x': float(cfl_x),
        'cfl_y': float(cfl_y),
        'cfl_effective': float(cfl_effective),
        'target_cfl': float(cfl),
        'dt': float(dt),
        'min_density': float(np.min(rho)),
        'min_pressure': float(np.min(p)),
        'is_stable': float(np.min(rho) > 0 and np.min(p) > 0),
    }


# ============================================================================
# TURBULENCE METRICS
# ============================================================================

def compute_turbulence_metrics(
    U: np.ndarray,
    dx: float,
    dy: float,
    gamma: float
) -> Dict[str, float]:
    """
    Compute essential MHD turbulence metrics.
    
    Args:
        U: Conservative variables
        dx, dy: Grid spacing
        gamma: Adiabatic index
    
    Returns:
        Dictionary with turbulence metrics
    """
    dV = dx * dy
    
    rho = U[0]
    vx = U[1] / rho
    vy = U[2] / rho
    Bx = U[3]
    By = U[4]
    
    v_mag = np.sqrt(vx**2 + vy**2)
    B_mag = np.sqrt(Bx**2 + By**2)
    
    # Get diagnostic fields
    _, pmag, Jz, omega_z = compute_diagnostics(U, dx, dy, gamma)
    
    # Energy statistics
    kinetic_energy = 0.5 * np.sum(rho * (vx**2 + vy**2)) * dV
    magnetic_energy = 0.5 * np.sum(Bx**2 + By**2) * dV
    total_energy = kinetic_energy + magnetic_energy
    energy_ratio = kinetic_energy / (magnetic_energy + 1e-10)
    
    # Enstrophy (integral of vorticity squared)
    enstrophy = np.sum(omega_z**2) * dV
    mean_vorticity = float(np.mean(np.abs(omega_z)))
    max_vorticity = float(np.max(np.abs(omega_z)))
    
    # Current statistics
    current_squared = np.sum(Jz**2) * dV
    mean_current = float(np.mean(np.abs(Jz)))
    max_current = float(np.max(np.abs(Jz)))
    rms_current = float(np.sqrt(np.mean(Jz**2)))
    
    # Cross helicity & residual energy
    cross_helicity = np.sum(vx * Bx + vy * By) * dV
    normalized_cross_helicity = cross_helicity / (
        np.sqrt(kinetic_energy * magnetic_energy) + 1e-10
    )
    residual_energy = kinetic_energy - magnetic_energy
    normalized_residual_energy = residual_energy / (total_energy + 1e-10)
    
    # Elsässer variables: z± = v ± B/√ρ
    sqrt_rho = np.sqrt(rho)
    zp_x = vx + Bx / sqrt_rho
    zp_y = vy + By / sqrt_rho
    zm_x = vx - Bx / sqrt_rho
    zm_y = vy - By / sqrt_rho
    
    zp_energy = 0.5 * np.sum(rho * (zp_x**2 + zp_y**2)) * dV
    zm_energy = 0.5 * np.sum(rho * (zm_x**2 + zm_y**2)) * dV
    energy_imbalance = (zp_energy - zm_energy) / (zp_energy + zm_energy + 1e-10)
    
    # Kurtosis (intermittency measure)
    omega_var = np.mean(omega_z**2)
    Jz_var = np.mean(Jz**2)
    
    vorticity_kurtosis = np.mean(omega_z**4) / (omega_var**2 + 1e-20)
    current_kurtosis = np.mean(Jz**4) / (Jz_var**2 + 1e-20)
    
    # Taylor microscale
    if mean_vorticity > 1e-10:
        taylor_microscale = np.sqrt(np.var(v_mag) / (mean_vorticity**2 + 1e-10))
    else:
        taylor_microscale = 0.0
    
    return {
        'kinetic_energy': float(kinetic_energy),
        'magnetic_energy': float(magnetic_energy),
        'total_energy': float(total_energy),
        'energy_ratio': float(energy_ratio),
        'enstrophy': float(enstrophy),
        'mean_vorticity': mean_vorticity,
        'max_vorticity': max_vorticity,
        'current_squared': float(current_squared),
        'mean_current': mean_current,
        'max_current': max_current,
        'rms_current': rms_current,
        'cross_helicity': float(cross_helicity),
        'normalized_cross_helicity': float(normalized_cross_helicity),
        'residual_energy': float(residual_energy),
        'normalized_residual_energy': float(normalized_residual_energy),
        'elsasser_plus_energy': float(zp_energy),
        'elsasser_minus_energy': float(zm_energy),
        'energy_imbalance': float(energy_imbalance),
        'vorticity_kurtosis': float(vorticity_kurtosis),
        'current_kurtosis': float(current_kurtosis),
        'taylor_microscale': float(taylor_microscale),
    }


# ============================================================================
# MASTER METRICS FUNCTION
# ============================================================================

def compute_all_metrics(
    U: np.ndarray,
    dx: float,
    dy: float,
    gamma: float,
    dt: float = 0.0,
    cfl: float = 0.4,
    initial_metrics: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Compute all MHD metrics for a given state.
    
    Args:
        U: Conservative variables [7, nx, ny]
        dx, dy: Grid spacing
        gamma: Adiabatic index
        dt: Current time step
        cfl: Target CFL number
        initial_metrics: Initial conservation metrics for error calculation
    
    Returns:
        Comprehensive dictionary of all metrics
    """
    # Conservation metrics
    conservation = compute_conservation_metrics(U, dx, dy, gamma)
    
    # Stability metrics
    stability = compute_stability_metrics(U, dx, dy, gamma, dt, cfl)
    
    # Turbulence metrics
    turbulence = compute_turbulence_metrics(U, dx, dy, gamma)
    
    # Conservation errors (if initial metrics provided)
    if initial_metrics is not None:
        mass_error = abs(conservation['total_mass'] - initial_metrics['total_mass']) / (
            initial_metrics['total_mass'] + 1e-10
        )
        energy_error = abs(conservation['total_energy'] - initial_metrics['total_energy']) / (
            initial_metrics['total_energy'] + 1e-10
        )
    else:
        mass_error = 0.0
        energy_error = 0.0
    
    # Combine all metrics
    results = {}
    
    # Add conservation metrics
    for key, value in conservation.items():
        results[key] = value
    
    # Add stability metrics
    for key, value in stability.items():
        results[key] = value
    
    # Add turbulence metrics
    for key, value in turbulence.items():
        results[key] = value
    
    # Add conservation errors
    results['mass_conservation_error'] = float(mass_error)
    results['energy_conservation_error'] = float(energy_error)
    
    return results
