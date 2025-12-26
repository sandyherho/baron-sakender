"""
Metrics for 2D Ideal MHD Analysis.

Provides reliable metrics for quantifying MHD dynamics with HLL solver:
    - Conservation metrics (mass, momentum, energy, magnetic flux)
    - Stability metrics (CFL, Mach numbers, plasma beta)
    - Turbulence metrics (enstrophy, current, cross helicity)


References:
    Biskamp, D. (2003). Magnetohydrodynamic Turbulence.
    Dedner, A., et al. (2002). J. Comput. Phys., 175(2), 645-673.
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
    
    # Current density Jz = dBy/dx - dBx/dy
    dBy_dx = (np.roll(By, -1, axis=0) - np.roll(By, 1, axis=0)) / (2 * dx)
    dBx_dy = (np.roll(Bx, -1, axis=1) - np.roll(Bx, 1, axis=1)) / (2 * dy)
    Jz = dBy_dx - dBx_dy
    
    # Vorticity wz = dvy/dx - dvx/dy
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
    
    These are the most reliable metrics - ideal MHD conserves:
        - Total mass: M = integral rho dV
        - Total momentum: P = integral rho*v dV
        - Total energy: E = integral (rho*v^2/2 + p/(gamma-1) + B^2/2) dV
        - Cross helicity: Hc = integral v.B dV
    
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
    
    # Cross helicity: Hc = integral v.B dV (MHD invariant in 2D)
    cross_helicity = np.sum(vx * Bx + vy * By) * dV
    
    # Divergence of B (should be ~0 with Dedner cleaning)
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
    
    # Alfven speed
    B2 = Bx**2 + By**2
    v_A = np.sqrt(B2 / rho)
    
    # Fast magnetosonic speed
    cf = np.sqrt(cs**2 + v_A**2)
    
    # Velocity magnitude
    v_mag = np.sqrt(vx**2 + vy**2)
    
    # Mach numbers
    mach_sonic = v_mag / (cs + 1e-10)
    mach_alfven = v_mag / (v_A + 1e-10)
    mach_fast = v_mag / (cf + 1e-10)
    
    # Plasma beta = 2p / B^2 (only where B is significant)
    beta = 2 * p / (B2 + 1e-10)
    
    # Filter extreme beta values (where B ~ 0)
    B_threshold = 0.01 * np.mean(np.sqrt(B2))
    beta_filtered = np.where(np.sqrt(B2) > B_threshold, beta, np.nan)
    
    # CFL numbers
    cfl_x = np.max(np.abs(vx) + cf) * dt / dx
    cfl_y = np.max(np.abs(vy) + cf) * dt / dy
    cfl_effective = max(cfl_x, cfl_y)
    
    return {
        'max_sonic_mach': float(np.max(mach_sonic)),
        'mean_sonic_mach': float(np.mean(mach_sonic)),
        'max_alfven_mach': float(np.nanmax(np.where(v_A > 1e-10, mach_alfven, np.nan))),
        'mean_alfven_mach': float(np.nanmean(np.where(v_A > 1e-10, mach_alfven, np.nan))),
        'max_fast_mach': float(np.max(mach_fast)),
        'max_velocity': float(np.max(v_mag)),
        'max_sound_speed': float(np.max(cs)),
        'max_alfven_speed': float(np.max(v_A)),
        'max_fast_speed': float(np.max(cf)),
        'min_beta': float(np.nanmin(beta_filtered)),
        'max_beta': float(np.nanmax(beta_filtered)),
        'mean_beta': float(np.nanmean(beta_filtered)),
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
    gamma: float,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compute reliable MHD turbulence metrics.
    
    Args:
        U: Conservative variables
        dx, dy: Grid spacing
        gamma: Adiabatic index
        verbose: Print progress
    
    Returns:
        Dictionary with turbulence metrics
    """
    dV = dx * dy
    
    rho = U[0]
    vx = U[1] / rho
    vy = U[2] / rho
    Bx = U[3]
    By = U[4]
    
    # Get diagnostic fields
    _, pmag, Jz, omega_z = compute_diagnostics(U, dx, dy, gamma)
    
    # === Energy Statistics (Reliable) ===
    kinetic_energy = 0.5 * np.sum(rho * (vx**2 + vy**2)) * dV
    magnetic_energy = 0.5 * np.sum(Bx**2 + By**2) * dV
    total_energy = kinetic_energy + magnetic_energy
    energy_ratio = kinetic_energy / (magnetic_energy + 1e-10)
    
    # === Enstrophy & Vorticity (Reliable) ===
    enstrophy = np.sum(omega_z**2) * dV
    mean_vorticity = float(np.mean(np.abs(omega_z)))
    max_vorticity = float(np.max(np.abs(omega_z)))
    rms_vorticity = float(np.sqrt(np.mean(omega_z**2)))
    
    # === Current Statistics (Reliable) ===
    current_squared = np.sum(Jz**2) * dV
    mean_current = float(np.mean(np.abs(Jz)))
    max_current = float(np.max(np.abs(Jz)))
    rms_current = float(np.sqrt(np.mean(Jz**2)))
    
    # === Cross Helicity (MHD Invariant) ===
    cross_helicity = np.sum(vx * Bx + vy * By) * dV
    sigma_c = cross_helicity / (np.sqrt(kinetic_energy * magnetic_energy) + 1e-10)
    
    # === Residual Energy ===
    residual_energy = kinetic_energy - magnetic_energy
    sigma_r = residual_energy / (total_energy + 1e-10)
    
    # === Elsasser Variables ===
    # z+ = v + B/sqrt(rho), z- = v - B/sqrt(rho)
    sqrt_rho = np.sqrt(rho)
    zp_x = vx + Bx / sqrt_rho
    zp_y = vy + By / sqrt_rho
    zm_x = vx - Bx / sqrt_rho
    zm_y = vy - By / sqrt_rho
    
    zp_energy = 0.5 * np.sum(rho * (zp_x**2 + zp_y**2)) * dV
    zm_energy = 0.5 * np.sum(rho * (zm_x**2 + zm_y**2)) * dV
    energy_imbalance = (zp_energy - zm_energy) / (zp_energy + zm_energy + 1e-10)
    
    # === Intermittency (Kurtosis) ===
    # Kurtosis > 3 indicates intermittency (non-Gaussian tails)
    omega_var = np.mean(omega_z**2)
    Jz_var = np.mean(Jz**2)
    
    vorticity_kurtosis = np.mean(omega_z**4) / (omega_var**2 + 1e-20)
    current_kurtosis = np.mean(Jz**4) / (Jz_var**2 + 1e-20)
    
    # === Taylor Microscale (Approximate) ===
    v_rms = np.sqrt(np.mean(vx**2 + vy**2))
    if rms_vorticity > 1e-10:
        taylor_microscale = v_rms / rms_vorticity
    else:
        taylor_microscale = 0.0
    
    return {
        # Energies
        'kinetic_energy': float(kinetic_energy),
        'magnetic_energy': float(magnetic_energy),
        'total_energy': float(total_energy),
        'energy_ratio': float(energy_ratio),
        # Vorticity
        'enstrophy': float(enstrophy),
        'mean_vorticity': mean_vorticity,
        'max_vorticity': max_vorticity,
        'rms_vorticity': rms_vorticity,
        # Current
        'current_squared': float(current_squared),
        'mean_current': mean_current,
        'max_current': max_current,
        'rms_current': rms_current,
        # Cross helicity
        'cross_helicity': float(cross_helicity),
        'normalized_cross_helicity': float(sigma_c),
        # Residual energy
        'residual_energy': float(residual_energy),
        'normalized_residual_energy': float(sigma_r),
        # Elsasser
        'elsasser_plus_energy': float(zp_energy),
        'elsasser_minus_energy': float(zm_energy),
        'energy_imbalance': float(energy_imbalance),
        # Intermittency
        'vorticity_kurtosis': float(vorticity_kurtosis),
        'current_kurtosis': float(current_kurtosis),
        # Scales
        'taylor_microscale': float(taylor_microscale),
    }


# ============================================================================
# COMPOSITE METRICS 
# ============================================================================

def compute_composite_metrics(
    U: np.ndarray,
    dx: float,
    dy: float,
    gamma: float,
    conservation_initial: Optional[Dict[str, float]] = None,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Compute composite metrics for MHD analysis.
    
    Args:
        U: Conservative variables
        dx, dy: Grid spacing
        gamma: Adiabatic index
        conservation_initial: Initial conservation metrics
        verbose: Print progress
    
    Returns:
        Dictionary with composite metrics
    """
    # Get base metrics
    turb = compute_turbulence_metrics(U, dx, dy, gamma, verbose=False)
    cons = compute_conservation_metrics(U, dx, dy, gamma)
    
    # === Dynamo Efficiency (Reliable) ===
    # Fraction of total fluctuation energy in magnetic form
    E_k = turb['kinetic_energy']
    E_m = turb['magnetic_energy']
    dynamo_efficiency = E_m / (E_k + E_m + 1e-10)
    
    # === Alignment Index (Reliable) ===
    # |sigma_c| measures v-B alignment
    alignment_index = abs(turb['normalized_cross_helicity'])
    
    # === Current Sheet Index (Reliable) ===
    # High J_max/J_rms indicates coherent current sheets
    j_rms = turb['rms_current']
    j_max = turb['max_current']
    coherent_structure_index = (j_max / (j_rms + 1e-10)) / 10
    
    # === Intermittency Index (Reliable) ===
    # Kurtosis deviation from Gaussian (kappa = 3)
    kurtosis_avg = 0.5 * (turb['vorticity_kurtosis'] + turb['current_kurtosis'])
    intermittency_index = max(0, (kurtosis_avg - 3) / 10)
    
    # === Conservation Quality (Most Reliable) ===
    conservation_quality = 1.0
    mass_error = 0.0
    energy_error = 0.0
    
    if conservation_initial is not None:
        mass_0 = conservation_initial['total_mass']
        energy_0 = conservation_initial['total_energy']
        
        mass_error = abs(cons['total_mass'] - mass_0) / (mass_0 + 1e-10)
        energy_error = abs(cons['total_energy'] - energy_0) / (energy_0 + 1e-10)
        
        conservation_quality = 1 - (mass_error + energy_error) / 2
        conservation_quality = max(0, min(1, conservation_quality))
    
    return {
        'dynamo_efficiency': float(dynamo_efficiency),
        'alignment_index': float(alignment_index),
        'coherent_structure_index': float(coherent_structure_index),
        'intermittency_index': float(intermittency_index),
        'conservation_quality': float(conservation_quality),
        'mass_conservation_error': float(mass_error),
        'energy_conservation_error': float(energy_error),
    }


# ============================================================================
# INFORMATION METRICS
# ============================================================================

def compute_information_metrics(
    U: np.ndarray,
    dx: float,
    dy: float,
    gamma: float,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Compute information-theoretic metrics.
    
    
    Args:
        U: Conservative variables
        dx, dy: Grid spacing
        gamma: Adiabatic index
        verbose: Print progress
    
    Returns:
        Dictionary with information metrics
    """
    rho = U[0]
    vx = U[1] / rho
    vy = U[2] / rho
    Bx = U[3]
    By = U[4]
    
    v_mag = np.sqrt(vx**2 + vy**2)
    B_mag = np.sqrt(Bx**2 + By**2)
    
    # Get vorticity and current
    _, _, Jz, omega_z = compute_diagnostics(U, dx, dy, gamma)
    
    def shannon_entropy(field, n_bins=64):
        """Compute Shannon entropy of a field."""
        f_min, f_max = np.min(field), np.max(field)
        if f_max - f_min < 1e-10:
            return 0.0
        f_norm = (field - f_min) / (f_max - f_min)
        hist, _ = np.histogram(f_norm.flatten(), bins=n_bins, range=(0, 1), density=True)
        hist = hist / (np.sum(hist) + 1e-10)
        mask = hist > 0
        return float(-np.sum(hist[mask] * np.log2(hist[mask] + 1e-20)))
    
    # Shannon entropy of key fields
    entropy_density = shannon_entropy(rho)
    entropy_velocity = shannon_entropy(v_mag)
    entropy_magnetic = shannon_entropy(B_mag)
    entropy_vorticity = shannon_entropy(omega_z)
    entropy_current = shannon_entropy(Jz)
    
    return {
        'entropy_density': entropy_density,
        'entropy_velocity': entropy_velocity,
        'entropy_magnetic': entropy_magnetic,
        'entropy_vorticity': entropy_vorticity,
        'entropy_current': entropy_current,
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
    conservation_initial: Optional[Dict[str, float]] = None,
    initial_metrics: Optional[Dict[str, float]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compute all reliable MHD metrics for a given state.
    
    Args:
        U: Conservative variables [7, nx, ny]
        dx, dy: Grid spacing
        gamma: Adiabatic index
        dt: Current time step
        cfl: Target CFL number
        conservation_initial: Initial conservation metrics
        initial_metrics: Alias for conservation_initial
        verbose: Print progress
    
    Returns:
        Comprehensive dictionary of reliable metrics
    """
    # Handle both parameter names
    if initial_metrics is not None and conservation_initial is None:
        conservation_initial = initial_metrics
    
    if verbose:
        print("      Computing conservation metrics...")
    conservation = compute_conservation_metrics(U, dx, dy, gamma)
    
    if verbose:
        print("      Computing stability metrics...")
    stability = compute_stability_metrics(U, dx, dy, gamma, dt, cfl)
    
    if verbose:
        print("      Computing turbulence metrics...")
    turbulence = compute_turbulence_metrics(U, dx, dy, gamma, verbose=False)
    
    if verbose:
        print("      Computing information metrics...")
    information = compute_information_metrics(U, dx, dy, gamma, verbose=False)
    
    if verbose:
        print("      Computing composite metrics...")
    composite = compute_composite_metrics(U, dx, dy, gamma, conservation_initial, verbose=False)
    
    # Combine all metrics with prefixes
    results = {}
    
    # Conservation metrics
    for key, value in conservation.items():
        results[f'cons_{key}'] = value
        results[key] = value
    
    # Stability metrics
    for key, value in stability.items():
        results[f'stab_{key}'] = value
        results[key] = value
    
    # Turbulence metrics
    for key, value in turbulence.items():
        results[f'turb_{key}'] = value
        results[key] = value
    
    # Information metrics
    for key, value in information.items():
        results[f'info_{key}'] = value
        results[key] = value
    
    # Composite metrics
    for key, value in composite.items():
        results[f'comp_{key}'] = value
        results[key] = value
    
    return results
