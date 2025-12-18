"""
Comprehensive Metrics for 2D Ideal MHD Analysis.

Implements rigorous measures for quantifying MHD dynamics:
    - Conservation metrics (mass, momentum, energy, helicity)
    - Stability metrics (CFL, Mach numbers, plasma beta)
    - Turbulence metrics (spectra, structure functions, dissipation)
    - Information-theoretic metrics (entropy, complexity, mutual information)
    - Novel composite turbulence metrics

Physical units are maintained throughout using SI with normalized
reference values appropriate for space plasma applications.

References:
    Biskamp, D. (2003). Magnetohydrodynamic Turbulence.
    Bruno, R., & Carbone, V. (2013). The Solar Wind as a Turbulence Laboratory.
    Frisch, U. (1995). Turbulence: The Legacy of A. N. Kolmogorov.
    Kantz, H., & Schreiber, T. (2004). Nonlinear Time Series Analysis.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from scipy.fft import fft2, fftfreq
from scipy.stats import entropy as scipy_entropy
from scipy.ndimage import uniform_filter


def compute_diagnostics(
    U: np.ndarray,
    dx: float,
    dy: float,
    gamma: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute basic diagnostic fields from conservative variables.
    
    Args:
        U: Conservative variables [6, nx, ny]
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
        - Total x-momentum: Px = ∫ ρvx dV
        - Total y-momentum: Py = ∫ ρvy dV
        - Total energy: E = ∫ (ρv²/2 + p/(γ-1) + B²/2) dV
        - Cross helicity: Hc = ∫ v·B dV
        - Magnetic flux: Φ = ∫ B dA (in 2D, ∫Bx and ∫By)
    
    Args:
        U: Conservative variables [6, nx, ny]
        dx, dy: Grid spacing
        gamma: Adiabatic index
    
    Returns:
        Dictionary with conservation metrics
    """
    dV = dx * dy  # Cell volume
    
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
    max_div_B = np.max(np.abs(div_B))
    mean_div_B = np.mean(np.abs(div_B))
    
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
        'max_div_B': float(max_div_B),
        'mean_div_B': float(mean_div_B),
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
    
    These metrics help monitor numerical stability and physical regime.
    
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
    
    # Alfvén speeds
    v_A = np.sqrt((Bx**2 + By**2) / rho)
    v_Ax = np.abs(Bx) / np.sqrt(rho)
    v_Ay = np.abs(By) / np.sqrt(rho)
    
    # Fast magnetosonic speed
    cf = np.sqrt(cs**2 + v_A**2)
    
    # Velocity magnitude
    v_mag = np.sqrt(vx**2 + vy**2)
    
    # Mach numbers
    mach_sonic = v_mag / cs
    mach_alfven = v_mag / (v_A + 1e-10)
    mach_fast = v_mag / cf
    
    # Plasma beta = 2 * thermal_pressure / magnetic_pressure
    beta = 2 * p / (Bx**2 + By**2 + 1e-10)
    
    # CFL numbers
    cfl_x = np.max(np.abs(vx) + cf) * dt / dx
    cfl_y = np.max(np.abs(vy) + cf) * dt / dy
    cfl_effective = max(cfl_x, cfl_y)
    
    # Positivity checks
    min_density = float(np.min(rho))
    min_pressure = float(np.min(p))
    
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
        'min_density': min_density,
        'min_pressure': min_pressure,
        'is_stable': bool(min_density > 0 and min_pressure > 0),
    }


# ============================================================================
# TURBULENCE METRICS (Standard MHD)
# ============================================================================

def compute_energy_spectrum(
    U: np.ndarray,
    dx: float,
    dy: float,
    gamma: float,
    n_bins: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 1D energy spectra E(k) for kinetic and magnetic energy.
    
    Uses 2D FFT and radial averaging.
    
    Args:
        U: Conservative variables
        dx, dy: Grid spacing
        gamma: Adiabatic index
        n_bins: Number of bins for spectrum
    
    Returns:
        Tuple of (k_bins, E_kinetic, E_magnetic)
    """
    nx, ny = U.shape[1], U.shape[2]
    
    rho = U[0]
    vx = U[1] / rho
    vy = U[2] / rho
    Bx = U[3]
    By = U[4]
    
    # 2D FFT of velocity components
    vx_hat = fft2(vx)
    vy_hat = fft2(vy)
    
    # Power spectra
    E_vx = np.abs(vx_hat)**2
    E_vy = np.abs(vy_hat)**2
    E_kinetic_2d = 0.5 * (E_vx + E_vy)
    
    # 2D FFT of magnetic field
    Bx_hat = fft2(Bx)
    By_hat = fft2(By)
    
    E_Bx = np.abs(Bx_hat)**2
    E_By = np.abs(By_hat)**2
    E_magnetic_2d = 0.5 * (E_Bx + E_By)
    
    # Wavenumber arrays
    kx = fftfreq(nx, dx) * 2 * np.pi
    ky = fftfreq(ny, dy) * 2 * np.pi
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    K_mag = np.sqrt(Kx**2 + Ky**2)
    
    # Radial binning
    k_max = np.min([np.max(np.abs(kx)), np.max(np.abs(ky))])
    k_bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
    E_kinetic_1d = np.zeros(n_bins)
    E_magnetic_1d = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (K_mag >= k_bins[i]) & (K_mag < k_bins[i + 1])
        if np.any(mask):
            E_kinetic_1d[i] = np.sum(E_kinetic_2d[mask])
            E_magnetic_1d[i] = np.sum(E_magnetic_2d[mask])
    
    # Normalize by bin width
    dk = k_bins[1] - k_bins[0]
    E_kinetic_1d /= dk
    E_magnetic_1d /= dk
    
    return k_centers, E_kinetic_1d, E_magnetic_1d


def compute_structure_functions(
    field: np.ndarray,
    max_lag: int = 50,
    orders: List[int] = [2, 3, 4]
) -> Dict[str, np.ndarray]:
    """
    Compute structure functions S_p(l) = <|δf(l)|^p>.
    
    Args:
        field: 2D field array
        max_lag: Maximum lag in grid points
        orders: Orders of structure functions
    
    Returns:
        Dictionary with lags and structure functions
    """
    nx, ny = field.shape
    lags = np.arange(1, max_lag + 1)
    
    results = {'lags': lags}
    
    for p in orders:
        S_p = np.zeros(max_lag)
        
        for i, lag in enumerate(lags):
            # X-direction increments
            delta_x = field[lag:, :] - field[:-lag, :]
            # Y-direction increments
            delta_y = field[:, lag:] - field[:, :-lag]
            
            # Combine
            S_p[i] = 0.5 * (np.mean(np.abs(delta_x)**p) + 
                           np.mean(np.abs(delta_y)**p))
        
        results[f'S_{p}'] = S_p
    
    return results


def compute_turbulence_metrics(
    U: np.ndarray,
    dx: float,
    dy: float,
    gamma: float,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compute comprehensive MHD turbulence metrics.
    
    Includes:
        - Energy statistics
        - Enstrophy and helicity
        - Current sheet analysis
        - Spectral indices
        - Elsässer variables
        - Dissipation estimates
    
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
    
    v_mag = np.sqrt(vx**2 + vy**2)
    B_mag = np.sqrt(Bx**2 + By**2)
    
    # Get diagnostic fields
    _, pmag, Jz, omega_z = compute_diagnostics(U, dx, dy, gamma)
    
    # ---- Energy Statistics ----
    kinetic_energy = 0.5 * np.sum(rho * (vx**2 + vy**2)) * dV
    magnetic_energy = 0.5 * np.sum(Bx**2 + By**2) * dV
    total_energy = kinetic_energy + magnetic_energy
    
    # Energy ratio
    energy_ratio = kinetic_energy / (magnetic_energy + 1e-10)
    
    # ---- Enstrophy ----
    enstrophy = np.sum(omega_z**2) * dV
    mean_vorticity = np.mean(np.abs(omega_z))
    max_vorticity = np.max(np.abs(omega_z))
    
    # ---- Current Statistics ----
    current_squared = np.sum(Jz**2) * dV
    mean_current = np.mean(np.abs(Jz))
    max_current = np.max(np.abs(Jz))
    
    # ---- Cross Helicity & Residual Energy ----
    cross_helicity = np.sum(vx * Bx + vy * By) * dV
    normalized_cross_helicity = cross_helicity / (
        np.sqrt(kinetic_energy * magnetic_energy) + 1e-10
    )
    
    residual_energy = kinetic_energy - magnetic_energy
    normalized_residual_energy = residual_energy / (total_energy + 1e-10)
    
    # ---- Elsässer Variables ----
    # z± = v ± B/√(ρ) (normalized)
    sqrt_rho = np.sqrt(rho)
    zp_x = vx + Bx / sqrt_rho
    zp_y = vy + By / sqrt_rho
    zm_x = vx - Bx / sqrt_rho
    zm_y = vy - By / sqrt_rho
    
    zp_energy = 0.5 * np.sum(rho * (zp_x**2 + zp_y**2)) * dV
    zm_energy = 0.5 * np.sum(rho * (zm_x**2 + zm_y**2)) * dV
    
    # Imbalance
    energy_imbalance = (zp_energy - zm_energy) / (zp_energy + zm_energy + 1e-10)
    
    # ---- Spectra (optional, can be slow) ----
    k_centers, E_k, E_m = compute_energy_spectrum(U, dx, dy, gamma, n_bins=30)
    
    # Fit spectral index in inertial range
    valid = (k_centers > k_centers[5]) & (k_centers < k_centers[-5]) & (E_k > 0)
    if np.sum(valid) > 5:
        log_k = np.log10(k_centers[valid])
        log_E_k = np.log10(E_k[valid] + 1e-20)
        log_E_m = np.log10(E_m[valid] + 1e-20)
        
        coeffs_k = np.polyfit(log_k, log_E_k, 1)
        coeffs_m = np.polyfit(log_k, log_E_m, 1)
        
        spectral_index_kinetic = coeffs_k[0]
        spectral_index_magnetic = coeffs_m[0]
    else:
        spectral_index_kinetic = np.nan
        spectral_index_magnetic = np.nan
    
    # ---- Intermittency (kurtosis) ----
    vorticity_kurtosis = np.mean(omega_z**4) / (np.mean(omega_z**2)**2 + 1e-10)
    current_kurtosis = np.mean(Jz**4) / (np.mean(Jz**2)**2 + 1e-10)
    
    # ---- Characteristic Scales ----
    # Taylor microscale (approximate)
    lambda_taylor = np.sqrt(v_mag.var() / (mean_vorticity**2 + 1e-10))
    
    # Integral scale (from autocorrelation)
    L_integral = np.nan  # Would need proper autocorrelation
    
    return {
        # Energy
        'kinetic_energy': float(kinetic_energy),
        'magnetic_energy': float(magnetic_energy),
        'total_turbulent_energy': float(total_energy),
        'energy_ratio_Ek_Em': float(energy_ratio),
        
        # Enstrophy & vorticity
        'enstrophy': float(enstrophy),
        'mean_vorticity': float(mean_vorticity),
        'max_vorticity': float(max_vorticity),
        
        # Current
        'current_squared_integral': float(current_squared),
        'mean_current_density': float(mean_current),
        'max_current_density': float(max_current),
        
        # Helicity & residual energy
        'cross_helicity': float(cross_helicity),
        'normalized_cross_helicity': float(normalized_cross_helicity),
        'residual_energy': float(residual_energy),
        'normalized_residual_energy': float(normalized_residual_energy),
        
        # Elsässer
        'elsasser_plus_energy': float(zp_energy),
        'elsasser_minus_energy': float(zm_energy),
        'energy_imbalance': float(energy_imbalance),
        
        # Spectra
        'spectral_index_kinetic': float(spectral_index_kinetic),
        'spectral_index_magnetic': float(spectral_index_magnetic),
        'k_spectrum': k_centers.tolist(),
        'E_kinetic_spectrum': E_k.tolist(),
        'E_magnetic_spectrum': E_m.tolist(),
        
        # Intermittency
        'vorticity_kurtosis': float(vorticity_kurtosis),
        'current_kurtosis': float(current_kurtosis),
        
        # Scales
        'taylor_microscale': float(lambda_taylor),
    }


# ============================================================================
# INFORMATION-THEORETIC METRICS
# ============================================================================

def compute_shannon_entropy(field: np.ndarray, n_bins: int = 64) -> float:
    """
    Compute Shannon entropy of a 2D field.
    
    H = -Σ p_i * log(p_i)
    
    Args:
        field: 2D array
        n_bins: Number of histogram bins
    
    Returns:
        Shannon entropy in bits
    """
    hist, _ = np.histogram(field.flatten(), bins=n_bins, density=True)
    hist = hist[hist > 0]  # Remove zeros
    return float(scipy_entropy(hist, base=2))


def compute_joint_entropy(field1: np.ndarray, field2: np.ndarray, n_bins: int = 32) -> float:
    """
    Compute joint Shannon entropy H(X,Y).
    """
    hist_2d, _, _ = np.histogram2d(
        field1.flatten(), field2.flatten(), bins=n_bins, density=True
    )
    hist_2d = hist_2d[hist_2d > 0]
    return float(scipy_entropy(hist_2d.flatten(), base=2))


def compute_mutual_information(field1: np.ndarray, field2: np.ndarray, n_bins: int = 32) -> float:
    """
    Compute mutual information I(X;Y) = H(X) + H(Y) - H(X,Y).
    """
    H_x = compute_shannon_entropy(field1, n_bins)
    H_y = compute_shannon_entropy(field2, n_bins)
    H_xy = compute_joint_entropy(field1, field2, n_bins)
    
    return float(H_x + H_y - H_xy)


def compute_permutation_entropy(series: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """
    Compute permutation entropy for a 1D time series.
    
    Args:
        series: 1D array
        order: Embedding dimension
        delay: Time delay
    
    Returns:
        Permutation entropy (normalized)
    """
    from math import factorial
    
    n = len(series)
    n_patterns = n - (order - 1) * delay
    
    if n_patterns <= 0:
        return np.nan
    
    # Extract patterns
    patterns = []
    for i in range(n_patterns):
        pattern = tuple(np.argsort([series[i + j * delay] for j in range(order)]))
        patterns.append(pattern)
    
    # Count unique patterns
    unique, counts = np.unique(patterns, axis=0, return_counts=True)
    probs = counts / n_patterns
    
    # Entropy
    H = -np.sum(probs * np.log2(probs + 1e-10))
    
    # Normalize by maximum possible entropy
    H_max = np.log2(factorial(order))
    
    return float(H / H_max)


def compute_complexity_index(field: np.ndarray, n_bins: int = 32) -> float:
    """
    Compute statistical complexity C = H * D.
    
    Where H is normalized Shannon entropy and D is disequilibrium
    (distance from uniform distribution).
    
    Args:
        field: 2D array
        n_bins: Number of bins
    
    Returns:
        Statistical complexity
    """
    hist, _ = np.histogram(field.flatten(), bins=n_bins, density=True)
    hist = hist / (np.sum(hist) + 1e-10)  # Ensure normalization
    
    # Shannon entropy (normalized)
    H = scipy_entropy(hist + 1e-10, base=2) / np.log2(n_bins)
    
    # Uniform distribution
    uniform = np.ones(n_bins) / n_bins
    
    # Jensen-Shannon divergence as disequilibrium
    m = 0.5 * (hist + uniform)
    D_js = 0.5 * (scipy_entropy(hist + 1e-10, m + 1e-10, base=2) + 
                  scipy_entropy(uniform, m + 1e-10, base=2))
    
    # Complexity
    C = H * D_js
    
    return float(C)


def compute_multiscale_entropy(
    field: np.ndarray,
    max_scale: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute multiscale entropy by coarse-graining.
    
    Args:
        field: 2D array
        max_scale: Maximum coarse-graining scale
    
    Returns:
        Tuple of (scales, entropy values)
    """
    scales = np.arange(1, max_scale + 1)
    entropies = np.zeros(max_scale)
    
    for i, scale in enumerate(scales):
        # Coarse-grain the field
        coarse = uniform_filter(field, size=scale, mode='wrap')
        coarse = coarse[::scale, ::scale]  # Downsample
        
        if coarse.size < 100:
            entropies[i] = np.nan
        else:
            entropies[i] = compute_shannon_entropy(coarse, n_bins=min(32, coarse.size // 10))
    
    return scales, entropies


def compute_information_metrics(
    U: np.ndarray,
    dx: float,
    dy: float,
    gamma: float,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compute comprehensive information-theoretic metrics.
    
    Includes:
        - Shannon entropy of fields
        - Mutual information between v and B
        - Statistical complexity
        - Multiscale entropy
        - Permutation entropy (for spatial slices)
    
    Args:
        U: Conservative variables
        dx, dy: Grid spacing
        gamma: Adiabatic index
        verbose: Print progress
    
    Returns:
        Dictionary with information metrics
    """
    rho, pmag, Jz, omega_z = compute_diagnostics(U, dx, dy, gamma)
    
    rho_field = U[0]
    vx = U[1] / rho_field
    vy = U[2] / rho_field
    Bx = U[3]
    By = U[4]
    
    v_mag = np.sqrt(vx**2 + vy**2)
    B_mag = np.sqrt(Bx**2 + By**2)
    
    if verbose:
        print("      Computing Shannon entropy...")
    
    # Shannon entropy of different fields
    H_density = compute_shannon_entropy(rho_field)
    H_velocity = compute_shannon_entropy(v_mag)
    H_magnetic = compute_shannon_entropy(B_mag)
    H_vorticity = compute_shannon_entropy(omega_z)
    H_current = compute_shannon_entropy(Jz)
    
    if verbose:
        print("      Computing mutual information...")
    
    # Mutual information between v and B
    MI_vx_Bx = compute_mutual_information(vx, Bx)
    MI_vy_By = compute_mutual_information(vy, By)
    MI_vmag_Bmag = compute_mutual_information(v_mag, B_mag)
    MI_vorticity_current = compute_mutual_information(omega_z, Jz)
    
    if verbose:
        print("      Computing statistical complexity...")
    
    # Statistical complexity
    C_density = compute_complexity_index(rho_field)
    C_velocity = compute_complexity_index(v_mag)
    C_magnetic = compute_complexity_index(B_mag)
    C_vorticity = compute_complexity_index(omega_z)
    C_current = compute_complexity_index(Jz)
    
    if verbose:
        print("      Computing multiscale entropy...")
    
    # Multiscale entropy
    scales_v, mse_velocity = compute_multiscale_entropy(v_mag, max_scale=8)
    scales_B, mse_magnetic = compute_multiscale_entropy(B_mag, max_scale=8)
    
    # Permutation entropy of central slices
    nx, ny = vx.shape
    PE_vx = compute_permutation_entropy(vx[nx//2, :], order=4, delay=1)
    PE_vy = compute_permutation_entropy(vy[:, ny//2], order=4, delay=1)
    PE_Bx = compute_permutation_entropy(Bx[nx//2, :], order=4, delay=1)
    PE_By = compute_permutation_entropy(By[:, ny//2], order=4, delay=1)
    
    # Total information content (sum of entropies)
    total_information = H_density + H_velocity + H_magnetic + H_vorticity + H_current
    
    return {
        # Shannon entropy
        'entropy_density': H_density,
        'entropy_velocity': H_velocity,
        'entropy_magnetic': H_magnetic,
        'entropy_vorticity': H_vorticity,
        'entropy_current': H_current,
        'total_information_content': total_information,
        
        # Mutual information
        'MI_vx_Bx': MI_vx_Bx,
        'MI_vy_By': MI_vy_By,
        'MI_velocity_magnetic': MI_vmag_Bmag,
        'MI_vorticity_current': MI_vorticity_current,
        
        # Statistical complexity
        'complexity_density': C_density,
        'complexity_velocity': C_velocity,
        'complexity_magnetic': C_magnetic,
        'complexity_vorticity': C_vorticity,
        'complexity_current': C_current,
        
        # Multiscale entropy
        'mse_scales': scales_v.tolist(),
        'mse_velocity': mse_velocity.tolist(),
        'mse_magnetic': mse_magnetic.tolist(),
        
        # Permutation entropy
        'PE_vx': PE_vx,
        'PE_vy': PE_vy,
        'PE_Bx': PE_Bx,
        'PE_By': PE_By,
    }


# ============================================================================
# NOVEL COMPOSITE METRICS
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
    Compute novel composite turbulence metrics that combine multiple diagnostics.
    
    These metrics provide integrated measures of MHD turbulence properties
    not captured by individual diagnostics.
    
    Args:
        U: Conservative variables
        dx, dy: Grid spacing
        gamma: Adiabatic index
        conservation_initial: Initial conservation metrics for comparison
        verbose: Print progress
    
    Returns:
        Dictionary with composite metrics
    """
    rho, pmag, Jz, omega_z = compute_diagnostics(U, dx, dy, gamma)
    
    rho_field = U[0]
    vx = U[1] / rho_field
    vy = U[2] / rho_field
    Bx = U[3]
    By = U[4]
    
    dV = dx * dy
    v_mag = np.sqrt(vx**2 + vy**2)
    B_mag = np.sqrt(Bx**2 + By**2)
    
    # ---- 1. Dynamo Efficiency Index ----
    # Ratio of magnetic energy growth rate to velocity field energy
    kinetic_energy = 0.5 * np.sum(rho_field * v_mag**2) * dV
    magnetic_energy = 0.5 * np.sum(B_mag**2) * dV
    
    # Cross helicity normalized by geometric mean
    cross_helicity = np.sum(vx * Bx + vy * By) * dV
    dynamo_efficiency = np.abs(cross_helicity) / (
        np.sqrt(kinetic_energy * magnetic_energy) + 1e-10
    )
    
    # ---- 2. MHD Complexity Index ----
    # Combines entropy and structure measures
    H_omega = compute_shannon_entropy(omega_z, n_bins=32)
    H_J = compute_shannon_entropy(Jz, n_bins=32)
    C_omega = compute_complexity_index(omega_z, n_bins=32)
    C_J = compute_complexity_index(Jz, n_bins=32)
    
    mhd_complexity_index = np.sqrt(
        (H_omega * C_omega + H_J * C_J) / 2
    )
    
    # ---- 3. Coherent Structure Index ----
    # Based on Q-criterion analog for MHD
    S_v = 0.5 * (
        (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2*dx) +
        (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2*dy)
    )
    
    # |ω|² - |S|² criterion (vorticity vs strain)
    Q_criterion = omega_z**2 - S_v**2
    coherent_area = np.sum(Q_criterion > np.std(Q_criterion)) / Q_criterion.size
    
    # Current sheet detection (high J regions)
    J_threshold = np.mean(np.abs(Jz)) + 2 * np.std(np.abs(Jz))
    current_sheet_area = np.sum(np.abs(Jz) > J_threshold) / Jz.size
    
    coherent_structure_index = 0.5 * (coherent_area + current_sheet_area)
    
    # ---- 4. Scale-Dependent Intermittency Index ----
    # Based on flatness at different scales
    struct_omega = compute_structure_functions(omega_z, max_lag=20, orders=[2, 4])
    struct_J = compute_structure_functions(Jz, max_lag=20, orders=[2, 4])
    
    # Flatness F = S_4 / S_2^2
    flatness_omega = struct_omega['S_4'] / (struct_omega['S_2']**2 + 1e-10)
    flatness_J = struct_J['S_4'] / (struct_J['S_2']**2 + 1e-10)
    
    # Intermittency increases with scale
    intermittency_index_omega = np.mean(np.diff(np.log(flatness_omega + 1e-10)))
    intermittency_index_J = np.mean(np.diff(np.log(flatness_J + 1e-10)))
    
    intermittency_index = 0.5 * (intermittency_index_omega + intermittency_index_J)
    
    # ---- 5. Cascade Efficiency Index ----
    # Measures how efficiently energy cascades from large to small scales
    k_centers, E_k, E_m = compute_energy_spectrum(U, dx, dy, gamma, n_bins=20)
    
    # Energy in large scales (k < k_max/4) vs small scales (k > k_max/4)
    k_transition = k_centers[len(k_centers)//4]
    large_scale_energy = np.sum(E_k[k_centers < k_transition] + E_m[k_centers < k_transition])
    small_scale_energy = np.sum(E_k[k_centers >= k_transition] + E_m[k_centers >= k_transition])
    
    cascade_efficiency = small_scale_energy / (large_scale_energy + 1e-10)
    
    # ---- 6. Alignment Index ----
    # Measures alignment between v and B (relevant for Alfvén waves)
    v_dot_B = vx * Bx + vy * By
    v_cross_B_z = vx * By - vy * Bx
    
    alignment = np.mean(v_dot_B) / (np.mean(v_mag * B_mag) + 1e-10)
    perpendicularity = np.mean(np.abs(v_cross_B_z)) / (np.mean(v_mag * B_mag) + 1e-10)
    
    alignment_index = np.abs(alignment)
    
    # ---- 7. Conservation Quality Index ----
    if conservation_initial is not None:
        cons_current = compute_conservation_metrics(U, dx, dy, gamma)
        
        mass_error = np.abs(cons_current['total_mass'] - conservation_initial['total_mass']) / (
            conservation_initial['total_mass'] + 1e-10
        )
        energy_error = np.abs(cons_current['total_energy'] - conservation_initial['total_energy']) / (
            conservation_initial['total_energy'] + 1e-10
        )
        
        conservation_quality = 1.0 - 0.5 * (mass_error + energy_error)
        conservation_quality = max(0.0, conservation_quality)
    else:
        conservation_quality = np.nan
        mass_error = np.nan
        energy_error = np.nan
    
    # ---- 8. Turbulent Dissipation Proxy ----
    # In 2D ideal MHD, no explicit dissipation, but we can estimate
    # from grid-scale structures
    omega_grad = np.sqrt(
        ((np.roll(omega_z, -1, axis=0) - np.roll(omega_z, 1, axis=0)) / (2*dx))**2 +
        ((np.roll(omega_z, -1, axis=1) - np.roll(omega_z, 1, axis=1)) / (2*dy))**2
    )
    J_grad = np.sqrt(
        ((np.roll(Jz, -1, axis=0) - np.roll(Jz, 1, axis=0)) / (2*dx))**2 +
        ((np.roll(Jz, -1, axis=1) - np.roll(Jz, 1, axis=1)) / (2*dy))**2
    )
    
    numerical_dissipation_proxy = np.mean(omega_grad**2 + J_grad**2) * dV
    
    return {
        'dynamo_efficiency_index': float(dynamo_efficiency),
        'mhd_complexity_index': float(mhd_complexity_index),
        'coherent_structure_index': float(coherent_structure_index),
        'current_sheet_fraction': float(current_sheet_area),
        'vortex_fraction': float(coherent_area),
        'intermittency_index': float(intermittency_index),
        'intermittency_vorticity': float(intermittency_index_omega),
        'intermittency_current': float(intermittency_index_J),
        'cascade_efficiency': float(cascade_efficiency),
        'alignment_index': float(alignment_index),
        'perpendicularity_index': float(perpendicularity),
        'conservation_quality': float(conservation_quality),
        'mass_conservation_error': float(mass_error) if conservation_initial else np.nan,
        'energy_conservation_error': float(energy_error) if conservation_initial else np.nan,
        'numerical_dissipation_proxy': float(numerical_dissipation_proxy),
    }


# ============================================================================
# MASTER METRICS FUNCTION
# ============================================================================

def compute_all_metrics(
    U: np.ndarray,
    dx: float,
    dy: float,
    gamma: float,
    conservation_initial: Optional[Dict[str, float]] = None,
    dt: float = 0.0,
    cfl: float = 0.4,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute all MHD metrics for a given state.
    
    This is the master function that calls all metric computations
    and returns a comprehensive dictionary.
    
    Args:
        U: Conservative variables
        dx, dy: Grid spacing
        gamma: Adiabatic index
        conservation_initial: Initial conservation metrics
        dt: Current time step
        cfl: Target CFL number
        verbose: Print progress
    
    Returns:
        Comprehensive dictionary of all metrics
    """
    results = {}
    
    if verbose:
        print("      Computing conservation metrics...")
    conservation = compute_conservation_metrics(U, dx, dy, gamma)
    results.update({f'cons_{k}': v for k, v in conservation.items()})
    
    if verbose:
        print("      Computing stability metrics...")
    stability = compute_stability_metrics(U, dx, dy, gamma, dt, cfl)
    results.update({f'stab_{k}': v for k, v in stability.items()})
    
    if verbose:
        print("      Computing turbulence metrics...")
    turbulence = compute_turbulence_metrics(U, dx, dy, gamma, verbose=False)
    # Exclude spectrum arrays from flat dict
    turb_scalars = {k: v for k, v in turbulence.items() 
                    if not isinstance(v, (list, np.ndarray))}
    results.update({f'turb_{k}': v for k, v in turb_scalars.items()})
    
    # Store spectra separately
    results['k_spectrum'] = turbulence['k_spectrum']
    results['E_kinetic_spectrum'] = turbulence['E_kinetic_spectrum']
    results['E_magnetic_spectrum'] = turbulence['E_magnetic_spectrum']
    
    if verbose:
        print("      Computing information metrics...")
    info = compute_information_metrics(U, dx, dy, gamma, verbose=False)
    info_scalars = {k: v for k, v in info.items() 
                    if not isinstance(v, (list, np.ndarray))}
    results.update({f'info_{k}': v for k, v in info_scalars.items()})
    
    # Store MSE separately
    results['mse_scales'] = info['mse_scales']
    results['mse_velocity'] = info['mse_velocity']
    results['mse_magnetic'] = info['mse_magnetic']
    
    if verbose:
        print("      Computing composite metrics...")
    composite = compute_composite_metrics(U, dx, dy, gamma, conservation_initial, verbose=False)
    results.update({f'comp_{k}': v for k, v in composite.items()})
    
    return results
