"""
Comprehensive Metrics for 2D Ideal MHD Analysis.

Provides essential, robust metrics for quantifying MHD dynamics:
    - Conservation metrics (mass, momentum, energy, magnetic flux)
    - Stability metrics (CFL, Mach numbers, plasma beta)
    - Turbulence metrics (enstrophy, current, cross helicity, spectra)
    - Information metrics (entropy, mutual information, complexity)
    - Composite metrics (dynamo efficiency, conservation quality)

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

def _compute_spectrum(field: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D power spectrum from 2D field using azimuthal averaging.
    
    Args:
        field: 2D field array
        dx, dy: Grid spacing
    
    Returns:
        Tuple of (wavenumbers, spectrum)
    """
    nx, ny = field.shape
    
    # 2D FFT
    fft2d = np.fft.fft2(field)
    power2d = np.abs(fft2d)**2
    
    # Wavenumber arrays
    kx = np.fft.fftfreq(nx, dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)
    
    # Azimuthal average
    k_max = min(np.max(np.abs(kx)), np.max(np.abs(ky)))
    n_bins = min(nx, ny) // 2
    k_bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
    spectrum = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (K >= k_bins[i]) & (K < k_bins[i+1])
        if np.sum(mask) > 0:
            spectrum[i] = np.mean(power2d[mask])
    
    return k_centers, spectrum


def compute_turbulence_metrics(
    U: np.ndarray,
    dx: float,
    dy: float,
    gamma: float,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compute essential MHD turbulence metrics including spectra.
    
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
    current_density_rms = rms_current
    
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
    
    # Compute energy spectra
    k_spectrum, E_k_spectrum = _compute_spectrum(v_mag, dx, dy)
    _, E_m_spectrum = _compute_spectrum(B_mag, dx, dy)
    
    # Fit spectral indices in inertial range
    idx_inertial = (k_spectrum > 2) & (k_spectrum < k_spectrum.max() * 0.5)
    
    spectral_index_kinetic = np.nan
    spectral_index_magnetic = np.nan
    
    if np.sum(idx_inertial) > 3:
        log_k = np.log10(k_spectrum[idx_inertial] + 1e-10)
        log_Ek = np.log10(E_k_spectrum[idx_inertial] + 1e-20)
        log_Em = np.log10(E_m_spectrum[idx_inertial] + 1e-20)
        
        # Linear fit for spectral index
        valid_k = np.isfinite(log_k) & np.isfinite(log_Ek)
        if np.sum(valid_k) > 2:
            spectral_index_kinetic = np.polyfit(log_k[valid_k], log_Ek[valid_k], 1)[0]
        
        valid_m = np.isfinite(log_k) & np.isfinite(log_Em)
        if np.sum(valid_m) > 2:
            spectral_index_magnetic = np.polyfit(log_k[valid_m], log_Em[valid_m], 1)[0]
    
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
        'current_density_rms': current_density_rms,
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
        'spectral_index_kinetic': float(spectral_index_kinetic),
        'spectral_index_magnetic': float(spectral_index_magnetic),
        'k_spectrum': k_spectrum,
        'E_kinetic_spectrum': E_k_spectrum,
        'E_magnetic_spectrum': E_m_spectrum,
    }


# ============================================================================
# INFORMATION-THEORETIC METRICS
# ============================================================================

def _shannon_entropy(field: np.ndarray, n_bins: int = 64) -> float:
    """Compute Shannon entropy of a field."""
    # Normalize field to [0, 1]
    f_min, f_max = np.min(field), np.max(field)
    if f_max - f_min < 1e-10:
        return 0.0
    
    f_norm = (field - f_min) / (f_max - f_min)
    
    # Histogram
    hist, _ = np.histogram(f_norm.flatten(), bins=n_bins, range=(0, 1), density=True)
    
    # Normalize to probability
    hist = hist / (np.sum(hist) + 1e-10)
    
    # Shannon entropy
    mask = hist > 0
    entropy = -np.sum(hist[mask] * np.log2(hist[mask] + 1e-20))
    
    return float(entropy)


def _mutual_information(field1: np.ndarray, field2: np.ndarray, n_bins: int = 32) -> float:
    """Compute mutual information between two fields."""
    # Normalize fields
    f1_min, f1_max = np.min(field1), np.max(field1)
    f2_min, f2_max = np.min(field2), np.max(field2)
    
    if f1_max - f1_min < 1e-10 or f2_max - f2_min < 1e-10:
        return 0.0
    
    f1_norm = (field1 - f1_min) / (f1_max - f1_min)
    f2_norm = (field2 - f2_min) / (f2_max - f2_min)
    
    # Joint histogram
    hist2d, _, _ = np.histogram2d(
        f1_norm.flatten(), f2_norm.flatten(),
        bins=n_bins, range=[[0, 1], [0, 1]]
    )
    
    # Normalize to joint probability
    pxy = hist2d / (np.sum(hist2d) + 1e-10)
    
    # Marginals
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    # Mutual information: I(X;Y) = Σ p(x,y) log(p(x,y) / (p(x)p(y)))
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if pxy[i, j] > 1e-10 and px[i] > 1e-10 and py[j] > 1e-10:
                mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
    
    return float(max(mi, 0.0))


def _statistical_complexity(field: np.ndarray, n_bins: int = 64) -> float:
    """
    Compute statistical complexity using disequilibrium × entropy.
    """
    f_min, f_max = np.min(field), np.max(field)
    if f_max - f_min < 1e-10:
        return 0.0
    
    f_norm = (field - f_min) / (f_max - f_min)
    
    # Histogram
    hist, _ = np.histogram(f_norm.flatten(), bins=n_bins, range=(0, 1), density=True)
    p = hist / (np.sum(hist) + 1e-10)
    
    # Uniform distribution
    p_uniform = np.ones(n_bins) / n_bins
    
    # Disequilibrium (Jensen-Shannon divergence from uniform)
    m = 0.5 * (p + p_uniform)
    
    def kl_div(p1, p2):
        mask = p1 > 0
        return np.sum(p1[mask] * np.log2(p1[mask] / (p2[mask] + 1e-20) + 1e-20))
    
    js_div = 0.5 * kl_div(p, m) + 0.5 * kl_div(p_uniform, m)
    
    # Entropy
    mask = p > 0
    entropy = -np.sum(p[mask] * np.log2(p[mask] + 1e-20))
    max_entropy = np.log2(n_bins)
    normalized_entropy = entropy / max_entropy
    
    # Complexity = disequilibrium × entropy
    complexity = js_div * normalized_entropy
    
    return float(complexity)


def compute_information_metrics(
    U: np.ndarray,
    dx: float,
    dy: float,
    gamma: float,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Compute information-theoretic metrics for MHD fields.
    
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
    
    # Shannon entropy of different fields
    entropy_density = _shannon_entropy(rho)
    entropy_velocity = _shannon_entropy(v_mag)
    entropy_magnetic = _shannon_entropy(B_mag)
    entropy_vorticity = _shannon_entropy(omega_z)
    entropy_current = _shannon_entropy(Jz)
    
    # Mutual information
    mi_v_B = _mutual_information(v_mag, B_mag)
    mi_omega_J = _mutual_information(omega_z, Jz)
    mi_rho_v = _mutual_information(rho, v_mag)
    
    # Statistical complexity
    complexity_vorticity = _statistical_complexity(omega_z)
    complexity_current = _statistical_complexity(Jz)
    complexity_density = _statistical_complexity(rho)
    
    return {
        'entropy_density': entropy_density,
        'entropy_velocity': entropy_velocity,
        'entropy_magnetic': entropy_magnetic,
        'shannon_entropy_density': entropy_density,
        'shannon_entropy_vorticity': entropy_vorticity,
        'shannon_entropy_current': entropy_current,
        'MI_velocity_magnetic': mi_v_B,
        'mutual_information_v_B': mi_v_B,
        'MI_vorticity_current': mi_omega_J,
        'MI_density_velocity': mi_rho_v,
        'complexity_vorticity': complexity_vorticity,
        'complexity_current': complexity_current,
        'complexity_density': complexity_density,
        'statistical_complexity': complexity_vorticity,
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
    Compute novel composite metrics for MHD turbulence analysis.
    
    Args:
        U: Conservative variables
        dx, dy: Grid spacing
        gamma: Adiabatic index
        conservation_initial: Initial conservation metrics for error calculation
        verbose: Print progress
    
    Returns:
        Dictionary with composite metrics
    """
    # Get base metrics
    turb = compute_turbulence_metrics(U, dx, dy, gamma, verbose=False)
    cons = compute_conservation_metrics(U, dx, dy, gamma)
    
    # Dynamo efficiency index: magnetic energy growth relative to kinetic
    E_k = turb['kinetic_energy']
    E_m = turb['magnetic_energy']
    dynamo_efficiency = E_m / (E_k + E_m + 1e-10)
    
    # MHD complexity index: combines cross-helicity, spectral index, intermittency
    sigma_c = abs(turb['normalized_cross_helicity'])
    alpha_k = abs(turb.get('spectral_index_kinetic', -1.67))
    kurtosis_avg = 0.5 * (turb['vorticity_kurtosis'] + turb['current_kurtosis'])
    mhd_complexity = (1 - sigma_c) * (alpha_k / 2) * np.log10(kurtosis_avg + 1)
    
    # Coherent structure index: based on current sheet strength
    j_rms = turb['rms_current']
    j_max = turb['max_current']
    coherent_structure_index = (j_max / (j_rms + 1e-10)) / 10  # Normalized
    
    # Cascade efficiency: based on spectral indices
    alpha_k = turb.get('spectral_index_kinetic', -1.67)
    alpha_m = turb.get('spectral_index_magnetic', -1.67)
    cascade_efficiency = 0.0
    if not np.isnan(alpha_k) and not np.isnan(alpha_m):
        # Closer to -5/3 or -3/2 indicates better cascade
        cascade_efficiency = 1 - abs(alpha_k + 5/3) / 2
        cascade_efficiency = max(0, min(1, cascade_efficiency))
    
    # Alignment index: v-B alignment
    alignment_index = abs(turb['normalized_cross_helicity'])
    
    # Intermittency index: based on kurtosis deviation from Gaussian (=3)
    intermittency_index = max(0, (kurtosis_avg - 3) / 10)
    
    # Conservation quality (if initial metrics provided)
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
        'dynamo_efficiency_index': float(dynamo_efficiency),
        'mhd_complexity_index': float(mhd_complexity),
        'coherent_structure_index': float(coherent_structure_index),
        'cascade_efficiency': float(cascade_efficiency),
        'alignment_index': float(alignment_index),
        'intermittency_index': float(intermittency_index),
        'conservation_quality': float(conservation_quality),
        'mass_conservation_error': float(mass_error),
        'energy_conservation_error': float(energy_error),
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
    Compute all MHD metrics for a given state.
    
    Args:
        U: Conservative variables [7, nx, ny]
        dx, dy: Grid spacing
        gamma: Adiabatic index
        dt: Current time step
        cfl: Target CFL number
        conservation_initial: Initial conservation metrics for error calculation
        initial_metrics: Alias for conservation_initial
        verbose: Print progress
    
    Returns:
        Comprehensive dictionary of all metrics
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
    
    # Add conservation metrics with prefix
    for key, value in conservation.items():
        results[f'cons_{key}'] = value
        results[key] = value  # Also without prefix for backward compatibility
    
    # Add stability metrics with prefix
    for key, value in stability.items():
        results[f'stab_{key}'] = value
        results[key] = value
    
    # Add turbulence metrics with prefix
    for key, value in turbulence.items():
        if not isinstance(value, np.ndarray):
            results[f'turb_{key}'] = value
            results[key] = value
        else:
            results[key] = value  # Keep arrays without prefix
    
    # Add information metrics with prefix
    for key, value in information.items():
        results[f'info_{key}'] = value
        results[key] = value
    
    # Add composite metrics with prefix
    for key, value in composite.items():
        results[f'comp_{key}'] = value
        results[key] = value
    
    return results
