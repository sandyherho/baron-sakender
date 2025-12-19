"""
JAX-accelerated integrator for 2D Ideal MHD with Divergence Cleaning.

Implements finite volume method with:
    - HLL (Harten-Lax-van Leer) approximate Riemann solver
    - RK2 (Heun's method) time integration
    - Hyperbolic-parabolic divergence cleaning (Dedner et al. 2002)
    - Adaptive time stepping based on CFL condition

The HLL solver is robust for MHD shocks and contact discontinuities,
though more diffusive than HLLD. It provides excellent stability for
turbulence simulations and shock-dominated flows.

References:
    Harten, A., Lax, P. D., & van Leer, B. (1983). On upstream
        differencing and Godunov-type schemes for hyperbolic
        conservation laws. SIAM Review, 25(1), 35-61.
    Dedner, A., et al. (2002). Hyperbolic divergence cleaning for
        the MHD equations. J. Comput. Phys., 175(2), 645-673.
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Tuple, Callable, Optional, Dict, Any, List
import numpy as np
from tqdm import tqdm


# ============================================================================
# Physical Constants and Numerical Parameters
# ============================================================================

# Divergence cleaning parameter (c_r = c_h / c_p)
# Dedner et al. recommend 0.18, but 0.5 provides more aggressive cleaning
C_R = 0.5

# Numerical floors for positivity preservation
# More conservative than 1e-10 to prevent near-vacuum instabilities
DENSITY_FLOOR = 1.0e-6
PRESSURE_FLOOR = 1.0e-6
ENERGY_FLOOR = 1.0e-8


# ============================================================================
# Primitive Variable Recovery
# ============================================================================

@jit
def _get_primitive(U: jnp.ndarray, gamma: float) -> Tuple[jnp.ndarray, ...]:
    """
    Convert conservative to primitive variables with floors.
    
    Args:
        U: Conservative variables [7, nx, ny]
        gamma: Adiabatic index
    
    Returns:
        Tuple of (rho, vx, vy, Bx, By, p, psi)
    """
    # Apply density floor
    rho = jnp.maximum(U[0], DENSITY_FLOOR)
    
    # Velocities
    vx = U[1] / rho
    vy = U[2] / rho
    
    # Magnetic field
    Bx = U[3]
    By = U[4]
    
    # Pressure from energy equation
    kinetic = 0.5 * rho * (vx**2 + vy**2)
    magnetic = 0.5 * (Bx**2 + By**2)
    internal = U[5] - kinetic - magnetic
    p = (gamma - 1) * jnp.maximum(internal, ENERGY_FLOOR)
    p = jnp.maximum(p, PRESSURE_FLOOR)
    
    # Divergence cleaning potential
    psi = U[6]
    
    return rho, vx, vy, Bx, By, p, psi


# ============================================================================
# Wave Speed Computations
# ============================================================================

@jit
def _compute_fast_speed(rho: jnp.ndarray, p: jnp.ndarray, 
                        Bx: jnp.ndarray, By: jnp.ndarray, 
                        gamma: float) -> jnp.ndarray:
    """
    Compute fast magnetosonic speed.
    
    c_f = sqrt(c_s^2 + v_A^2)
    
    where c_s = sqrt(gamma * p / rho) is sound speed
    and v_A = |B| / sqrt(rho) is Alfvén speed.
    """
    # Sound speed squared
    cs2 = gamma * p / rho
    
    # Alfvén speed squared
    B2 = Bx**2 + By**2
    va2 = B2 / rho
    
    # Fast magnetosonic speed
    cf = jnp.sqrt(cs2 + va2)
    
    return cf


@jit
def _compute_max_speed(U: jnp.ndarray, gamma: float) -> float:
    """
    Compute maximum characteristic speed for CFL condition.
    
    Returns max(|v| + c_f) over the domain.
    """
    rho, vx, vy, Bx, By, p, _ = _get_primitive(U, gamma)
    
    cf = _compute_fast_speed(rho, p, Bx, By, gamma)
    
    # Maximum speed in each direction
    max_speed_x = jnp.max(jnp.abs(vx) + cf)
    max_speed_y = jnp.max(jnp.abs(vy) + cf)
    
    return jnp.maximum(max_speed_x, max_speed_y)


# ============================================================================
# Flux Functions
# ============================================================================

@jit
def _flux_x(U: jnp.ndarray, gamma: float, c_h: float) -> jnp.ndarray:
    """
    Compute x-direction flux with divergence cleaning.
    
    F = [ρvx, ρvx² + P* - Bx², ρvxvy - BxBy, ψ, Byvx - Bxvy, 
         (E + P*)vx - Bx(v·B), ch²Bx]
    
    Note: The Bx flux is ψ (from cleaning) rather than the usual 0.
    """
    rho, vx, vy, Bx, By, p, psi = _get_primitive(U, gamma)
    
    # Total pressure (thermal + magnetic)
    pt = p + 0.5 * (Bx**2 + By**2)
    
    # v·B
    vB = vx * Bx + vy * By
    
    F = jnp.zeros_like(U)
    F = F.at[0].set(rho * vx)                              # Mass flux
    F = F.at[1].set(rho * vx**2 + pt - Bx**2)             # x-momentum
    F = F.at[2].set(rho * vx * vy - Bx * By)              # y-momentum
    F = F.at[3].set(psi)                                   # Bx (cleaning)
    F = F.at[4].set(By * vx - Bx * vy)                    # By (induction)
    F = F.at[5].set((U[5] + pt) * vx - Bx * vB)           # Energy
    F = F.at[6].set(c_h * c_h * Bx)                       # ψ flux
    
    return F


@jit
def _flux_y(U: jnp.ndarray, gamma: float, c_h: float) -> jnp.ndarray:
    """
    Compute y-direction flux with divergence cleaning.
    
    G = [ρvy, ρvxvy - BxBy, ρvy² + P* - By², Bxvy - Byvx, ψ,
         (E + P*)vy - By(v·B), ch²By]
    """
    rho, vx, vy, Bx, By, p, psi = _get_primitive(U, gamma)
    
    pt = p + 0.5 * (Bx**2 + By**2)
    vB = vx * Bx + vy * By
    
    G = jnp.zeros_like(U)
    G = G.at[0].set(rho * vy)                              # Mass flux
    G = G.at[1].set(rho * vx * vy - Bx * By)              # x-momentum
    G = G.at[2].set(rho * vy**2 + pt - By**2)             # y-momentum
    G = G.at[3].set(Bx * vy - By * vx)                    # Bx (induction)
    G = G.at[4].set(psi)                                   # By (cleaning)
    G = G.at[5].set((U[5] + pt) * vy - By * vB)           # Energy
    G = G.at[6].set(c_h * c_h * By)                       # ψ flux
    
    return G


# ============================================================================
# HLL Riemann Solver
# ============================================================================

@jit
def _hll_flux_x(UL: jnp.ndarray, UR: jnp.ndarray, 
               gamma: float, c_h: float) -> jnp.ndarray:
    """
    HLL approximate Riemann solver for x-direction.
    
    The HLL flux is:
        F^HLL = (S+ F^L - S- F^R + S+ S- (U^R - U^L)) / (S+ - S-)
    
    Wave speeds include the divergence cleaning speed ±c_h.
    
    Args:
        UL: Left state
        UR: Right state
        gamma: Adiabatic index
        c_h: Cleaning wave speed
    
    Returns:
        HLL flux at interface
    """
    # Left state primitives
    rhoL, vxL, vyL, BxL, ByL, pL, psiL = _get_primitive(UL, gamma)
    cfL = _compute_fast_speed(rhoL, pL, BxL, ByL, gamma)
    
    # Right state primitives
    rhoR, vxR, vyR, BxR, ByR, pR, psiR = _get_primitive(UR, gamma)
    cfR = _compute_fast_speed(rhoR, pR, BxR, ByR, gamma)
    
    # Wave speed estimates (include cleaning speed)
    SL = jnp.minimum(jnp.minimum(vxL - cfL, vxR - cfR), -c_h)
    SR = jnp.maximum(jnp.maximum(vxL + cfL, vxR + cfR), c_h)
    
    # Fluxes at left and right states
    FL = _flux_x(UL, gamma, c_h)
    FR = _flux_x(UR, gamma, c_h)
    
    # HLL flux (vectorized over all components)
    denom = SR - SL + 1e-10  # Avoid division by zero
    
    F_hll = jnp.where(
        SL >= 0,
        FL,
        jnp.where(
            SR <= 0,
            FR,
            (SR * FL - SL * FR + SL * SR * (UR - UL)) / denom
        )
    )
    
    return F_hll


@jit
def _hll_flux_y(UL: jnp.ndarray, UR: jnp.ndarray, 
               gamma: float, c_h: float) -> jnp.ndarray:
    """
    HLL approximate Riemann solver for y-direction.
    """
    # Left state primitives
    rhoL, vxL, vyL, BxL, ByL, pL, psiL = _get_primitive(UL, gamma)
    cfL = _compute_fast_speed(rhoL, pL, BxL, ByL, gamma)
    
    # Right state primitives
    rhoR, vxR, vyR, BxR, ByR, pR, psiR = _get_primitive(UR, gamma)
    cfR = _compute_fast_speed(rhoR, pR, BxR, ByR, gamma)
    
    # Wave speed estimates (include cleaning speed)
    SL = jnp.minimum(jnp.minimum(vyL - cfL, vyR - cfR), -c_h)
    SR = jnp.maximum(jnp.maximum(vyL + cfL, vyR + cfR), c_h)
    
    # Fluxes
    GL = _flux_y(UL, gamma, c_h)
    GR = _flux_y(UR, gamma, c_h)
    
    # HLL flux
    denom = SR - SL + 1e-10
    
    G_hll = jnp.where(
        SL >= 0,
        GL,
        jnp.where(
            SR <= 0,
            GR,
            (SR * GL - SL * GR + SL * SR * (UR - UL)) / denom
        )
    )
    
    return G_hll


# ============================================================================
# Right-Hand Side Computation
# ============================================================================

@partial(jit, static_argnums=(1, 2))
def _compute_rhs(U: jnp.ndarray, dx: float, dy: float, 
                 gamma: float, c_h: float) -> jnp.ndarray:
    """
    Compute the right-hand side of the semi-discrete equations.
    
    dU/dt = -∂F/∂x - ∂G/∂y + S
    
    where S contains the divergence cleaning source term.
    
    Uses finite volume method with HLL fluxes.
    
    Args:
        U: Conservative variables [7, nx, ny]
        dx, dy: Grid spacing (static for JIT)
        gamma: Adiabatic index
        c_h: Cleaning wave speed
    
    Returns:
        Right-hand side array [7, nx, ny]
    """
    # Get left and right states for x-direction (periodic BC via roll)
    UL_x = jnp.roll(U, 1, axis=1)   # U_{i-1,j}
    UR_x = U                         # U_{i,j}
    
    # HLL fluxes at i-1/2 interfaces
    F_left = _hll_flux_x(UL_x, UR_x, gamma, c_h)
    
    # HLL fluxes at i+1/2 interfaces
    F_right = _hll_flux_x(U, jnp.roll(U, -1, axis=1), gamma, c_h)
    
    # Get left and right states for y-direction
    UL_y = jnp.roll(U, 1, axis=2)   # U_{i,j-1}
    UR_y = U                         # U_{i,j}
    
    # HLL fluxes at j-1/2 interfaces
    G_left = _hll_flux_y(UL_y, UR_y, gamma, c_h)
    
    # HLL fluxes at j+1/2 interfaces
    G_right = _hll_flux_y(U, jnp.roll(U, -1, axis=2), gamma, c_h)
    
    # Flux divergence: -∂F/∂x - ∂G/∂y
    rhs = -(F_right - F_left) / dx - (G_right - G_left) / dy
    
    # Divergence cleaning source term: -c_h²/c_p² ψ = -c_r² c_h / Δx_min * ψ
    min_dx = min(dx, dy)
    damping_rate = C_R * C_R * c_h / min_dx
    rhs = rhs.at[6].add(-damping_rate * U[6])
    
    return rhs


# ============================================================================
# Time Stepping
# ============================================================================

@partial(jit, static_argnums=(1, 2))
def _apply_floors(U: jnp.ndarray, dx: float, dy: float, gamma: float) -> jnp.ndarray:
    """
    Apply positivity-preserving floors to conservative variables.
    
    Ensures:
        - ρ ≥ DENSITY_FLOOR
        - p ≥ PRESSURE_FLOOR (by adjusting energy)
    """
    # Apply density floor
    rho = jnp.maximum(U[0], DENSITY_FLOOR)
    U = U.at[0].set(rho)
    
    # Recompute pressure and apply floor
    vx = U[1] / rho
    vy = U[2] / rho
    Bx = U[3]
    By = U[4]
    
    kinetic = 0.5 * rho * (vx**2 + vy**2)
    magnetic = 0.5 * (Bx**2 + By**2)
    internal = U[5] - kinetic - magnetic
    
    # If pressure would be negative, fix energy
    p = (gamma - 1) * internal
    p_floor = jnp.maximum(p, PRESSURE_FLOOR)
    
    # Adjust energy where pressure was floored
    internal_floor = p_floor / (gamma - 1)
    E_new = kinetic + magnetic + internal_floor
    U = U.at[5].set(E_new)
    
    return U


@partial(jit, static_argnums=(1, 2))
def _rk2_step(U: jnp.ndarray, dx: float, dy: float, 
              dt: float, gamma: float, c_h: float) -> jnp.ndarray:
    """
    Perform one RK2 (Heun's method) time step.
    
    U^{n+1} = U^n + Δt/2 * (k1 + k2)
    
    where:
        k1 = L(U^n)
        k2 = L(U^n + Δt * k1)
    
    This is second-order accurate in time.
    
    Args:
        U: Current state [7, nx, ny]
        dx, dy: Grid spacing (static)
        dt: Time step
        gamma: Adiabatic index
        c_h: Cleaning wave speed
    
    Returns:
        Updated state after one time step
    """
    # Stage 1: Forward Euler predictor
    k1 = _compute_rhs(U, dx, dy, gamma, c_h)
    U_star = U + dt * k1
    
    # Apply floors to intermediate state
    U_star = _apply_floors(U_star, dx, dy, gamma)
    
    # Stage 2: Corrector
    k2 = _compute_rhs(U_star, dx, dy, gamma, c_h)
    
    # Final update (Heun's method)
    U_new = U + 0.5 * dt * (k1 + k2)
    
    # Apply floors to final state
    U_new = _apply_floors(U_new, dx, dy, gamma)
    
    return U_new


# ============================================================================
# Main Solver Class
# ============================================================================

class MHDIntegrator:
    """
    JAX-accelerated MHD integrator with HLL Riemann solver.
    
    Features:
        - HLL approximate Riemann solver (robust for shocks)
        - RK2 time integration (second-order accurate)
        - Dedner divergence cleaning
        - Adaptive CFL-based time stepping
        - Positivity-preserving floors
    
    Example:
        >>> from baron_sakender.core.mhd_system import MHDSystem
        >>> system = MHDSystem(nx=256, ny=256)
        >>> system.init_orszag_tang()
        >>> solver = MHDIntegrator(cfl=0.4)
        >>> result = solver.run(system, t_end=1.0, save_dt=0.1)
    
    Attributes:
        cfl: CFL number for time stepping
        use_gpu: Whether to use GPU acceleration
        backend: 'gpu' or 'cpu'
    """
    
    def __init__(
        self,
        cfl: float = 0.4,
        use_gpu: bool = False
    ):
        """
        Initialize the integrator.
        
        Args:
            cfl: CFL number (default 0.4, conservative for HLL)
            use_gpu: Whether to use GPU acceleration
        """
        self.cfl = cfl
        self.use_gpu = use_gpu
        
        # Configure JAX device
        if use_gpu:
            try:
                jax.devices('gpu')
                self.backend = 'gpu'
            except RuntimeError:
                print("Warning: GPU not available, using CPU")
                self.backend = 'cpu'
        else:
            self.backend = 'cpu'
    
    def _compute_dt(self, U: jnp.ndarray, dx: float, dy: float, gamma: float) -> float:
        """
        Compute time step based on CFL condition.
        
        Δt = CFL * min(Δx, Δy) / max(|v| + c_f)
        """
        max_speed = float(_compute_max_speed(U, gamma))
        max_speed = max(max_speed, 1e-10)  # Avoid division by zero
        
        dt = self.cfl * min(dx, dy) / max_speed
        
        return dt
    
    def run(
        self,
        system,
        t_end: float,
        save_dt: float = 0.1,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run MHD simulation.
        
        Args:
            system: MHDSystem instance with initial conditions
            t_end: Final simulation time
            save_dt: Time interval for saving snapshots
            verbose: Print progress
        
        Returns:
            Dictionary with simulation results
        """
        if not system._initialized:
            raise ValueError("MHD system not initialized. Call init_* method first.")
        
        # Extract parameters
        gamma = system.gamma
        dx = system.dx
        dy = system.dy
        
        # Initialize state (convert to JAX array)
        U = jnp.array(system.U)
        t = 0.0
        
        # Storage
        snapshots = [(0.0, np.array(U))]
        next_save = save_dt
        step_count = 0
        
        # Progress bar
        if verbose:
            pbar = tqdm(total=int(t_end / save_dt), desc="      Simulating", unit="snap")
        
        while t < t_end:
            # Compute time step
            c_h = float(_compute_max_speed(U, gamma))
            dt = self._compute_dt(U, dx, dy, gamma)
            
            # Don't overshoot
            if t + dt > t_end:
                dt = t_end - t
            
            # RK2 step
            U = _rk2_step(U, dx, dy, dt, gamma, c_h)
            t += dt
            step_count += 1
            
            # Save snapshot
            if t >= next_save or t >= t_end:
                snapshots.append((t, np.array(U)))
                next_save += save_dt
                if verbose:
                    pbar.update(1)
        
        if verbose:
            pbar.close()
        
        return {
            'snapshots': snapshots,
            'system': system,
            'params': system.params,
            't_end': t_end,
            'n_snapshots': len(snapshots),
            'total_steps': step_count,
            'backend': self.backend,
        }
    
    def run_with_diagnostics(
        self,
        system,
        t_end: float,
        save_dt: float = 0.1,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run simulation with diagnostic tracking.
        
        Args:
            system: MHDSystem instance with initial conditions
            t_end: Final simulation time
            save_dt: Time interval for saving snapshots
            verbose: Print progress
        
        Returns:
            Dictionary with simulation results and diagnostics
        """
        from .metrics import compute_conservation_metrics, compute_stability_metrics
        
        if not system._initialized:
            raise ValueError("MHD system not initialized. Call init_* method first.")
        
        # Extract parameters
        gamma = system.gamma
        dx = system.dx
        dy = system.dy
        
        # Initialize state
        U = jnp.array(system.U)
        t = 0.0
        
        # Storage
        snapshots = [(0.0, np.array(U))]
        times = [0.0]
        conservation_history = []
        stability_history = []
        next_save = save_dt
        step_count = 0
        
        # Initial diagnostics
        U_np = np.array(U)
        cons = compute_conservation_metrics(U_np, dx, dy, gamma)
        stab = compute_stability_metrics(U_np, dx, dy, gamma, dt=0.001, cfl=self.cfl)
        conservation_history.append(cons)
        stability_history.append(stab)
        
        # Progress bar
        if verbose:
            pbar = tqdm(total=int(t_end / save_dt), desc="      Simulating", unit="snap")
        
        while t < t_end:
            # Compute time step
            c_h = float(_compute_max_speed(U, gamma))
            dt = self._compute_dt(U, dx, dy, gamma)
            
            # Don't overshoot
            if t + dt > t_end:
                dt = t_end - t
            
            # RK2 step
            U = _rk2_step(U, dx, dy, dt, gamma, c_h)
            t += dt
            step_count += 1
            
            # Save snapshot and diagnostics
            if t >= next_save or t >= t_end:
                U_np = np.array(U)
                snapshots.append((t, U_np))
                times.append(t)
                
                cons = compute_conservation_metrics(U_np, dx, dy, gamma)
                stab = compute_stability_metrics(U_np, dx, dy, gamma, dt=dt, cfl=self.cfl)
                conservation_history.append(cons)
                stability_history.append(stab)
                
                next_save += save_dt
                if verbose:
                    pbar.update(1)
        
        if verbose:
            pbar.close()
        
        return {
            'snapshots': snapshots,
            'system': system,
            'params': system.params,
            'times': np.array(times),
            't_end': t_end,
            'n_snapshots': len(snapshots),
            'total_steps': step_count,
            'backend': self.backend,
            'conservation_history': conservation_history,
            'stability_history': stability_history,
        }
    
    def __repr__(self) -> str:
        return (
            f"MHDIntegrator(cfl={self.cfl}, backend='{self.backend}', solver='HLL')"
        )


# Backward compatibility alias
MHDSolver = MHDIntegrator
