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
from typing import Tuple, Callable, Optional
import numpy as np


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
        >>> integrator = MHDIntegrator(system, cfl=0.4)
        >>> t, U = integrator.step()  # Single time step
        >>> integrator.evolve(t_end=1.0)  # Evolve to t=1.0
    
    Attributes:
        system: MHDSystem instance
        cfl: CFL number for time stepping
        gamma: Adiabatic index
        dx, dy: Grid spacing
        t: Current simulation time
        U: Current state array (JAX array)
        c_h: Current cleaning wave speed
    """
    
    def __init__(
        self,
        system,
        cfl: float = 0.4,
        use_gpu: bool = False
    ):
        """
        Initialize the integrator.
        
        Args:
            system: MHDSystem instance with initial conditions
            cfl: CFL number (default 0.4, conservative for HLL)
            use_gpu: Whether to use GPU acceleration
        """
        self.system = system
        self.cfl = cfl
        self.gamma = system.gamma
        self.dx = system.dx
        self.dy = system.dy
        
        # Configure JAX device
        if use_gpu:
            try:
                jax.devices('gpu')
            except RuntimeError:
                print("Warning: GPU not available, using CPU")
        
        # Initialize state (convert to JAX array)
        self.U = jnp.array(system.U)
        self.t = 0.0
        
        # Initial cleaning speed
        self.c_h = float(_compute_max_speed(self.U, self.gamma))
        
        # Track step count
        self.step_count = 0
    
    def compute_dt(self) -> float:
        """
        Compute time step based on CFL condition.
        
        Δt = CFL * min(Δx, Δy) / max(|v| + c_f)
        """
        max_speed = float(_compute_max_speed(self.U, self.gamma))
        max_speed = max(max_speed, 1e-10)  # Avoid division by zero
        
        dt = self.cfl * min(self.dx, self.dy) / max_speed
        
        return dt
    
    def step(self, dt: Optional[float] = None) -> Tuple[float, jnp.ndarray]:
        """
        Perform a single time step.
        
        Args:
            dt: Time step (computed from CFL if None)
        
        Returns:
            Tuple of (new_time, new_state)
        """
        if dt is None:
            dt = self.compute_dt()
        
        # Update cleaning speed
        self.c_h = float(_compute_max_speed(self.U, self.gamma))
        
        # RK2 step
        self.U = _rk2_step(self.U, self.dx, self.dy, dt, self.gamma, self.c_h)
        
        # Update time
        self.t += dt
        self.step_count += 1
        
        return self.t, self.U
    
    def evolve(
        self,
        t_end: float,
        callback: Optional[Callable] = None,
        callback_interval: float = 0.1,
        max_steps: int = 1000000
    ) -> Tuple[float, jnp.ndarray]:
        """
        Evolve the system to a target time.
        
        Args:
            t_end: Target final time
            callback: Optional function called periodically with (t, U)
            callback_interval: Time interval between callbacks
            max_steps: Maximum number of steps (safety limit)
        
        Returns:
            Tuple of (final_time, final_state)
        """
        next_callback = self.t + callback_interval
        steps = 0
        
        while self.t < t_end and steps < max_steps:
            # Compute dt, potentially reducing to hit t_end exactly
            dt = self.compute_dt()
            if self.t + dt > t_end:
                dt = t_end - self.t
            
            # Take step
            self.step(dt)
            steps += 1
            
            # Callback
            if callback is not None and self.t >= next_callback:
                callback(self.t, self.U)
                next_callback = self.t + callback_interval
        
        if steps >= max_steps:
            print(f"Warning: Reached maximum steps ({max_steps})")
        
        return self.t, self.U
    
    def get_state(self) -> np.ndarray:
        """Get current state as NumPy array."""
        return np.array(self.U)
    
    def set_state(self, U: np.ndarray):
        """Set state from NumPy array."""
        self.U = jnp.array(U)
    
    def get_diagnostics(self) -> dict:
        """
        Get current diagnostic quantities.
        
        Returns:
            Dictionary with diagnostic values
        """
        U = self.U
        rho, vx, vy, Bx, By, p, psi = _get_primitive(U, self.gamma)
        
        # Compute various diagnostics
        dV = self.dx * self.dy
        
        return {
            'time': self.t,
            'step_count': self.step_count,
            'dt': self.compute_dt(),
            'c_h': self.c_h,
            'total_mass': float(jnp.sum(rho) * dV),
            'total_energy': float(jnp.sum(U[5]) * dV),
            'kinetic_energy': float(0.5 * jnp.sum(rho * (vx**2 + vy**2)) * dV),
            'magnetic_energy': float(0.5 * jnp.sum(Bx**2 + By**2) * dV),
            'max_div_B': float(jnp.max(jnp.abs(
                (jnp.roll(Bx, -1, axis=0) - jnp.roll(Bx, 1, axis=0)) / (2*self.dx) +
                (jnp.roll(By, -1, axis=1) - jnp.roll(By, 1, axis=1)) / (2*self.dy)
            ))),
            'min_density': float(jnp.min(rho)),
            'min_pressure': float(jnp.min(p)),
            'max_velocity': float(jnp.max(jnp.sqrt(vx**2 + vy**2))),
            'max_psi': float(jnp.max(jnp.abs(psi))),
        }
    
    def __repr__(self) -> str:
        return (
            f"MHDIntegrator(t={self.t:.4f}, steps={self.step_count}, "
            f"cfl={self.cfl}, solver='HLL')"
        )


# Backward compatibility alias
MHDSolver = MHDIntegrator
