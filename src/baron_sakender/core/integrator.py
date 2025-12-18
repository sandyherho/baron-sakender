"""
JAX-Accelerated Integrator for 2D Ideal MHD with Divergence Cleaning.

Implements high-performance numerical integration using JAX with
automatic GPU/CPU backend selection. Uses HLL Riemann solver for
robust shock capturing with RK2 time stepping and hyperbolic/parabolic
divergence cleaning (Dedner et al. 2002).

Features:
    - JIT-compiled flux computations
    - Automatic GPU/CPU backend selection
    - HLL approximate Riemann solver
    - Second-order Runge-Kutta time integration
    - Adaptive CFL-based time stepping
    - Hyperbolic/parabolic divergence cleaning for ∇·B = 0

The divergence cleaning adds a scalar field ψ that advects divergence
errors at speed c_h and damps them with rate c_h/c_p:
    ∂B/∂t + ... + ∇ψ = 0
    ∂ψ/∂t + c_h²∇·B = -(c_h/c_p)² ψ

References:
    Dedner, A., et al. (2002). Hyperbolic divergence cleaning for the
        MHD equations. J. Comput. Phys., 175(2), 645-673.
    Harten, A., Lax, P. D., & van Leer, B. (1983). On upstream
        differencing and Godunov-type schemes for hyperbolic
        conservation laws.
"""

import os
from typing import Dict, Any, Optional, List, Tuple
from functools import partial

import numpy as np
from tqdm import tqdm

# Configure JAX before import
def _configure_jax(use_gpu: bool = False):
    """Configure JAX backend before first use."""
    if use_gpu:
        os.environ['JAX_PLATFORM_NAME'] = 'gpu'
    else:
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'

# Import JAX
import jax
import jax.numpy as jnp
from jax import jit, lax

from .mhd_system import MHDSystem


# ============================================================================
# DIVERGENCE CLEANING PARAMETERS (Dedner et al. 2002)
# ============================================================================

# c_r = c_h / c_p ratio (controls damping rate)
# Higher values = faster damping of divergence errors
# Dedner et al. recommend c_r ~ 0.18 but more aggressive values work better
# in practice for finite volume schemes
C_R = 0.5  # More aggressive than original 0.18


# ============================================================================
# JAX-COMPILED MHD KERNELS WITH DIVERGENCE CLEANING
# ============================================================================

@jit
def _primitive_from_conservative(U: jnp.ndarray, gamma: float) -> Tuple:
    """
    Convert conservative to primitive variables (JIT-compiled).
    
    Args:
        U: Conservative variables [7, nx, ny]
        gamma: Adiabatic index
    
    Returns:
        Tuple of (rho, vx, vy, Bx, By, p, psi)
    """
    rho = U[0]
    vx = U[1] / rho
    vy = U[2] / rho
    Bx = U[3]
    By = U[4]
    psi = U[6]
    
    kinetic = 0.5 * rho * (vx**2 + vy**2)
    magnetic = 0.5 * (Bx**2 + By**2)
    p = (gamma - 1) * (U[5] - kinetic - magnetic)
    p = jnp.maximum(p, 1e-10)
    
    return rho, vx, vy, Bx, By, p, psi


@jit
def _max_wave_speed(U: jnp.ndarray, gamma: float) -> Tuple[float, float, float]:
    """
    Compute maximum wave speeds for CFL condition.
    
    Returns max(|vx| + cf), max(|vy| + cf), and c_h for divergence cleaning.
    """
    rho, vx, vy, Bx, By, p, psi = _primitive_from_conservative(U, gamma)
    
    # Sound speed
    cs2 = gamma * p / rho
    
    # Alfvén speeds
    ca2 = (Bx**2 + By**2) / rho
    
    # Fast magnetosonic speed (approximate)
    cf = jnp.sqrt(cs2 + ca2)
    
    sx = jnp.max(jnp.abs(vx) + cf)
    sy = jnp.max(jnp.abs(vy) + cf)
    
    # Divergence cleaning speed (max fast speed)
    c_h = jnp.maximum(sx, sy)
    
    return sx, sy, c_h


@jit
def _flux_x(U: jnp.ndarray, gamma: float, c_h: float) -> jnp.ndarray:
    """
    Compute x-direction flux with divergence cleaning (JIT-compiled).
    
    F = [ρvx, ρvx² + P* - Bx², ρvxvy - BxBy, ψ, Byvx - Bxvy, (E+P*)vx - Bx(v·B), c_h²Bx]
    
    The flux for Bx now includes ψ, and there's a flux for ψ proportional to Bx.
    """
    rho, vx, vy, Bx, By, p, psi = _primitive_from_conservative(U, gamma)
    pt = p + 0.5 * (Bx**2 + By**2)  # Total pressure
    
    F = jnp.zeros_like(U)
    F = F.at[0].set(rho * vx)
    F = F.at[1].set(rho * vx**2 + pt - Bx**2)
    F = F.at[2].set(rho * vx * vy - Bx * By)
    F = F.at[3].set(psi)  # Divergence cleaning: flux for Bx is ψ
    F = F.at[4].set(By * vx - Bx * vy)
    F = F.at[5].set((U[5] + pt) * vx - Bx * (vx * Bx + vy * By))
    F = F.at[6].set(c_h * c_h * Bx)  # Flux for ψ is c_h² * Bx
    
    return F


@jit
def _flux_y(U: jnp.ndarray, gamma: float, c_h: float) -> jnp.ndarray:
    """
    Compute y-direction flux with divergence cleaning (JIT-compiled).
    
    G = [ρvy, ρvxvy - BxBy, ρvy² + P* - By², Bxvy - Byvx, ψ, (E+P*)vy - By(v·B), c_h²By]
    
    The flux for By now includes ψ, and there's a flux for ψ proportional to By.
    """
    rho, vx, vy, Bx, By, p, psi = _primitive_from_conservative(U, gamma)
    pt = p + 0.5 * (Bx**2 + By**2)
    
    G = jnp.zeros_like(U)
    G = G.at[0].set(rho * vy)
    G = G.at[1].set(rho * vx * vy - Bx * By)
    G = G.at[2].set(rho * vy**2 + pt - By**2)
    G = G.at[3].set(Bx * vy - By * vx)
    G = G.at[4].set(psi)  # Divergence cleaning: flux for By is ψ
    G = G.at[5].set((U[5] + pt) * vy - By * (vx * Bx + vy * By))
    G = G.at[6].set(c_h * c_h * By)  # Flux for ψ is c_h² * By
    
    return G


@jit
def _hll_flux_x(UL: jnp.ndarray, UR: jnp.ndarray, gamma: float, c_h: float) -> jnp.ndarray:
    """
    HLL approximate Riemann solver for x-direction with divergence cleaning.
    
    The HLL solver approximates the solution with two waves:
    F_HLL = (S_R*F_L - S_L*F_R + S_L*S_R*(U_R - U_L)) / (S_R - S_L)
    
    Wave speed estimates include the divergence cleaning speed c_h.
    """
    rhoL, vxL, vyL, BxL, ByL, pL, psiL = _primitive_from_conservative(UL, gamma)
    rhoR, vxR, vyR, BxR, ByR, pR, psiR = _primitive_from_conservative(UR, gamma)
    
    # Fast magnetosonic speeds
    csL = jnp.sqrt(gamma * pL / rhoL)
    csR = jnp.sqrt(gamma * pR / rhoR)
    caL = jnp.sqrt((BxL**2 + ByL**2) / rhoL)
    caR = jnp.sqrt((BxR**2 + ByR**2) / rhoR)
    cfL = jnp.sqrt(csL**2 + caL**2)
    cfR = jnp.sqrt(csR**2 + caR**2)
    
    # Wave speed estimates (include c_h for divergence cleaning)
    SL = jnp.minimum(jnp.minimum(vxL - cfL, vxR - cfR), -c_h)
    SR = jnp.maximum(jnp.maximum(vxL + cfL, vxR + cfR), c_h)
    
    # Fluxes
    FL = _flux_x(UL, gamma, c_h)
    FR = _flux_x(UR, gamma, c_h)
    
    # HLL flux
    F_hll = jnp.where(
        SL >= 0, FL,
        jnp.where(
            SR <= 0, FR,
            (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL + 1e-10)
        )
    )
    
    return F_hll


@jit
def _hll_flux_y(UL: jnp.ndarray, UR: jnp.ndarray, gamma: float, c_h: float) -> jnp.ndarray:
    """
    HLL approximate Riemann solver for y-direction with divergence cleaning.
    """
    rhoL, vxL, vyL, BxL, ByL, pL, psiL = _primitive_from_conservative(UL, gamma)
    rhoR, vxR, vyR, BxR, ByR, pR, psiR = _primitive_from_conservative(UR, gamma)
    
    csL = jnp.sqrt(gamma * pL / rhoL)
    csR = jnp.sqrt(gamma * pR / rhoR)
    caL = jnp.sqrt((BxL**2 + ByL**2) / rhoL)
    caR = jnp.sqrt((BxR**2 + ByR**2) / rhoR)
    cfL = jnp.sqrt(csL**2 + caL**2)
    cfR = jnp.sqrt(csR**2 + caR**2)
    
    # Include c_h in wave speed bounds
    SL = jnp.minimum(jnp.minimum(vyL - cfL, vyR - cfR), -c_h)
    SR = jnp.maximum(jnp.maximum(vyL + cfL, vyR + cfR), c_h)
    
    GL = _flux_y(UL, gamma, c_h)
    GR = _flux_y(UR, gamma, c_h)
    
    G_hll = jnp.where(
        SL >= 0, GL,
        jnp.where(
            SR <= 0, GR,
            (SR * GL - SL * GR + SL * SR * (UR - UL)) / (SR - SL + 1e-10)
        )
    )
    
    return G_hll


@partial(jit, static_argnums=(1, 2))
def _compute_rhs(U: jnp.ndarray, dx: float, dy: float, gamma: float, c_h: float) -> jnp.ndarray:
    """
    Compute right-hand side of MHD equations using finite volume method.
    
    dU/dt = -1/dx * (F_{i+1/2} - F_{i-1/2}) - 1/dy * (G_{j+1/2} - G_{j-1/2}) + S
    
    where S contains the source term for divergence cleaning damping.
    
    Divergence cleaning (Dedner et al. 2002):
        ∂ψ/∂t + c_h²∇·B = -(c_h/c_p)² ψ
        
    With c_p = c_h/c_r where c_r = C_R (user parameter), the damping becomes:
        source_ψ = -c_r² * ψ * c_h / min(dx, dy)
        
    This scales properly with grid resolution for effective cleaning.
    """
    # X-direction fluxes at cell interfaces
    UL_x = jnp.roll(U, 1, axis=1)  # U_{i-1}
    F_minus = _hll_flux_x(UL_x, U, gamma, c_h)  # F_{i-1/2}
    F_plus = jnp.roll(F_minus, -1, axis=1)  # F_{i+1/2}
    
    # Y-direction fluxes at cell interfaces
    UL_y = jnp.roll(U, 1, axis=2)  # U_{j-1}
    G_minus = _hll_flux_y(UL_y, U, gamma, c_h)  # G_{j-1/2}
    G_plus = jnp.roll(G_minus, -1, axis=2)  # G_{j+1/2}
    
    # Compute RHS from fluxes
    rhs = -(F_plus - F_minus) / dx - (G_plus - G_minus) / dy
    
    # Add source term for divergence cleaning (parabolic damping)
    # Following Dedner et al., the damping term is -(c_h/c_p)² ψ
    # With c_p = c_h/C_R, this becomes -C_R² * ψ
    # We scale by c_h/min(dx,dy) to ensure proper damping at all resolutions
    min_dx = min(dx, dy)  # Use Python min since dx, dy are static
    damping_rate = C_R * C_R * c_h / min_dx
    rhs = rhs.at[6].add(-damping_rate * U[6])
    
    return rhs


@partial(jit, static_argnums=(1, 2))
def _rk2_step(U: jnp.ndarray, dx: float, dy: float, dt: float, gamma: float, c_h: float) -> jnp.ndarray:
    """
    Second-order Runge-Kutta (Heun's method) time step with divergence cleaning.
    
    k1 = f(U^n)
    k2 = f(U^n + dt*k1)
    U^{n+1} = U^n + dt/2 * (k1 + k2)
    """
    k1 = _compute_rhs(U, dx, dy, gamma, c_h)
    U_star = U + dt * k1
    
    # Apply positivity preservation
    U_star = U_star.at[0].set(jnp.maximum(U_star[0], 1e-10))  # Density floor
    
    k2 = _compute_rhs(U_star, dx, dy, gamma, c_h)
    U_new = U + 0.5 * dt * (k1 + k2)
    
    # Apply floors
    U_new = U_new.at[0].set(jnp.maximum(U_new[0], 1e-10))  # Density
    
    return U_new


# ============================================================================
# SOLVER CLASS
# ============================================================================

class MHDSolver:
    """
    JAX-Accelerated Solver for 2D Ideal MHD with Divergence Cleaning.
    
    Provides high-performance numerical integration with automatic
    GPU/CPU backend selection, adaptive time stepping, and comprehensive
    diagnostics. Uses hyperbolic/parabolic divergence cleaning
    (Dedner et al. 2002) to maintain ∇·B ≈ 0.
    
    Attributes:
        cfl: CFL number for time step selection
        use_gpu: Whether GPU acceleration is enabled
        backend: JAX backend string ('cpu' or 'gpu')
    
    Example:
        >>> solver = MHDSolver(cfl=0.4, use_gpu=False)
        >>> system = MHDSystem(nx=256, ny=256)
        >>> system.init_orszag_tang()
        >>> result = solver.run(system, t_end=3.0, save_dt=0.05)
    """
    
    def __init__(self, cfl: float = 0.4, use_gpu: bool = False):
        """
        Initialize solver.
        
        Args:
            cfl: CFL number for stability (default: 0.4)
            use_gpu: Use GPU acceleration if available (default: False)
        """
        self.cfl = cfl
        self.use_gpu = use_gpu
        
        # Configure JAX backend
        _configure_jax(use_gpu)
        
        self.backend = jax.default_backend()
        self.devices = jax.devices()
    
    def run(
        self,
        system: MHDSystem,
        t_end: float,
        save_dt: float = 0.05,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run MHD simulation.
        
        Args:
            system: MHDSystem instance with initial conditions
            t_end: End time [τ_A]
            save_dt: Time interval for saving snapshots
            verbose: Print progress information
        
        Returns:
            Dictionary with simulation results
        """
        if not system._initialized:
            raise ValueError("MHD system not initialized. Call init_* method first.")
        
        dx = system.dx
        dy = system.dy
        gamma = system.gamma
        
        # Convert to JAX arrays
        U = jnp.array(system.U)
        
        if verbose:
            print(f"      Starting simulation to t={t_end:.2f}")
            print(f"      JAX backend: {self.backend}")
            print(f"      Divergence cleaning: enabled (c_r={C_R:.2f})")
            print("      Compiling JAX kernels...")
        
        # Warmup JIT compilation
        sx, sy, c_h = _max_wave_speed(U, gamma)
        dt_test = min(self.cfl * dx / float(sx), self.cfl * dy / float(sy))
        _ = _rk2_step(U, dx, dy, dt_test, gamma, float(c_h))
        jax.block_until_ready(_)
        
        if verbose:
            print("      Running integration...")
        
        # Time stepping loop
        t = 0.0
        step = 0
        snapshots = [(0.0, np.array(U))]
        next_save = save_dt
        
        # Estimate total steps for progress bar
        sx_init, sy_init, c_h_init = _max_wave_speed(U, gamma)
        dt_est = min(self.cfl * dx / float(sx_init), self.cfl * dy / float(sy_init))
        total_steps_est = int(t_end / dt_est) + 1
        
        pbar = tqdm(
            total=total_steps_est,
            desc="      Integrating",
            disable=not verbose,
            ncols=70,
            leave=True
        )
        
        while t < t_end:
            # Compute time step from CFL condition
            sx, sy, c_h = _max_wave_speed(U, gamma)
            dt = min(
                self.cfl * dx / float(sx),
                self.cfl * dy / float(sy),
                t_end - t
            )
            
            # Single RK2 step with divergence cleaning
            U = _rk2_step(U, dx, dy, dt, gamma, float(c_h))
            jax.block_until_ready(U)
            
            t += dt
            step += 1
            pbar.update(1)
            
            # Save snapshot
            if t >= next_save or abs(t - t_end) < 1e-10:
                snapshots.append((t, np.array(U)))
                next_save += save_dt
        
        pbar.close()
        
        if verbose:
            print(f"      ✓ Simulation complete: {len(snapshots)} snapshots, {step} steps")
        
        return {
            'snapshots': snapshots,
            'system': system,
            'params': system.params,
            't_end': t_end,
            'total_steps': step,
            'n_snapshots': len(snapshots),
            'backend': self.backend,
        }
    
    def run_with_diagnostics(
        self,
        system: MHDSystem,
        t_end: float,
        save_dt: float = 0.05,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run simulation with full diagnostic tracking.
        
        This method computes and stores conservation metrics,
        stability metrics, and turbulence diagnostics at each snapshot.
        
        Args:
            system: MHDSystem instance
            t_end: End time
            save_dt: Save interval
            verbose: Print progress
        
        Returns:
            Dictionary with simulation results and diagnostics
        """
        from .metrics import (
            compute_conservation_metrics,
            compute_stability_metrics,
            compute_diagnostics
        )
        
        if not system._initialized:
            raise ValueError("MHD system not initialized.")
        
        dx = system.dx
        dy = system.dy
        gamma = system.gamma
        
        U = jnp.array(system.U)
        
        if verbose:
            print(f"      Starting diagnostic run to t={t_end:.2f}")
            print(f"      JAX backend: {self.backend}")
            print(f"      Divergence cleaning: enabled (c_r={C_R:.2f})")
            print("      Compiling JAX kernels...")
        
        # Warmup
        sx, sy, c_h = _max_wave_speed(U, gamma)
        dt_test = min(self.cfl * dx / float(sx), self.cfl * dy / float(sy))
        _ = _rk2_step(U, dx, dy, dt_test, gamma, float(c_h))
        jax.block_until_ready(_)
        
        if verbose:
            print("      Running integration with diagnostics...")
        
        # Storage
        t = 0.0
        step = 0
        snapshots = [(0.0, np.array(U))]
        next_save = save_dt
        
        # Diagnostic time series
        times = [0.0]
        conservation_history = []
        stability_history = []
        
        # Initial diagnostics
        U_np = np.array(U)
        rho, pmag, Jz, omega = compute_diagnostics(U_np, dx, dy, gamma)
        
        cons0 = compute_conservation_metrics(U_np, dx, dy, gamma)
        conservation_history.append(cons0)
        
        stab0 = compute_stability_metrics(U_np, dx, dy, gamma, 0.0, self.cfl)
        stability_history.append(stab0)
        
        # Estimate steps
        sx_init, sy_init, c_h_init = _max_wave_speed(U, gamma)
        dt_est = min(self.cfl * dx / float(sx_init), self.cfl * dy / float(sy_init))
        total_steps_est = int(t_end / dt_est) + 1
        
        pbar = tqdm(
            total=total_steps_est,
            desc="      Integrating",
            disable=not verbose,
            ncols=70
        )
        
        while t < t_end:
            sx, sy, c_h = _max_wave_speed(U, gamma)
            dt = min(
                self.cfl * dx / float(sx),
                self.cfl * dy / float(sy),
                t_end - t
            )
            
            # RK2 step with divergence cleaning
            U = _rk2_step(U, dx, dy, dt, gamma, float(c_h))
            jax.block_until_ready(U)
            
            t += dt
            step += 1
            pbar.update(1)
            
            if t >= next_save or abs(t - t_end) < 1e-10:
                U_np = np.array(U)
                snapshots.append((t, U_np))
                times.append(t)
                
                # Conservation metrics
                cons = compute_conservation_metrics(U_np, dx, dy, gamma)
                conservation_history.append(cons)
                
                # Stability metrics
                stab = compute_stability_metrics(U_np, dx, dy, gamma, dt, self.cfl)
                stability_history.append(stab)
                
                next_save += save_dt
        
        pbar.close()
        
        if verbose:
            print(f"      ✓ Complete: {len(snapshots)} snapshots, {step} steps")
        
        return {
            'snapshots': snapshots,
            'times': np.array(times),
            'system': system,
            'params': system.params,
            't_end': t_end,
            'total_steps': step,
            'n_snapshots': len(snapshots),
            'backend': self.backend,
            'conservation_history': conservation_history,
            'stability_history': stability_history,
        }
