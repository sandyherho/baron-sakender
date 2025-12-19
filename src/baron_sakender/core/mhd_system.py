"""
2D Ideal MHD System Definition with Divergence Cleaning.

The ideal magnetohydrodynamics (MHD) equations describe the dynamics
of electrically conducting fluids in magnetic fields. This module
implements the Orszag-Tang vortex and related configurations with
hyperbolic/parabolic divergence cleaning (Dedner et al. 2002).

Governing equations (conservative form with divergence cleaning):
    ∂ρ/∂t + ∇·(ρv) = 0                    (mass conservation)
    ∂(ρv)/∂t + ∇·(ρvv + P*I - BB) = 0     (momentum conservation)
    ∂B/∂t - ∇×(v×B) + ∇ψ = 0              (magnetic induction)
    ∂E/∂t + ∇·((E+P*)v - B(v·B)) = 0      (energy conservation)
    ∂ψ/∂t + c_h²∇·B = -c_h²/c_p² ψ        (divergence cleaning)

where:
    P* = p + B²/2  (total pressure)
    E = ρv²/2 + p/(γ-1) + B²/2  (total energy density)
    ψ = divergence cleaning potential
    c_h = hyperbolic cleaning speed
    c_p = parabolic damping rate

Physical units (normalized):
    Length: L₀ [m] - domain size / 2π
    Time: τ_A = L₀/v_A [s] - Alfvén crossing time
    Velocity: v_A = B₀/√(μ₀ρ₀) [m/s] - Alfvén velocity
    Density: ρ₀ [kg/m³] - reference density
    Magnetic field: B₀ [T] - reference field strength
    Pressure: P₀ = B₀²/μ₀ [Pa] - magnetic pressure unit

References:
    Orszag, S. A., & Tang, C.-M. (1979). Small-scale structure of
        two-dimensional magnetohydrodynamic turbulence.
    Stone, J. M., et al. (2008). Athena: A new code for astrophysical MHD.
    Dedner, A., et al. (2002). Hyperbolic divergence cleaning for the MHD
        equations. J. Comput. Phys., 175(2), 645-673.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field


# Physical constants
MU_0 = 4 * np.pi * 1e-7  # Vacuum permeability [H/m]


@dataclass
class MHDParams:
    """
    Container for MHD simulation parameters.
    
    Attributes:
        gamma: Adiabatic index (default: 5/3 for ideal gas)
        nx, ny: Grid resolution
        Lx, Ly: Domain size [L₀]
        rho_0: Reference density [kg/m³]
        B_0: Reference magnetic field [T]
        v_A: Alfvén velocity [m/s]
        tau_A: Alfvén time [s]
        L_0: Reference length [m]
    """
    gamma: float = 5.0 / 3.0
    nx: int = 256
    ny: int = 256
    Lx: float = 2 * np.pi
    Ly: float = 2 * np.pi
    
    # Physical reference values (SI units)
    rho_0: float = 1.0e-12      # Reference density [kg/m³] (typical solar wind)
    B_0: float = 1.0e-9         # Reference magnetic field [T] (1 nT)
    L_0: float = 1.0e6          # Reference length [m] (1000 km)
    
    # Derived quantities (computed in __post_init__)
    v_A: float = field(init=False)
    tau_A: float = field(init=False)
    P_0: float = field(init=False)
    
    def __post_init__(self):
        """Compute derived physical quantities."""
        self.v_A = self.B_0 / np.sqrt(MU_0 * self.rho_0)  # Alfvén velocity
        self.tau_A = self.L_0 / self.v_A                   # Alfvén time
        self.P_0 = self.B_0**2 / MU_0                      # Magnetic pressure
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'gamma': self.gamma,
            'nx': self.nx,
            'ny': self.ny,
            'Lx': self.Lx,
            'Ly': self.Ly,
            'rho_0': self.rho_0,
            'B_0': self.B_0,
            'L_0': self.L_0,
            'v_A': self.v_A,
            'tau_A': self.tau_A,
            'P_0': self.P_0,
        }


class MHDSystem:
    """
    2D Ideal MHD System with Divergence Cleaning.
    
    Implements various initial conditions for MHD simulations including
    the classic Orszag-Tang vortex problem. Uses hyperbolic/parabolic
    divergence cleaning (Dedner et al. 2002) to maintain ∇·B ≈ 0.
    
    Attributes:
        params: MHDParams containing simulation parameters
        U: Conservative variables array [7, nx, ny]
        x, y: Coordinate arrays
        X, Y: Meshgrid coordinates
        dx, dy: Grid spacing
    
    Conservative variables U[i]:
        U[0] = ρ       (density)
        U[1] = ρ*vx    (x-momentum)
        U[2] = ρ*vy    (y-momentum)
        U[3] = Bx      (x-magnetic field)
        U[4] = By      (y-magnetic field)
        U[5] = E       (total energy)
        U[6] = ψ       (divergence cleaning potential)
    
    Example:
        >>> system = MHDSystem(nx=256, ny=256)
        >>> system.init_orszag_tang()
        >>> print(f"Grid: {system.params.nx}x{system.params.ny}")
    """
    
    def __init__(
        self,
        nx: int = 256,
        ny: int = 256,
        gamma: float = 5.0 / 3.0,
        Lx: float = 2 * np.pi,
        Ly: float = 2 * np.pi,
        rho_0: float = 1.0e-12,
        B_0: float = 1.0e-9,
        L_0: float = 1.0e6,
    ):
        """
        Initialize MHD system.
        
        Args:
            nx: Number of grid points in x
            ny: Number of grid points in y
            gamma: Adiabatic index
            Lx: Domain size in x [L₀]
            Ly: Domain size in y [L₀]
            rho_0: Reference density [kg/m³]
            B_0: Reference magnetic field [T]
            L_0: Reference length [m]
        """
        self.params = MHDParams(
            gamma=gamma, nx=nx, ny=ny, Lx=Lx, Ly=Ly,
            rho_0=rho_0, B_0=B_0, L_0=L_0
        )
        
        # Grid setup
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.x = np.linspace(self.dx / 2, Lx - self.dx / 2, nx)
        self.y = np.linspace(self.dy / 2, Ly - self.dy / 2, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # Conservative variables: [rho, rho*vx, rho*vy, Bx, By, E, psi]
        # psi is the divergence cleaning potential (Dedner et al. 2002)
        self.U = np.zeros((7, nx, ny), dtype=np.float64)
        
        # Track initialization
        self._initialized = False
        self._init_type = None
    
    @property
    def gamma(self) -> float:
        return self.params.gamma
    
    @property
    def nx(self) -> int:
        return self.params.nx
    
    @property
    def ny(self) -> int:
        return self.params.ny
    
    def init_orszag_tang(self, amplitude: float = 1.0):
        """
        Initialize Orszag-Tang vortex problem.
        
        The Orszag-Tang vortex is a standard test problem for MHD codes
        that develops complex shock structures and turbulence.
        
        Initial conditions (normalized):
            ρ = γ²
            vx = -sin(y)
            vy = sin(x)
            Bx = -sin(y)
            By = sin(2x)
            p = γ
            ψ = 0 (divergence cleaning potential)
        
        Args:
            amplitude: Velocity/field amplitude multiplier
        """
        gamma = self.gamma
        
        # Density: ρ = γ² (ensures Mach ~ 1)
        rho = gamma**2 * np.ones((self.nx, self.ny))
        
        # Velocity field: incompressible vortex
        vx = -amplitude * np.sin(self.Y)
        vy = amplitude * np.sin(self.X)
        
        # Magnetic field: periodic with mode coupling
        Bx = -amplitude * np.sin(self.Y)
        By = amplitude * np.sin(2 * self.X)
        
        # Pressure: p = γ (gives β ~ 1)
        p = gamma * np.ones((self.nx, self.ny))
        
        # Set conservative variables
        self.U[0] = rho
        self.U[1] = rho * vx
        self.U[2] = rho * vy
        self.U[3] = Bx
        self.U[4] = By
        
        # Total energy: E = p/(γ-1) + ρv²/2 + B²/2
        kinetic = 0.5 * rho * (vx**2 + vy**2)
        magnetic = 0.5 * (Bx**2 + By**2)
        internal = p / (gamma - 1)
        self.U[5] = internal + kinetic + magnetic
        
        # Divergence cleaning potential (initially zero)
        self.U[6] = np.zeros((self.nx, self.ny))
        
        self._initialized = True
        self._init_type = 'orszag_tang'
    
    def init_strong_magnetic(self, beta: float = 0.1):
        """
        Initialize with strong magnetic field (low plasma beta).
        
        Args:
            beta: Plasma beta (thermal/magnetic pressure ratio)
        """
        gamma = self.gamma
        
        # Density
        rho = gamma**2 * np.ones((self.nx, self.ny))
        
        # Velocity field
        vx = -np.sin(self.Y)
        vy = np.sin(self.X)
        
        # Stronger magnetic field
        B_amp = np.sqrt(2 * gamma / beta)
        Bx = -B_amp * np.sin(self.Y)
        By = B_amp * np.sin(2 * self.X)
        
        # Adjusted pressure for target beta
        p = gamma * np.ones((self.nx, self.ny))
        
        self.U[0] = rho
        self.U[1] = rho * vx
        self.U[2] = rho * vy
        self.U[3] = Bx
        self.U[4] = By
        
        kinetic = 0.5 * rho * (vx**2 + vy**2)
        magnetic = 0.5 * (Bx**2 + By**2)
        internal = p / (gamma - 1)
        self.U[5] = internal + kinetic + magnetic
        
        # Divergence cleaning potential
        self.U[6] = np.zeros((self.nx, self.ny))
        
        self._initialized = True
        self._init_type = 'strong_magnetic'
    
    def init_current_sheet(self, width: float = 0.2, perturbation: float = 0.01):
        """
        Initialize Harris-like current sheet configuration.
        
        Creates a current sheet with Bx varying in y (not x!) so that
        J_z = ∂B_x/∂y ≠ 0. Includes optional perturbation to trigger
        tearing mode instability.
        
        The Harris sheet equilibrium:
            Bx = B0 * tanh((y - Ly/2) / w)
            By = perturbation * sin(2πx/Lx)
            ρ = ρ0 * (1 + 1/cosh²((y - Ly/2) / w))  (pressure balance)
            p adjusted for total pressure balance
        
        Args:
            width: Current sheet width [L₀]
            perturbation: Amplitude of By perturbation to seed instability
        """
        gamma = self.gamma
        
        # Current sheet center at y = Ly/2
        y_center = self.params.Ly / 2
        
        # Harris sheet profile: Bx = B0 * tanh((y - y_center) / width)
        # This creates J_z = dBx/dy = B0/width * sech²((y - y_center)/width)
        B0 = 1.0
        Bx = B0 * np.tanh((self.Y - y_center) / width)
        
        # Small By perturbation to trigger tearing instability
        # Using a sinusoidal perturbation in x
        By = perturbation * np.sin(2 * np.pi * self.X / self.params.Lx)
        
        # Density profile for pressure balance
        # In Harris equilibrium: p + B²/2 = const
        # Use enhanced density in the current sheet for stability
        sech2 = 1.0 / np.cosh((self.Y - y_center) / width)**2
        rho = 1.0 + 0.5 * sech2  # Enhanced density in sheet
        
        # Ensure minimum density
        rho = np.maximum(rho, 0.2)
        
        # No initial flow (quiescent state)
        vx = np.zeros((self.nx, self.ny))
        vy = np.zeros((self.nx, self.ny))
        
        # Pressure for total pressure balance
        # P_total = p + B²/2 = const = p_inf + B0²/2
        # where p_inf is pressure far from sheet (where Bx → ±B0)
        p_inf = 0.5  # Background pressure
        magnetic_pressure = 0.5 * (Bx**2 + By**2)
        p = p_inf + 0.5 * B0**2 - magnetic_pressure
        
        # Ensure minimum pressure
        p = np.maximum(p, 0.1)
        
        # Set conservative variables
        self.U[0] = rho
        self.U[1] = rho * vx
        self.U[2] = rho * vy
        self.U[3] = Bx
        self.U[4] = By
        
        kinetic = 0.5 * rho * (vx**2 + vy**2)
        magnetic = 0.5 * (Bx**2 + By**2)
        internal = p / (gamma - 1)
        self.U[5] = internal + kinetic + magnetic
        
        # Divergence cleaning potential
        self.U[6] = np.zeros((self.nx, self.ny))
        
        self._initialized = True
        self._init_type = 'current_sheet'
    
    def init_alfven_wave(self, amplitude: float = 0.1, k: int = 1):
        """
        Initialize circularly polarized Alfvén wave.
        
        Alfvén waves are exact nonlinear solutions of ideal MHD.
        For a wave propagating along x with background field B0x:
            δvy = A * sin(kx)
            δBy = A * sin(kx)  (for rightward propagating wave)
        
        The perturbations satisfy δv = ±δB/√ρ (Alfvén relation).
        
        Args:
            amplitude: Wave amplitude
            k: Wavenumber (number of wavelengths in domain)
        """
        gamma = self.gamma
        
        # Background state
        rho = np.ones((self.nx, self.ny))
        B0 = 1.0
        
        # Alfvén wave perturbation
        phase = k * self.X
        vx = np.zeros((self.nx, self.ny))
        vy = amplitude * np.sin(phase)
        
        # Background Bx + perturbation By
        # For Alfvén wave: δBy = δvy * √ρ (in normalized units with vA = 1)
        Bx = B0 * np.ones((self.nx, self.ny))
        By = amplitude * np.sin(phase)
        
        # Pressure (uniform background)
        p = 0.1 * np.ones((self.nx, self.ny))
        
        self.U[0] = rho
        self.U[1] = rho * vx
        self.U[2] = rho * vy
        self.U[3] = Bx
        self.U[4] = By
        
        kinetic = 0.5 * rho * (vx**2 + vy**2)
        magnetic = 0.5 * (Bx**2 + By**2)
        internal = p / (gamma - 1)
        self.U[5] = internal + kinetic + magnetic
        
        # Divergence cleaning potential
        self.U[6] = np.zeros((self.nx, self.ny))
        
        self._initialized = True
        self._init_type = 'alfven_wave'
    
    def get_primitive(self, U: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
        """
        Convert conservative to primitive variables.
        
        Args:
            U: Conservative variables (uses self.U if None)
        
        Returns:
            Tuple of (rho, vx, vy, Bx, By, p)
        """
        if U is None:
            U = self.U
        
        rho = U[0]
        vx = U[1] / rho
        vy = U[2] / rho
        Bx = U[3]
        By = U[4]
        
        # Pressure from energy
        kinetic = 0.5 * rho * (vx**2 + vy**2)
        magnetic = 0.5 * (Bx**2 + By**2)
        p = (self.gamma - 1) * (U[5] - kinetic - magnetic)
        p = np.maximum(p, 1e-10)  # Pressure floor
        
        return rho, vx, vy, Bx, By, p
    
    def get_total_pressure(self, U: Optional[np.ndarray] = None) -> np.ndarray:
        """Get total pressure P* = p + B²/2."""
        rho, vx, vy, Bx, By, p = self.get_primitive(U)
        return p + 0.5 * (Bx**2 + By**2)
    
    def describe(self) -> str:
        """Return detailed description of the system."""
        params = self.params
        return f"""
2D Ideal MHD System with Divergence Cleaning
=============================================
Grid Resolution: {params.nx} × {params.ny}
Domain Size: [{params.Lx:.4f}] × [{params.Ly:.4f}] L₀
Grid Spacing: dx={self.dx:.6f}, dy={self.dy:.6f}

Physical Parameters:
  γ (adiabatic index) = {params.gamma:.4f}
  
Reference Units (SI):
  L₀ = {params.L_0:.2e} m
  ρ₀ = {params.rho_0:.2e} kg/m³
  B₀ = {params.B_0:.2e} T

Derived Quantities:
  v_A = {params.v_A:.2e} m/s (Alfvén velocity)
  τ_A = {params.tau_A:.2e} s (Alfvén time)
  P₀ = {params.P_0:.2e} Pa (magnetic pressure)

Initialization: {self._init_type if self._initialized else 'Not initialized'}

Governing Equations (with divergence cleaning):
  ∂ρ/∂t + ∇·(ρv) = 0
  ∂(ρv)/∂t + ∇·(ρvv + P*I - BB) = 0
  ∂B/∂t - ∇×(v×B) + ∇ψ = 0
  ∂E/∂t + ∇·((E+P*)v - B(v·B)) = 0
  ∂ψ/∂t + c_h²∇·B = -c_h²/c_p² ψ

Divergence Cleaning (Dedner et al. 2002):
  Hyperbolic-parabolic cleaning with c_h = max wave speed
"""
    
    def __repr__(self) -> str:
        return (
            f"MHDSystem(nx={self.nx}, ny={self.ny}, "
            f"gamma={self.gamma:.2f}, init={self._init_type})"
        )
    
    def __str__(self) -> str:
        return self.__repr__()
