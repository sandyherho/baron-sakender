"""
Visualization module for 2D Ideal MHD simulations.

Provides dark-themed, publication-quality visualizations:
    - Field plots (density, magnetic pressure, current, vorticity)
    - Diagnostic time series (energy, conservation, stability)
    - GIF animations

All plots use a consistent dark theme with high contrast colors.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Set dark theme globally
plt.style.use('dark_background')

# Color schemes
COLORS = {
    'kinetic': '#00d4aa',      # Cyan-green
    'magnetic': '#ff6b9d',     # Pink
    'total': '#ffd93d',        # Yellow
    'error': '#ffd93d',        # Yellow
    'mach': '#ffd93d',         # Yellow
    'helicity': '#00d4aa',     # Cyan-green
    'divB': '#ff6b9d',         # Pink
}


def setup_dark_figure(
    nrows: int = 1, 
    ncols: int = 1, 
    figsize: Tuple[float, float] = (12, 8)
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a figure with dark background."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='#1a1a2e')
    
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)
    
    for ax in axes.flat:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('white')
    
    return fig, axes


class Animator:
    """
    Animator class for MHD simulation visualization.
    
    Provides methods to create static plots, diagnostic plots, and animations.
    
    Attributes:
        fps: Frames per second for animations
        dpi: Resolution for saved images
    """
    
    def __init__(self, fps: int = 20, dpi: int = 150):
        """
        Initialize the Animator.
        
        Args:
            fps: Frames per second for animations
            dpi: Resolution for saved images
        """
        self.fps = fps
        self.dpi = dpi
    
    def create_static_plot(
        self,
        result: Dict[str, Any],
        save_path: Path,
        title: str = "MHD Fields",
        metrics: Optional[Dict[str, float]] = None
    ) -> plt.Figure:
        """
        Create static plot of final simulation state.
        
        Args:
            result: Simulation result dictionary
            save_path: Path to save figure
            title: Plot title
            metrics: Optional metrics to display
        
        Returns:
            matplotlib Figure
        """
        # Get final state
        t_final, U = result['snapshots'][-1]
        system = result['system']
        dx = system.dx
        dy = system.dy
        gamma = system.gamma
        
        return plot_fields(U, dx, dy, gamma, t_final, title, save_path, self.dpi)
    
    def create_metrics_plot(
        self,
        times: np.ndarray,
        conservation_history: List[Dict[str, float]],
        stability_history: List[Dict[str, float]],
        save_path: Path,
        title: str = "Diagnostics"
    ) -> plt.Figure:
        """
        Create diagnostic time series plot.
        
        Args:
            times: Time array
            conservation_history: List of conservation metrics
            stability_history: List of stability metrics
            save_path: Path to save figure
            title: Plot title
        
        Returns:
            matplotlib Figure
        """
        # Combine histories
        metrics_history = []
        for cons, stab in zip(conservation_history, stability_history):
            combined = {}
            combined.update(cons)
            combined.update(stab)
            metrics_history.append(combined)
        
        return plot_diagnostics(times, metrics_history, title, save_path, self.dpi)
    
    def create_animation(
        self,
        result: Dict[str, Any],
        save_path: Path,
        title: str = "MHD Simulation",
        verbose: bool = True
    ) -> Optional[animation.FuncAnimation]:
        """
        Create animated GIF of simulation evolution.
        
        Args:
            result: Simulation result dictionary
            save_path: Path to save GIF
            title: Animation title
            verbose: Print progress
        
        Returns:
            matplotlib Animation object or None if save_path provided
        """
        system = result['system']
        dx = system.dx
        dy = system.dy
        gamma = system.gamma
        
        states = [U for t, U in result['snapshots']]
        times = np.array([t for t, U in result['snapshots']])
        
        if verbose:
            print(f"      Creating animation with {len(states)} frames...")
        
        return create_animation_impl(
            states, times, dx, dy, gamma, title, save_path, self.fps
        )


def plot_fields(
    U: np.ndarray,
    dx: float,
    dy: float,
    gamma: float,
    t: float,
    title: str = "MHD Fields",
    save_path: Optional[Path] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Plot 2D field snapshots: density, magnetic pressure, current, vorticity.
    
    Args:
        U: Conservative variables [7, nx, ny]
        dx, dy: Grid spacing
        gamma: Adiabatic index
        t: Current time
        title: Plot title
        save_path: Path to save figure
        dpi: Resolution
    
    Returns:
        matplotlib Figure
    """
    nx, ny = U.shape[1], U.shape[2]
    
    # Compute fields
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
    
    # Create coordinate arrays
    Lx = nx * dx
    Ly = ny * dy
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create figure
    fig, axes = setup_dark_figure(2, 2, figsize=(12, 10))
    
    # Plot settings
    fields = [
        (rho, r'Density $\rho$ [$\rho_0$]', 'jet'),
        (pmag, r'Magnetic Pressure $P_B$ [$P_0$]', 'hot'),
        (Jz, r'Current Density $J_z$ [$J_0$]', 'RdBu_r'),
        (omega_z, r'Vorticity $\omega_z$ [$\tau_A^{-1}$]', 'RdBu_r'),
    ]
    
    for ax, (field, label, cmap) in zip(axes.flat, fields):
        if 'RdBu' in cmap:
            # Symmetric colormap for signed quantities
            vmax = np.max(np.abs(field))
            vmin = -vmax
        else:
            vmin, vmax = np.min(field), np.max(field)
        
        im = ax.pcolormesh(X, Y, field, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax.set_xlabel(r'$x$ [$L_0$]')
        ax.set_ylabel(r'$y$ [$L_0$]')
        ax.set_title(label)
        ax.set_aspect('equal')
        
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    fig.suptitle(f'{title}\nt = {t:.2f} τ_A', fontsize=14, color='white')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(), 
                    edgecolor='none', bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_diagnostics(
    times: np.ndarray,
    metrics_history: List[Dict[str, float]],
    title: str = "Diagnostics",
    save_path: Optional[Path] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Plot diagnostic time series with simplified metrics.
    
    Args:
        times: Time array
        metrics_history: List of metrics dictionaries
        title: Plot title
        save_path: Path to save figure
        dpi: Resolution
    
    Returns:
        matplotlib Figure
    """
    fig, axes = setup_dark_figure(2, 3, figsize=(15, 8))
    
    # Extract metrics
    kinetic_energy = [m.get('kinetic_energy', 0) for m in metrics_history]
    magnetic_energy = [m.get('magnetic_energy', 0) for m in metrics_history]
    total_energy = [m.get('total_energy', 0) for m in metrics_history]
    cross_helicity = [m.get('cross_helicity', 0) for m in metrics_history]
    max_div_B = [m.get('max_div_B', 0) for m in metrics_history]
    max_sonic_mach = [m.get('max_sonic_mach', 0) for m in metrics_history]
    
    # Energy conservation error
    if total_energy[0] > 0:
        energy_error = [abs(e - total_energy[0]) / total_energy[0] * 100 
                        for e in total_energy]
    else:
        energy_error = [0] * len(total_energy)
    
    # Mass conservation error
    total_mass = [m.get('total_mass', 0) for m in metrics_history]
    if total_mass[0] > 0:
        mass_error = [abs(m - total_mass[0]) / total_mass[0] * 100 
                      for m in total_mass]
    else:
        mass_error = [0] * len(total_mass)
    
    # 1. Energy Evolution
    ax = axes[0, 0]
    ax.plot(times, kinetic_energy, color=COLORS['kinetic'], label='Kinetic', linewidth=2)
    ax.plot(times, magnetic_energy, color=COLORS['magnetic'], label='Magnetic', linewidth=2)
    ax.set_xlabel(r'Time [$\tau_A$]')
    ax.set_ylabel(r'Energy [$E_0$]')
    ax.set_title('Energy Evolution')
    ax.legend(loc='best', framealpha=0.8)
    ax.grid(True, alpha=0.3)
    
    # 2. Energy Conservation Error
    ax = axes[0, 1]
    ax.semilogy(times, [max(e, 1e-15) for e in energy_error], 
                color=COLORS['error'], linewidth=2)
    ax.set_xlabel(r'Time [$\tau_A$]')
    ax.set_ylabel('Energy Error [%]')
    ax.set_title('Energy Conservation')
    ax.grid(True, alpha=0.3)
    
    # 3. Mass Conservation Error
    ax = axes[0, 2]
    ax.semilogy(times, [max(e, 1e-15) for e in mass_error], 
                color=COLORS['kinetic'], linewidth=2)
    ax.set_xlabel(r'Time [$\tau_A$]')
    ax.set_ylabel('Mass Error [%]')
    ax.set_title('Mass Conservation')
    ax.grid(True, alpha=0.3)
    
    # 4. Divergence Constraint
    ax = axes[1, 0]
    ax.semilogy(times, [max(d, 1e-15) for d in max_div_B], 
                color=COLORS['divB'], linewidth=2)
    ax.set_xlabel(r'Time [$\tau_A$]')
    ax.set_ylabel(r'Max $|\nabla \cdot B|$')
    ax.set_title('Divergence Constraint')
    ax.grid(True, alpha=0.3)
    
    # 5. Mach Number Evolution
    ax = axes[1, 1]
    ax.plot(times, max_sonic_mach, color=COLORS['mach'], linewidth=2)
    ax.set_xlabel(r'Time [$\tau_A$]')
    ax.set_ylabel('Max Sonic Mach')
    ax.set_title('Mach Number Evolution')
    ax.grid(True, alpha=0.3)
    
    # 6. Cross Helicity Evolution
    ax = axes[1, 2]
    ax.plot(times, cross_helicity, color=COLORS['helicity'], linewidth=2)
    ax.set_xlabel(r'Time [$\tau_A$]')
    ax.set_ylabel('Cross Helicity')
    ax.set_title('Cross Helicity Evolution')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, color='white')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(),
                    edgecolor='none', bbox_inches='tight')
        plt.close(fig)
    
    return fig


def create_animation_impl(
    states: List[np.ndarray],
    times: np.ndarray,
    dx: float,
    dy: float,
    gamma: float,
    title: str = "MHD Simulation",
    save_path: Optional[Path] = None,
    fps: int = 10
) -> Optional[animation.FuncAnimation]:
    """
    Create animated GIF of simulation evolution.
    
    Args:
        states: List of state arrays [7, nx, ny]
        times: Time array
        dx, dy: Grid spacing
        gamma: Adiabatic index
        title: Animation title
        save_path: Path to save GIF
        fps: Frames per second
    
    Returns:
        matplotlib Animation object or None if save_path provided
    """
    if len(states) == 0:
        warnings.warn("No states provided for animation")
        return None
    
    nx, ny = states[0].shape[1], states[0].shape[2]
    Lx, Ly = nx * dx, ny * dy
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Setup figure
    fig, axes = setup_dark_figure(2, 2, figsize=(12, 10))
    
    # Compute global min/max for consistent colorscales
    all_rho = [U[0] for U in states]
    all_pmag = [0.5 * (U[3]**2 + U[4]**2) for U in states]
    
    rho_min, rho_max = np.min(all_rho), np.max(all_rho)
    pmag_min, pmag_max = np.min(all_pmag), np.max(all_pmag)
    
    # For Jz and omega, compute max magnitude for symmetric colorscale
    def compute_Jz(U):
        rho = U[0]
        Bx, By = U[3], U[4]
        dBy_dx = (np.roll(By, -1, axis=0) - np.roll(By, 1, axis=0)) / (2 * dx)
        dBx_dy = (np.roll(Bx, -1, axis=1) - np.roll(Bx, 1, axis=1)) / (2 * dy)
        return dBy_dx - dBx_dy
    
    def compute_omega(U):
        rho = U[0]
        vx, vy = U[1] / rho, U[2] / rho
        dvy_dx = (np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0)) / (2 * dx)
        dvx_dy = (np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1)) / (2 * dy)
        return dvy_dx - dvx_dy
    
    Jz_max = max(np.max(np.abs(compute_Jz(U))) for U in states)
    omega_max = max(np.max(np.abs(compute_omega(U))) for U in states)
    
    # Create initial plots
    U = states[0]
    rho = U[0]
    pmag = 0.5 * (U[3]**2 + U[4]**2)
    Jz = compute_Jz(U)
    omega = compute_omega(U)
    
    fields_data = [
        (rho, r'Density $\rho$', 'jet', rho_min, rho_max),
        (pmag, r'Magnetic Pressure $P_B$', 'hot', pmag_min, pmag_max),
        (Jz, r'Current $J_z$', 'RdBu_r', -Jz_max, Jz_max),
        (omega, r'Vorticity $\omega_z$', 'RdBu_r', -omega_max, omega_max),
    ]
    
    images = []
    for ax, (field, label, cmap, vmin, vmax) in zip(axes.flat, fields_data):
        im = ax.pcolormesh(X, Y, field, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax.set_xlabel(r'$x$ [$L_0$]')
        ax.set_ylabel(r'$y$ [$L_0$]')
        ax.set_title(label)
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, shrink=0.8)
        images.append(im)
    
    time_text = fig.suptitle(f'{title}\nt = {times[0]:.2f} τ_A', fontsize=14, color='white')
    plt.tight_layout()
    
    def update(frame):
        U = states[frame]
        rho = U[0]
        pmag = 0.5 * (U[3]**2 + U[4]**2)
        Jz = compute_Jz(U)
        omega = compute_omega(U)
        
        fields = [rho, pmag, Jz, omega]
        for im, field in zip(images, fields):
            im.set_array(field.ravel())
        
        time_text.set_text(f'{title}\nt = {times[frame]:.2f} τ_A')
        return images + [time_text]
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(states), interval=1000//fps, blit=False
    )
    
    if save_path:
        writer = animation.PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=100)
        plt.close(fig)
        return None
    
    return anim


def plot_final_summary(
    metrics: Dict[str, float],
    title: str = "Final State Summary",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create summary plot of final simulation state.
    
    Args:
        metrics: Final metrics dictionary
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    fig, axes = setup_dark_figure(1, 3, figsize=(15, 4))
    
    # 1. Energy partition
    ax = axes[0, 0]
    energies = [
        metrics.get('kinetic_energy', 0),
        metrics.get('magnetic_energy', 0),
        metrics.get('thermal_energy', 0),
    ]
    labels = ['Kinetic', 'Magnetic', 'Thermal']
    colors = [COLORS['kinetic'], COLORS['magnetic'], '#ff9f43']
    
    bars = ax.bar(labels, energies, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel(r'Energy [$E_0$]')
    ax.set_title('Energy Partition')
    
    # 2. Key metrics
    ax = axes[0, 1]
    key_metrics = [
        ('Mach (sonic)', metrics.get('max_sonic_mach', 0)),
        ('Mach (Alfvén)', metrics.get('max_alfven_mach', 0)),
        ('Plasma β (mean)', metrics.get('mean_beta', 0)),
        ('Energy ratio', metrics.get('energy_ratio', 0)),
    ]
    names = [m[0] for m in key_metrics]
    values = [m[1] for m in key_metrics]
    
    bars = ax.barh(names, values, color=COLORS['mach'], edgecolor='white', linewidth=1.5)
    ax.set_xlabel('Value')
    ax.set_title('Key Metrics')
    
    # 3. Conservation quality
    ax = axes[0, 2]
    cons_metrics = [
        ('Mass error', metrics.get('mass_conservation_error', 0) * 100),
        ('Energy error', metrics.get('energy_conservation_error', 0) * 100),
        ('Max ∇·B', metrics.get('max_div_B', 0)),
    ]
    names = [m[0] for m in cons_metrics]
    values = [m[1] for m in cons_metrics]
    
    bars = ax.barh(names, values, color=COLORS['error'], edgecolor='white', linewidth=1.5)
    ax.set_xlabel('Value')
    ax.set_title('Conservation Quality')
    ax.set_xscale('log')
    
    fig.suptitle(title, fontsize=14, color='white')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                    edgecolor='none', bbox_inches='tight')
    
    return fig


# Legacy function alias for backward compatibility
def create_animation(
    states: List[np.ndarray],
    times: np.ndarray,
    dx: float,
    dy: float,
    gamma: float,
    title: str = "MHD Simulation",
    save_path: Optional[Path] = None,
    fps: int = 10
) -> Optional[animation.FuncAnimation]:
    """Legacy alias for create_animation_impl."""
    return create_animation_impl(states, times, dx, dy, gamma, title, save_path, fps)
