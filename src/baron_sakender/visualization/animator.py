"""
Visualization module for 2D Ideal MHD Simulations.

Provides the Animator class for creating:
    - Static field plots (density, magnetic pressure, current, vorticity)
    - Diagnostic time series (energy, conservation, Mach numbers)
    - GIF animations of field evolution

Uses dark theme with publication-quality output.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from typing import Dict, List, Optional, Any, Tuple


# Set dark style globally
plt.style.use('dark_background')


class Animator:
    """
    Animation and visualization class for MHD simulations.
    
    Attributes:
        fps: Frames per second for animations
        dpi: Resolution for saved figures
    """
    
    def __init__(self, fps: int = 10, dpi: int = 150):
        """
        Initialize the Animator.
        
        Args:
            fps: Frames per second for GIF animations
            dpi: Dots per inch for saved figures
        """
        self.fps = fps
        self.dpi = dpi
        
        # Color scheme
        self.colors = {
            'kinetic': '#00D4AA',      # Cyan-green
            'magnetic': '#FF6B9D',     # Pink
            'energy_error': '#FFD93D', # Yellow
            'mass_error': '#00D4AA',   # Cyan
            'div_B': '#FF6B9D',        # Pink
            'mach': '#FFD93D',         # Yellow
            'cross_helicity': '#00D4AA', # Cyan
            'grid': '#2a2a3e',         # Dark grid
            'text': '#ffffff',         # White text
        }
        
        # Figure background
        self.fig_facecolor = '#1a1a2e'
        self.ax_facecolor = '#16213e'
    
    def create_static_plot(
        self,
        result: Dict[str, Any],
        filename: str,
        title: str = "MHD Simulation",
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Create a static 2x2 plot of MHD fields.
        
        Panels:
            - Density
            - Magnetic Pressure
            - Current Density (Jz)
            - Vorticity (omega_z)
        
        Args:
            result: Simulation result dictionary with 'U', 'x', 'y', 't', 'params'
            filename: Output file path
            title: Plot title
            metrics: Optional metrics dictionary (unused, for API compatibility)
        """
        U = result['U']
        x = result['x']
        y = result['y']
        t = result['t']
        params = result['params']
        
        gamma = params.get('gamma', 5/3)
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        dy = y[1] - y[0] if len(y) > 1 else 1.0
        
        # Extract fields
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
        
        # Vorticity omega_z = dvy/dx - dvx/dy
        dvy_dx = (np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0)) / (2 * dx)
        dvx_dy = (np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1)) / (2 * dy)
        omega_z = dvy_dx - dvx_dy
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor=self.fig_facecolor)
        fig.suptitle(f"{title}\nt = {t:.2f} τ_A", fontsize=14, color=self.colors['text'])
        
        # Plot data
        fields = [
            (rho, r'Density $\rho$ [$\rho_0$]', 'jet'),
            (pmag, r'Magnetic Pressure $P_B$ [$P_0$]', 'hot'),
            (Jz, r'Current Density $J_z$ [$J_0$]', 'RdBu_r'),
            (omega_z, r'Vorticity $\omega_z$ [$\tau_A^{-1}$]', 'RdBu_r'),
        ]
        
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        for ax, (field, label, cmap) in zip(axes.flat, fields):
            ax.set_facecolor(self.ax_facecolor)
            
            # Symmetric colorbar for signed quantities
            if 'RdBu' in cmap:
                vmax = np.max(np.abs(field))
                vmin = -vmax
            else:
                vmin, vmax = np.min(field), np.max(field)
            
            im = ax.pcolormesh(X, Y, field, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
            ax.set_xlabel(r'x [$L_0$]', color=self.colors['text'])
            ax.set_ylabel(r'y [$L_0$]', color=self.colors['text'])
            ax.set_title(label, color=self.colors['text'])
            ax.set_aspect('equal')
            ax.tick_params(colors=self.colors['text'])
            
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.yaxis.set_tick_params(color=self.colors['text'])
            plt.setp(cbar.ax.get_yticklabels(), color=self.colors['text'])
        
        plt.tight_layout()
        plt.savefig(filename, dpi=self.dpi, facecolor=self.fig_facecolor, 
                    bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    
    def create_metrics_plot(
        self,
        times: np.ndarray,
        conservation_history: List[Dict[str, float]],
        stability_history: List[Dict[str, float]],
        filename: str,
        title: str = "Diagnostics"
    ) -> None:
        """
        Create diagnostic time series plot.
        
        Panels (2x3):
            - Energy Evolution (kinetic & magnetic)
            - Energy Conservation Error
            - Mass Conservation Error
            - Divergence Constraint (max |div B|)
            - Mach Number Evolution
            - Cross Helicity Evolution
        
        Args:
            times: Array of time points
            conservation_history: List of conservation metric dicts
            stability_history: List of stability metric dicts
            filename: Output file path
            title: Plot title
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=self.fig_facecolor)
        fig.suptitle(f"{title}", fontsize=14, color=self.colors['text'])
        
        times = np.array(times)
        
        # Helper to extract time series
        def extract(history, key, default=0.0):
            return np.array([h.get(key, default) for h in history])
        
        # === Panel 1: Energy Evolution ===
        ax = axes[0, 0]
        ax.set_facecolor(self.ax_facecolor)
        
        E_k = extract(conservation_history, 'kinetic_energy')
        E_m = extract(conservation_history, 'magnetic_energy')
        
        ax.plot(times, E_k, color=self.colors['kinetic'], label='Kinetic', linewidth=2)
        ax.plot(times, E_m, color=self.colors['magnetic'], label='Magnetic', linewidth=2)
        ax.set_xlabel(r'Time [$\tau_A$]', color=self.colors['text'])
        ax.set_ylabel(r'Energy [$E_0$]', color=self.colors['text'])
        ax.set_title('Energy Evolution', color=self.colors['text'])
        ax.legend(facecolor=self.ax_facecolor, edgecolor='gray')
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.tick_params(colors=self.colors['text'])
        
        # === Panel 2: Energy Conservation ===
        ax = axes[0, 1]
        ax.set_facecolor(self.ax_facecolor)
        
        E_total = extract(conservation_history, 'total_energy')
        if len(E_total) > 0 and E_total[0] != 0:
            E_error = np.abs(E_total - E_total[0]) / E_total[0] * 100
        else:
            E_error = np.zeros_like(times)
        
        ax.semilogy(times, E_error + 1e-16, color=self.colors['energy_error'], linewidth=2)
        ax.set_xlabel(r'Time [$\tau_A$]', color=self.colors['text'])
        ax.set_ylabel('Energy Error [%]', color=self.colors['text'])
        ax.set_title('Energy Conservation', color=self.colors['text'])
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.tick_params(colors=self.colors['text'])
        
        # === Panel 3: Mass Conservation ===
        ax = axes[0, 2]
        ax.set_facecolor(self.ax_facecolor)
        
        M_total = extract(conservation_history, 'total_mass')
        if len(M_total) > 0 and M_total[0] != 0:
            M_error = np.abs(M_total - M_total[0]) / M_total[0] * 100
        else:
            M_error = np.zeros_like(times)
        
        ax.semilogy(times, M_error + 1e-16, color=self.colors['mass_error'], linewidth=2)
        ax.set_xlabel(r'Time [$\tau_A$]', color=self.colors['text'])
        ax.set_ylabel('Mass Error [%]', color=self.colors['text'])
        ax.set_title('Mass Conservation', color=self.colors['text'])
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.tick_params(colors=self.colors['text'])
        
        # === Panel 4: Divergence Constraint ===
        ax = axes[1, 0]
        ax.set_facecolor(self.ax_facecolor)
        
        div_B = extract(conservation_history, 'max_div_B')
        ax.semilogy(times, div_B + 1e-16, color=self.colors['div_B'], linewidth=2)
        ax.set_xlabel(r'Time [$\tau_A$]', color=self.colors['text'])
        ax.set_ylabel(r'Max $|\nabla \cdot B|$', color=self.colors['text'])
        ax.set_title('Divergence Constraint', color=self.colors['text'])
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.tick_params(colors=self.colors['text'])
        
        # === Panel 5: Mach Number ===
        ax = axes[1, 1]
        ax.set_facecolor(self.ax_facecolor)
        
        mach = extract(stability_history, 'max_sonic_mach')
        ax.plot(times, mach, color=self.colors['mach'], linewidth=2)
        ax.set_xlabel(r'Time [$\tau_A$]', color=self.colors['text'])
        ax.set_ylabel('Max Sonic Mach', color=self.colors['text'])
        ax.set_title('Mach Number Evolution', color=self.colors['text'])
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.tick_params(colors=self.colors['text'])
        
        # === Panel 6: Cross Helicity ===
        ax = axes[1, 2]
        ax.set_facecolor(self.ax_facecolor)
        
        Hc = extract(conservation_history, 'cross_helicity')
        ax.plot(times, Hc, color=self.colors['cross_helicity'], linewidth=2)
        ax.set_xlabel(r'Time [$\tau_A$]', color=self.colors['text'])
        ax.set_ylabel('Cross Helicity', color=self.colors['text'])
        ax.set_title('Cross Helicity Evolution', color=self.colors['text'])
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.tick_params(colors=self.colors['text'])
        
        plt.tight_layout()
        plt.savefig(filename, dpi=self.dpi, facecolor=self.fig_facecolor,
                    bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    
    def create_animation(
        self,
        result: Dict[str, Any],
        filename: str,
        title: str = "MHD Simulation",
        verbose: bool = False
    ) -> None:
        """
        Create GIF animation of density field evolution.
        
        Args:
            result: Simulation result with 'history' containing snapshots
            filename: Output GIF file path
            title: Animation title
            verbose: Print progress
        """
        history = result.get('history', [])
        if not history:
            if verbose:
                print("No history available for animation")
            return
        
        x = result['x']
        y = result['y']
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Get density range across all frames
        all_rho = [snap['U'][0] for snap in history]
        vmin = min(np.min(rho) for rho in all_rho)
        vmax = max(np.max(rho) for rho in all_rho)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 7), facecolor=self.fig_facecolor)
        ax.set_facecolor(self.ax_facecolor)
        
        # Initial plot
        rho = history[0]['U'][0]
        im = ax.pcolormesh(X, Y, rho, cmap='jet', vmin=vmin, vmax=vmax, shading='auto')
        ax.set_xlabel(r'x [$L_0$]', color=self.colors['text'])
        ax.set_ylabel(r'y [$L_0$]', color=self.colors['text'])
        ax.set_aspect('equal')
        ax.tick_params(colors=self.colors['text'])
        
        cbar = fig.colorbar(im, ax=ax, label=r'$\rho$ [$\rho_0$]')
        cbar.ax.yaxis.set_tick_params(color=self.colors['text'])
        plt.setp(cbar.ax.get_yticklabels(), color=self.colors['text'])
        cbar.set_label(r'$\rho$ [$\rho_0$]', color=self.colors['text'])
        
        time_text = ax.set_title(f"{title}\nt = 0.00 τ_A", color=self.colors['text'])
        
        def update(frame):
            rho = history[frame]['U'][0]
            t = history[frame]['t']
            im.set_array(rho.ravel())
            time_text.set_text(f"{title}\nt = {t:.2f} τ_A")
            return [im, time_text]
        
        if verbose:
            print(f"Creating animation with {len(history)} frames...")
        
        anim = animation.FuncAnimation(
            fig, update, frames=len(history),
            interval=1000//self.fps, blit=True
        )
        
        anim.save(filename, writer='pillow', fps=self.fps, 
                  savefig_kwargs={'facecolor': self.fig_facecolor})
        plt.close(fig)
        
        if verbose:
            print(f"Animation saved to {filename}")


# =============================================================================
# Standalone functions for backward compatibility
# =============================================================================

def create_field_plot(
    U: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    gamma: float,
    filename: str,
    title: str = "MHD Fields"
) -> None:
    """
    Standalone function to create field plot.
    
    Args:
        U: Conservative variables [7, nx, ny]
        x, y: Grid coordinates
        t: Current time
        gamma: Adiabatic index
        filename: Output file path
        title: Plot title
    """
    result = {
        'U': U,
        'x': x,
        'y': y,
        't': t,
        'params': {'gamma': gamma}
    }
    animator = Animator()
    animator.create_static_plot(result, filename, title)


def create_diagnostic_plot(
    times: np.ndarray,
    conservation_history: List[Dict],
    stability_history: List[Dict],
    filename: str,
    title: str = "Diagnostics"
) -> None:
    """
    Standalone function to create diagnostic plot.
    
    Args:
        times: Time array
        conservation_history: List of conservation metrics
        stability_history: List of stability metrics
        filename: Output file path
        title: Plot title
    """
    animator = Animator()
    animator.create_metrics_plot(times, conservation_history, stability_history, filename, title)
