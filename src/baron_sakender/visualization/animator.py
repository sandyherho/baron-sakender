"""
Stunning Visualization for 2D Ideal MHD Simulations.

Creates beautiful dark-themed 2D visualizations and animations
with professional aesthetics for MHD turbulence diagnostics.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import warnings
import io

from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm

warnings.filterwarnings('ignore')


class Animator:
    """
    Create stunning visualizations for 2D MHD simulations.
    
    Features professional dark-themed aesthetics with scientific
    colormaps optimized for MHD quantities.
    """
    
    # Dark theme colors
    COLOR_BG = '#0A0E14'
    COLOR_BG_LIGHTER = '#11151C'
    COLOR_BG_PANEL = '#151A23'
    COLOR_ACCENT_CYAN = '#00F5D4'
    COLOR_ACCENT_MAGENTA = '#FF6B9D'
    COLOR_ACCENT_YELLOW = '#FFE66D'
    COLOR_TEXT = '#E6EDF3'
    COLOR_TITLE = '#FFFFFF'
    COLOR_GRID = '#1E2633'
    
    def __init__(self, fps: int = 20, dpi: int = 150):
        """
        Initialize animator.
        
        Args:
            fps: Frames per second for animations
            dpi: Resolution for output images
        """
        self.fps = fps
        self.dpi = dpi
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib dark theme styling."""
        plt.style.use('dark_background')
        plt.rcParams.update({
            'figure.facecolor': self.COLOR_BG,
            'axes.facecolor': self.COLOR_BG_LIGHTER,
            'axes.edgecolor': self.COLOR_GRID,
            'axes.labelcolor': self.COLOR_TEXT,
            'axes.titlecolor': self.COLOR_TITLE,
            'xtick.color': self.COLOR_TEXT,
            'ytick.color': self.COLOR_TEXT,
            'text.color': self.COLOR_TEXT,
            'grid.color': self.COLOR_GRID,
            'grid.alpha': 0.4,
            'legend.facecolor': self.COLOR_BG_PANEL,
            'legend.edgecolor': self.COLOR_GRID,
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'mathtext.fontset': 'cm',
        })
    
    def _plasma_cmap(self):
        """Custom plasma colormap for density/energy."""
        colors = [
            (0.0, 0.0, 0.0), (0.1, 0.0, 0.3), (0.2, 0.0, 0.5), (0.4, 0.0, 0.7),
            (0.0, 0.3, 0.8), (0.0, 0.6, 0.7), (0.0, 0.8, 0.4), (0.2, 0.9, 0.2),
            (0.6, 0.95, 0.1), (0.95, 0.9, 0.1), (1.0, 0.6, 0.0), (1.0, 0.3, 0.0),
            (1.0, 0.0, 0.2), (1.0, 0.4, 0.6), (1.0, 0.8, 0.9), (1.0, 1.0, 1.0),
        ]
        return LinearSegmentedColormap.from_list('plasma_mhd', colors, N=512)
    
    def _magnetic_cmap(self):
        """Custom colormap for magnetic field quantities."""
        colors = [
            (0.0, 0.0, 0.0), (0.05, 0.0, 0.15), (0.15, 0.0, 0.4), (0.3, 0.0, 0.6),
            (0.1, 0.1, 0.8), (0.0, 0.4, 0.9), (0.0, 0.7, 0.9), (0.2, 0.9, 0.7),
            (0.5, 1.0, 0.5), (0.9, 1.0, 0.3), (1.0, 0.8, 0.2), (1.0, 0.5, 0.1),
            (1.0, 0.2, 0.1), (1.0, 0.5, 0.5), (1.0, 1.0, 1.0),
        ]
        return LinearSegmentedColormap.from_list('magnetic', colors, N=512)
    
    def _diverging_cmap(self):
        """Custom diverging colormap for signed quantities (J, ω)."""
        colors = [
            (0.0, 0.3, 1.0), (0.0, 0.5, 0.9), (0.3, 0.7, 1.0), (0.6, 0.85, 1.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.85, 0.6), (1.0, 0.6, 0.3), (1.0, 0.4, 0.1), (1.0, 0.2, 0.0),
        ]
        return LinearSegmentedColormap.from_list('div_mhd', colors, N=512)
    
    def _compute_diagnostics(
        self, 
        U: np.ndarray, 
        dx: float, 
        dy: float, 
        gamma: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute diagnostic fields."""
        rho = U[0]
        vx = U[1] / rho
        vy = U[2] / rho
        Bx = U[3]
        By = U[4]
        
        # Magnetic pressure
        pmag = 0.5 * (Bx**2 + By**2)
        
        # Current density Jz
        dBy_dx = (np.roll(By, -1, axis=0) - np.roll(By, 1, axis=0)) / (2 * dx)
        dBx_dy = (np.roll(Bx, -1, axis=1) - np.roll(Bx, 1, axis=1)) / (2 * dy)
        Jz = dBy_dx - dBx_dy
        
        # Vorticity ωz
        dvy_dx = (np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0)) / (2 * dx)
        dvx_dy = (np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1)) / (2 * dy)
        omega_z = dvy_dx - dvx_dy
        
        return rho, pmag, Jz, omega_z
    
    def _compute_fixed_ranges(
        self,
        snapshots: List[Tuple[float, np.ndarray]],
        dx: float,
        dy: float,
        gamma: float
    ) -> Dict[str, Tuple[float, float]]:
        """Compute fixed colorbar ranges from all snapshots."""
        all_rho, all_pmag, all_jz, all_omega = [], [], [], []
        
        for t, U in snapshots:
            rho, pmag, Jz, omega = self._compute_diagnostics(U, dx, dy, gamma)
            all_rho.extend([rho.min(), rho.max()])
            all_pmag.extend([pmag.min(), pmag.max()])
            all_jz.extend([Jz.min(), Jz.max()])
            all_omega.extend([omega.min(), omega.max()])
        
        # Symmetric ranges for signed quantities
        jz_max = max(abs(min(all_jz)), abs(max(all_jz)))
        omega_max = max(abs(min(all_omega)), abs(max(all_omega)))
        
        return {
            'rho': (min(all_rho), max(all_rho)),
            'pmag': (min(all_pmag), max(all_pmag)),
            'Jz': (-jz_max, jz_max),
            'omega': (-omega_max, omega_max)
        }
    
    def create_static_plot(
        self,
        result: Dict[str, Any],
        filepath: str,
        title: str = "2D MHD Simulation",
        metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Create comprehensive static visualization of final state.
        
        Args:
            result: Simulation result dictionary
            filepath: Output file path
            title: Plot title
            metrics: Optional metrics dictionary
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        snapshots = result['snapshots']
        params = result['params']
        t_final, U_final = snapshots[-1]
        
        dx = params.Lx / params.nx
        dy = params.Ly / params.ny
        
        # Compute diagnostics
        rho, pmag, Jz, omega = self._compute_diagnostics(
            U_final, dx, dy, params.gamma
        )
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor=self.COLOR_BG)
        fig.patch.set_facecolor(self.COLOR_BG)
        
        extent = [0, params.Lx, 0, params.Ly]
        
        # Density
        im1 = axes[0, 0].imshow(
            rho.T, origin='lower', extent=extent, 
            cmap=self._plasma_cmap(), aspect='equal'
        )
        axes[0, 0].set_title(r'Density $\rho$ [$\rho_0$]', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel(r'$x$ [$L_0$]', fontsize=12)
        axes[0, 0].set_ylabel(r'$y$ [$L_0$]', fontsize=12)
        self._add_colorbar(axes[0, 0], im1, r'$\rho$')
        
        # Magnetic pressure
        im2 = axes[0, 1].imshow(
            pmag.T, origin='lower', extent=extent,
            cmap=self._magnetic_cmap(), aspect='equal'
        )
        axes[0, 1].set_title(r'Magnetic Pressure $P_B$ [$P_0$]', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel(r'$x$ [$L_0$]', fontsize=12)
        axes[0, 1].set_ylabel(r'$y$ [$L_0$]', fontsize=12)
        self._add_colorbar(axes[0, 1], im2, r'$B^2/2$')
        
        # Current density
        jz_max = max(abs(Jz.min()), abs(Jz.max()))
        im3 = axes[1, 0].imshow(
            Jz.T, origin='lower', extent=extent,
            cmap=self._diverging_cmap(), vmin=-jz_max, vmax=jz_max, aspect='equal'
        )
        axes[1, 0].set_title(r'Current Density $J_z$ [$J_0$]', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel(r'$x$ [$L_0$]', fontsize=12)
        axes[1, 0].set_ylabel(r'$y$ [$L_0$]', fontsize=12)
        self._add_colorbar(axes[1, 0], im3, r'$J_z$')
        
        # Vorticity
        omega_max = max(abs(omega.min()), abs(omega.max()))
        im4 = axes[1, 1].imshow(
            omega.T, origin='lower', extent=extent,
            cmap=self._diverging_cmap(), vmin=-omega_max, vmax=omega_max, aspect='equal'
        )
        axes[1, 1].set_title(r'Vorticity $\omega_z$ [$\tau_A^{-1}$]', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel(r'$x$ [$L_0$]', fontsize=12)
        axes[1, 1].set_ylabel(r'$y$ [$L_0$]', fontsize=12)
        self._add_colorbar(axes[1, 1], im4, r'$\omega_z$')
        
        # Style axes
        for ax in axes.flat:
            ax.set_facecolor(self.COLOR_BG)
            ax.tick_params(colors=self.COLOR_TEXT, labelsize=10)
            for spine in ax.spines.values():
                spine.set_color(self.COLOR_GRID)
        
        # Title
        fig.suptitle(
            f'{title}\nt = {t_final:.2f} τ_A',
            fontsize=18, fontweight='bold', color=self.COLOR_TITLE, y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        plt.savefig(
            filepath, dpi=self.dpi,
            facecolor=self.COLOR_BG, edgecolor='none',
            bbox_inches='tight'
        )
        plt.close(fig)
    
    def _add_colorbar(self, ax, im, label):
        """Add styled colorbar to axis."""
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(label, fontsize=11, fontweight='bold')
        cbar.ax.tick_params(colors=self.COLOR_TEXT, labelsize=9)
        cbar.outline.set_edgecolor(self.COLOR_GRID)
    
    def create_animation(
        self,
        result: Dict[str, Any],
        filepath: str,
        title: str = "Orszag-Tang Vortex",
        verbose: bool = True
    ):
        """
        Create animated GIF of MHD simulation.
        
        Args:
            result: Simulation result dictionary
            filepath: Output file path
            title: Animation title
            verbose: Print progress
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        snapshots = result['snapshots']
        params = result['params']
        n_frames = len(snapshots)
        
        dx = params.Lx / params.nx
        dy = params.Ly / params.ny
        
        if verbose:
            print(f"      Computing fixed colorbar ranges...")
        
        # Pre-compute ranges
        ranges = self._compute_fixed_ranges(snapshots, dx, dy, params.gamma)
        
        if verbose:
            print(f"      Pre-computing all frame data...")
        
        # Pre-compute all diagnostics
        all_data = []
        for t, U in snapshots:
            rho, pmag, Jz, omega = self._compute_diagnostics(U, dx, dy, params.gamma)
            all_data.append((t, rho.T.copy(), pmag.T.copy(), Jz.T.copy(), omega.T.copy()))
        
        if verbose:
            print(f"      Rendering {n_frames} frames...")
        
        # Setup figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor=self.COLOR_BG)
        fig.patch.set_facecolor(self.COLOR_BG)
        
        extent = [0, params.Lx, 0, params.Ly]
        
        # Initial frame
        t0, rho0, pmag0, jz0, omega0 = all_data[0]
        
        im1 = axes[0, 0].imshow(
            rho0, origin='lower', extent=extent, cmap=self._plasma_cmap(),
            vmin=ranges['rho'][0], vmax=ranges['rho'][1], aspect='equal'
        )
        im2 = axes[0, 1].imshow(
            pmag0, origin='lower', extent=extent, cmap=self._magnetic_cmap(),
            vmin=ranges['pmag'][0], vmax=ranges['pmag'][1], aspect='equal'
        )
        im3 = axes[1, 0].imshow(
            jz0, origin='lower', extent=extent, cmap=self._diverging_cmap(),
            vmin=ranges['Jz'][0], vmax=ranges['Jz'][1], aspect='equal'
        )
        im4 = axes[1, 1].imshow(
            omega0, origin='lower', extent=extent, cmap=self._diverging_cmap(),
            vmin=ranges['omega'][0], vmax=ranges['omega'][1], aspect='equal'
        )
        
        ims = [im1, im2, im3, im4]
        titles = [r'Density $\rho$', r'Magnetic Pressure $P_B$',
                  r'Current Density $J_z$', r'Vorticity $\omega_z$']
        clabels = [r'$\rho$ [$\rho_0$]', r'$B^2/2$ [$P_0$]',
                   r'$J_z$ [$J_0$]', r'$\omega_z$ [$\tau_A^{-1}$]']
        
        for idx, ax in enumerate(axes.flat):
            ax.set_xlabel(r'$x$ [$L_0$]', fontsize=12, fontweight='bold')
            ax.set_ylabel(r'$y$ [$L_0$]', fontsize=12, fontweight='bold')
            ax.set_title(titles[idx], fontsize=14, fontweight='bold', pad=10)
            ax.tick_params(colors=self.COLOR_TEXT, labelsize=10)
            for spine in ax.spines.values():
                spine.set_color(self.COLOR_GRID)
            self._add_colorbar(ax, ims[idx], clabels[idx])
        
        time_text = fig.text(
            0.5, 0.02, '', ha='center', fontsize=16,
            fontweight='bold', color=self.COLOR_ACCENT_CYAN, family='monospace'
        )
        
        fig.suptitle(
            f'{title}: 2D Ideal MHD', fontsize=18,
            fontweight='bold', color=self.COLOR_TITLE, y=0.97
        )
        
        plt.tight_layout(rect=[0, 0.04, 1, 0.94])
        
        def update(frame):
            t, rho, pmag, jz, omega = all_data[frame]
            im1.set_array(rho)
            im2.set_array(pmag)
            im3.set_array(jz)
            im4.set_array(omega)
            time_text.set_text(f't = {t:.2f} τ_A')
            return [im1, im2, im3, im4, time_text]
        
        anim = FuncAnimation(
            fig, update, frames=n_frames, interval=1000 // self.fps, blit=True
        )
        
        if verbose:
            print(f"      Saving {filepath.name}...")
        
        anim.save(
            str(filepath), writer=PillowWriter(fps=self.fps), dpi=100,
            savefig_kwargs={'facecolor': self.COLOR_BG}
        )
        plt.close()
        
        if verbose:
            print(f"      ✓ Animation saved")
    
    def create_metrics_plot(
        self,
        times: np.ndarray,
        conservation_history: List[Dict[str, float]],
        stability_history: List[Dict[str, float]],
        filepath: str,
        title: str = "MHD Diagnostics"
    ):
        """
        Create time series plot of key metrics.
        
        Args:
            times: Time array
            conservation_history: List of conservation metrics
            stability_history: List of stability metrics
            filepath: Output file path
            title: Plot title
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=self.COLOR_BG)
        
        # Extract metrics
        kinetic = np.array([m.get('kinetic_energy', np.nan) for m in conservation_history])
        magnetic = np.array([m.get('magnetic_energy', np.nan) for m in conservation_history])
        total = np.array([m.get('total_energy', np.nan) for m in conservation_history])
        mass = np.array([m.get('total_mass', np.nan) for m in conservation_history])
        div_B = np.array([m.get('max_div_B', np.nan) for m in conservation_history])
        cross_hel = np.array([m.get('cross_helicity', np.nan) for m in conservation_history])
        
        mach = np.array([m.get('max_sonic_mach', np.nan) for m in stability_history])
        beta = np.array([m.get('min_beta', np.nan) for m in stability_history])
        
        # Energy evolution
        axes[0, 0].plot(times, kinetic, '-', color=self.COLOR_ACCENT_CYAN, lw=2, label='Kinetic')
        axes[0, 0].plot(times, magnetic, '-', color=self.COLOR_ACCENT_MAGENTA, lw=2, label='Magnetic')
        axes[0, 0].set_xlabel('Time [τ_A]', fontsize=12)
        axes[0, 0].set_ylabel('Energy [E₀]', fontsize=12)
        axes[0, 0].set_title('Energy Evolution', fontsize=14, fontweight='bold')
        axes[0, 0].legend(loc='best')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Total energy conservation
        E0 = total[0]
        energy_error = np.abs(total - E0) / E0 * 100
        axes[0, 1].semilogy(times, energy_error + 1e-15, '-', color=self.COLOR_ACCENT_YELLOW, lw=2)
        axes[0, 1].set_xlabel('Time [τ_A]', fontsize=12)
        axes[0, 1].set_ylabel('Energy Error [%]', fontsize=12)
        axes[0, 1].set_title('Energy Conservation', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Mass conservation
        M0 = mass[0]
        mass_error = np.abs(mass - M0) / M0 * 100
        axes[0, 2].semilogy(times, mass_error + 1e-15, '-', color=self.COLOR_ACCENT_CYAN, lw=2)
        axes[0, 2].set_xlabel('Time [τ_A]', fontsize=12)
        axes[0, 2].set_ylabel('Mass Error [%]', fontsize=12)
        axes[0, 2].set_title('Mass Conservation', fontsize=14, fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Divergence of B
        axes[1, 0].semilogy(times, div_B + 1e-15, '-', color=self.COLOR_ACCENT_MAGENTA, lw=2)
        axes[1, 0].set_xlabel('Time [τ_A]', fontsize=12)
        axes[1, 0].set_ylabel('Max |∇·B|', fontsize=12)
        axes[1, 0].set_title('Divergence Constraint', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Mach number
        axes[1, 1].plot(times, mach, '-', color=self.COLOR_ACCENT_YELLOW, lw=2)
        axes[1, 1].set_xlabel('Time [τ_A]', fontsize=12)
        axes[1, 1].set_ylabel('Max Sonic Mach', fontsize=12)
        axes[1, 1].set_title('Mach Number Evolution', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Cross helicity
        axes[1, 2].plot(times, cross_hel, '-', color=self.COLOR_ACCENT_CYAN, lw=2)
        axes[1, 2].set_xlabel('Time [τ_A]', fontsize=12)
        axes[1, 2].set_ylabel('Cross Helicity', fontsize=12)
        axes[1, 2].set_title('Cross Helicity Evolution', fontsize=14, fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Style
        for ax in axes.flat:
            ax.set_facecolor(self.COLOR_BG_LIGHTER)
            ax.tick_params(colors=self.COLOR_TEXT)
            for spine in ax.spines.values():
                spine.set_color(self.COLOR_GRID)
        
        fig.suptitle(title, fontsize=18, fontweight='bold', color=self.COLOR_TITLE, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        plt.savefig(
            filepath, dpi=self.dpi,
            facecolor=self.COLOR_BG, edgecolor='none',
            bbox_inches='tight'
        )
        plt.close(fig)
