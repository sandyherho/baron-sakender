#!/usr/bin/env python3
"""
Command-line interface for Baron-Sakender MHD Simulator.

Usage:
    baron-sakender case1              # Run Orszag-Tang vortex
    baron-sakender case2              # Run strong magnetic field
    baron-sakender case3              # Run current sheet
    baron-sakender case4              # Run Alfven wave
    baron-sakender --help             # Show help
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np

# Try to import package modules
try:
    from baron_sakender.core.solver import MHDSolver
    from baron_sakender.core.initial_conditions import get_initial_condition
    from baron_sakender.core.metrics import (
        compute_conservation_metrics,
        compute_stability_metrics,
        compute_turbulence_metrics,
        compute_information_metrics,
        compute_composite_metrics,
        compute_all_metrics
    )
    from baron_sakender.visualization.animator import Animator
except ImportError:
    # Fallback for development
    from core.solver import MHDSolver
    from core.initial_conditions import get_initial_condition
    from core.metrics import (
        compute_conservation_metrics,
        compute_stability_metrics,
        compute_turbulence_metrics,
        compute_information_metrics,
        compute_composite_metrics,
        compute_all_metrics
    )
    from visualization.animator import Animator


# =============================================================================
# Scenario Configurations
# =============================================================================

SCENARIOS = {
    'case1': {
        'name': 'Case 1 - Orszag-Tang Vortex',
        'init_type': 'orszag_tang',
        'nx': 256,
        'ny': 256,
        'Lx': 2 * np.pi,
        'Ly': 2 * np.pi,
        't_end': 3.0,
        'save_dt': 0.05,
        'cfl': 0.4,
        'gamma': 5/3,
    },
    'case2': {
        'name': 'Case 2 - Strong Magnetic Field',
        'init_type': 'strong_magnetic',
        'nx': 256,
        'ny': 256,
        'Lx': 2 * np.pi,
        'Ly': 2 * np.pi,
        't_end': 2.0,
        'save_dt': 0.04,
        'cfl': 0.35,
        'gamma': 5/3,
    },
    'case3': {
        'name': 'Case 3 - Current Sheet',
        'init_type': 'current_sheet',
        'nx': 256,
        'ny': 256,
        'Lx': 2 * np.pi,
        'Ly': 2 * np.pi,
        't_end': 2.5,
        'save_dt': 0.05,
        'cfl': 0.4,
        'gamma': 5/3,
    },
    'case4': {
        'name': 'Case 4 - Alfven Wave',
        'init_type': 'alfven_wave',
        'nx': 128,
        'ny': 128,
        'Lx': 2 * np.pi,
        'Ly': 2 * np.pi,
        't_end': 4.0,
        'save_dt': 0.1,
        'cfl': 0.4,
        'gamma': 5/3,
    },
}

# Physical normalization constants
RHO_0 = 1e-12  # kg/m^3 (typical coronal density)
B_0 = 1e-9     # T (typical coronal magnetic field)
L_0 = 1e6      # m (typical coronal length scale)


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_file: str, verbose: bool = False) -> logging.Logger:
    """Setup logging to file and console."""
    logger = logging.getLogger('baron_sakender')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    return logger


def log_metrics(logger: logging.Logger, metrics: Dict[str, float], category: str) -> None:
    """
    Log metrics in a clean format.
    
    Only logs reliable metrics (excludes spectral indices and cascade efficiency).
    """
    logger.info(f"\n{category.upper()}:")
    
    # Define which metrics to show per category
    show_metrics = {
        'CONSERVATION': [
            'total_mass', 'total_momentum', 'total_momentum_x', 'total_momentum_y',
            'total_energy', 'kinetic_energy', 'magnetic_energy', 'thermal_energy',
            'cross_helicity', 'max_div_B', 'mean_div_B'
        ],
        'STABILITY': [
            'max_sonic_mach', 'mean_sonic_mach', 'max_alfven_mach', 'mean_alfven_mach',
            'max_fast_mach', 'max_velocity', 'max_sound_speed', 'max_alfven_speed',
            'max_fast_speed', 'min_beta', 'max_beta', 'mean_beta',
            'cfl_x', 'cfl_y', 'cfl_effective', 'target_cfl', 'dt',
            'min_density', 'min_pressure', 'is_stable'
        ],
        'TURBULENCE': [
            'kinetic_energy', 'magnetic_energy', 'total_energy', 'energy_ratio',
            'enstrophy', 'mean_vorticity', 'max_vorticity', 'rms_vorticity',
            'current_squared', 'mean_current', 'max_current', 'rms_current',
            'cross_helicity', 'normalized_cross_helicity',
            'residual_energy', 'normalized_residual_energy',
            'elsasser_plus_energy', 'elsasser_minus_energy', 'energy_imbalance',
            'vorticity_kurtosis', 'current_kurtosis', 'taylor_microscale'
        ],
        'INFORMATION METRICS': [
            'entropy_density', 'entropy_velocity', 'entropy_magnetic',
            'entropy_vorticity', 'entropy_current'
        ],
        'COMPOSITE METRICS': [
            'dynamo_efficiency', 'alignment_index', 'coherent_structure_index',
            'intermittency_index', 'conservation_quality',
            'mass_conservation_error', 'energy_conservation_error'
        ],
    }
    
    keys_to_show = show_metrics.get(category.upper(), list(metrics.keys()))
    
    for key in keys_to_show:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                if abs(value) < 1e-3 or abs(value) > 1e4:
                    logger.info(f"  {key}: {value:.6e}")
                else:
                    logger.info(f"  {key}: {value:.6f}")
            else:
                logger.info(f"  {key}: {value}")


# =============================================================================
# Main Simulation Runner
# =============================================================================

def run_simulation(
    scenario_key: str,
    output_dir: str = 'outputs',
    verbose: bool = False,
    use_gpu: bool = False
) -> Dict[str, Any]:
    """
    Run a complete MHD simulation for a given scenario.
    
    Args:
        scenario_key: Key from SCENARIOS dict (e.g., 'case1')
        output_dir: Directory for output files
        verbose: Enable verbose output
        use_gpu: Use GPU acceleration (JAX)
    
    Returns:
        Dictionary with simulation results and timing information
    """
    # Get scenario configuration
    if scenario_key not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_key}. "
                        f"Available: {list(SCENARIOS.keys())}")
    
    config = SCENARIOS[scenario_key]
    scenario_name = f"{scenario_key}_{config['init_type']}"
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Setup logging
    log_file = f"logs/{scenario_name}.log"
    logger = setup_logging(log_file, verbose)
    
    # Timing dictionary
    timing = {}
    
    # Log header
    logger.info("=" * 70)
    logger.info("2D IDEAL MHD SIMULATION - BARON-SAKENDER")
    logger.info(f"Scenario: {config['name']}")
    logger.info("=" * 70)
    logger.info("")
    
    # Log parameters
    logger.info("GRID PARAMETERS:")
    logger.info(f"  nx = {config['nx']}")
    logger.info(f"  ny = {config['ny']}")
    logger.info(f"  Lx = {config['Lx']:.4f}")
    logger.info(f"  Ly = {config['Ly']:.4f}")
    logger.info("")
    
    logger.info("PHYSICAL PARAMETERS:")
    logger.info(f"  gamma = {config['gamma']:.4f}")
    logger.info(f"  rho_0 = {RHO_0:.2e} kg/m^3")
    logger.info(f"  B_0 = {B_0:.2e} T")
    logger.info(f"  L_0 = {L_0:.2e} m")
    logger.info("")
    
    logger.info("SIMULATION PARAMETERS:")
    logger.info(f"  t_end = {config['t_end']}")
    logger.info(f"  save_dt = {config['save_dt']}")
    logger.info(f"  CFL = {config['cfl']}")
    logger.info(f"  use_gpu = {use_gpu}")
    logger.info(f"  init_type = {config['init_type']}")
    logger.info("=" * 70)
    logger.info("")
    
    # Initialize system
    t_start = time.perf_counter()
    
    # Grid setup
    nx, ny = config['nx'], config['ny']
    Lx, Ly = config['Lx'], config['Ly']
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    dx, dy = x[1] - x[0], y[1] - y[0]
    
    # Get initial conditions
    U0 = get_initial_condition(
        config['init_type'], 
        x, y, 
        gamma=config['gamma']
    )
    
    timing['system_init'] = time.perf_counter() - t_start
    
    # Initialize solver
    t_start = time.perf_counter()
    solver = MHDSolver(
        nx=nx, ny=ny,
        dx=dx, dy=dy,
        gamma=config['gamma'],
        cfl=config['cfl'],
        use_gpu=use_gpu
    )
    timing['solver_init'] = time.perf_counter() - t_start
    
    # Run simulation
    t_start = time.perf_counter()
    
    # Storage for history
    history = []
    times = []
    conservation_history = []
    stability_history = []
    
    U = U0.copy()
    t = 0.0
    t_end = config['t_end']
    save_dt = config['save_dt']
    next_save = 0.0
    
    # Compute initial metrics
    initial_conservation = compute_conservation_metrics(U, dx, dy, config['gamma'])
    
    while t < t_end:
        # Get adaptive dt
        dt = solver.compute_dt(U)
        
        # Don't overshoot end time or save time
        dt = min(dt, t_end - t, next_save - t + 1e-10)
        
        # Advance solution
        U = solver.step(U, dt)
        t += dt
        
        # Save snapshot
        if t >= next_save - 1e-10:
            # Compute metrics
            cons = compute_conservation_metrics(U, dx, dy, config['gamma'])
            stab = compute_stability_metrics(U, dx, dy, config['gamma'], dt, config['cfl'])
            
            history.append({'U': U.copy(), 't': t})
            times.append(t)
            conservation_history.append(cons)
            stability_history.append(stab)
            
            next_save += save_dt
            
            if verbose:
                print(f"  t = {t:.3f} / {t_end:.3f}, E_k = {cons['kinetic_energy']:.4f}, "
                      f"E_m = {cons['magnetic_energy']:.4f}")
    
    timing['simulation'] = time.perf_counter() - t_start
    
    # Compute final metrics
    t_start = time.perf_counter()
    
    final_metrics = compute_all_metrics(
        U, dx, dy, config['gamma'], 
        dt=dt, cfl=config['cfl'],
        conservation_initial=initial_conservation
    )
    
    timing['metrics'] = time.perf_counter() - t_start
    
    # Log final metrics
    logger.info("=" * 70)
    logger.info("FINAL METRICS:")
    logger.info("=" * 70)
    
    # Conservation metrics
    cons_metrics = compute_conservation_metrics(U, dx, dy, config['gamma'])
    log_metrics(logger, cons_metrics, 'CONSERVATION')
    
    # Stability metrics
    stab_metrics = compute_stability_metrics(U, dx, dy, config['gamma'], dt, config['cfl'])
    log_metrics(logger, stab_metrics, 'STABILITY')
    
    # Turbulence metrics
    turb_metrics = compute_turbulence_metrics(U, dx, dy, config['gamma'])
    log_metrics(logger, turb_metrics, 'TURBULENCE')
    
    # Information metrics
    info_metrics = compute_information_metrics(U, dx, dy, config['gamma'])
    log_metrics(logger, info_metrics, 'INFORMATION METRICS')
    
    # Composite metrics
    comp_metrics = compute_composite_metrics(U, dx, dy, config['gamma'], initial_conservation)
    log_metrics(logger, comp_metrics, 'COMPOSITE METRICS')
    
    logger.info("=" * 70)
    
    # Create result dictionary
    result = {
        'U': U,
        'x': x,
        'y': y,
        't': t,
        'params': config,
        'history': history,
        'times': times,
        'conservation_history': conservation_history,
        'stability_history': stability_history,
        'final_metrics': final_metrics,
    }
    
    # Generate visualizations
    t_start = time.perf_counter()
    
    animator = Animator(fps=10, dpi=150)
    
    # Field plot
    png_file = f"{output_dir}/{scenario_name}_fields.png"
    animator.create_static_plot(result, png_file, config['name'], final_metrics)
    
    # Metrics plot
    metrics_png = f"{output_dir}/{scenario_name}_metrics.png"
    animator.create_metrics_plot(
        np.array(times), 
        conservation_history, 
        stability_history,
        metrics_png,
        f"{config['name']} - Diagnostics"
    )
    
    timing['png_save'] = time.perf_counter() - t_start
    timing['visualization'] = timing['png_save']
    
    # Create GIF animation
    t_start = time.perf_counter()
    gif_file = f"{output_dir}/{scenario_name}.gif"
    animator.create_animation(result, gif_file, config['name'], verbose=verbose)
    timing['gif_save'] = time.perf_counter() - t_start
    
    # Save data
    t_start = time.perf_counter()
    
    # Save to NetCDF
    try:
        import netCDF4 as nc
        nc_file = f"{output_dir}/{scenario_name}.nc"
        with nc.Dataset(nc_file, 'w', format='NETCDF4') as ds:
            # Dimensions
            ds.createDimension('x', nx)
            ds.createDimension('y', ny)
            ds.createDimension('time', len(history))
            ds.createDimension('var', 7)
            
            # Coordinates
            ds.createVariable('x', 'f8', ('x',))[:] = x
            ds.createVariable('y', 'f8', ('y',))[:] = y
            ds.createVariable('time', 'f8', ('time',))[:] = times
            
            # Data
            U_all = np.array([h['U'] for h in history])
            ds.createVariable('U', 'f8', ('time', 'var', 'x', 'y'))[:] = U_all
            
            # Attributes
            ds.scenario = config['name']
            ds.gamma = config['gamma']
            ds.cfl = config['cfl']
        timing['netcdf_save'] = time.perf_counter() - t_start
    except ImportError:
        timing['netcdf_save'] = 0.0
    
    # Save metrics to CSV
    t_start = time.perf_counter()
    try:
        import pandas as pd
        
        # Combine all metrics into dataframe
        all_data = []
        for i, t_val in enumerate(times):
            row = {'time': t_val}
            row.update(conservation_history[i])
            row.update(stability_history[i])
            all_data.append(row)
        
        df = pd.DataFrame(all_data)
        csv_file = f"{output_dir}/{scenario_name}_metrics.csv"
        df.to_csv(csv_file, index=False)
        timing['csv_save'] = time.perf_counter() - t_start
    except ImportError:
        timing['csv_save'] = 0.0
    
    # Log timing
    logger.info("=" * 70)
    logger.info("TIMING BREAKDOWN:")
    logger.info("=" * 70)
    total_time = 0
    for key, val in timing.items():
        logger.info(f"  {key}: {val:.3f} s")
        total_time += val
    logger.info("  " + "-" * 40)
    logger.info(f"  TOTAL: {total_time:.3f} s")
    logger.info("=" * 70)
    
    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SIMULATION SUMMARY:")
    logger.info("=" * 70)
    logger.info("")
    logger.info("ERRORS: None")
    logger.info("")
    logger.info("WARNINGS: None")
    logger.info("")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 70)
    logger.info(f"Simulation completed: {scenario_name}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 70)
    
    return result


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Baron-Sakender: 2D Ideal MHD Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  baron-sakender case1          Run Orszag-Tang vortex benchmark
  baron-sakender case2          Run strong magnetic field case
  baron-sakender case3          Run current sheet case
  baron-sakender case4          Run Alfven wave propagation
  baron-sakender case1 -v       Run with verbose output
  baron-sakender case1 --gpu    Run with GPU acceleration

Available scenarios:
  case1  Orszag-Tang Vortex (standard MHD benchmark)
  case2  Strong Magnetic Field (low-beta regime)
  case3  Current Sheet (reconnection setup)
  case4  Alfven Wave (wave propagation test)
        """
    )
    
    parser.add_argument(
        'scenario',
        choices=list(SCENARIOS.keys()),
        help='Scenario to run'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='outputs',
        help='Output directory (default: outputs)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU acceleration (requires JAX with CUDA)'
    )
    
    args = parser.parse_args()
    
    try:
        result = run_simulation(
            args.scenario,
            output_dir=args.output,
            verbose=args.verbose,
            use_gpu=args.gpu
        )
        print(f"\nSimulation completed successfully!")
        print(f"Outputs saved to: {args.output}/")
        return 0
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
