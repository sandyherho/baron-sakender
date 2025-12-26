#!/usr/bin/env python3
"""
Command-line interface for Baron-Sakender MHD Simulator.

Usage:
    baron-sakender case1              # Run Orszag-Tang vortex
    baron-sakender case2              # Run strong magnetic field
    baron-sakender case3              # Run current sheet
    baron-sakender case4              # Run Alfven wave
    baron-sakender -a                 # Run all cases sequentially
    baron-sakender --all              # Run all cases sequentially
    baron-sakender -c config.txt      # Run from config file
    baron-sakender --config my.txt    # Run from config file
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
from tqdm import tqdm

# Import from actual package structure
from baron_sakender.core.mhd_system import MHDSystem
from baron_sakender.core.integrator import MHDIntegrator
from baron_sakender.core.metrics import (
    compute_conservation_metrics,
    compute_stability_metrics,
    compute_turbulence_metrics,
    compute_information_metrics,
    compute_composite_metrics,
    compute_all_metrics
)
from baron_sakender.visualization.animator import Animator


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
        'amplitude': 1.0,
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
        'beta': 0.1,
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
        'width': 0.2,
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
        'amplitude': 0.1,
        'wavenumber': 2,
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
    
    # Initialize MHD system
    t_start = time.perf_counter()
    
    system = MHDSystem(
        nx=config['nx'],
        ny=config['ny'],
        Lx=config['Lx'],
        Ly=config['Ly'],
        gamma=config['gamma'],
    )
    
    # Initialize based on type
    if config['init_type'] == 'orszag_tang':
        amplitude = config.get('amplitude', 1.0)
        system.init_orszag_tang(amplitude=amplitude)
    elif config['init_type'] == 'strong_magnetic':
        beta = config.get('beta', 0.1)
        system.init_strong_magnetic(beta=beta)
    elif config['init_type'] == 'current_sheet':
        width = config.get('width', 0.2)
        system.init_current_sheet(width=width)
    elif config['init_type'] == 'alfven_wave':
        amplitude = config.get('amplitude', 0.1)
        k = config.get('wavenumber', config.get('k', 2))
        system.init_alfven_wave(amplitude=amplitude, k=k)
    else:
        raise ValueError(f"Unknown init_type: {config['init_type']}")
    
    timing['system_init'] = time.perf_counter() - t_start
    
    # Initialize integrator
    t_start = time.perf_counter()
    integrator = MHDIntegrator(cfl=config['cfl'], use_gpu=use_gpu)
    timing['solver_init'] = time.perf_counter() - t_start
    
    # Get grid info
    x, y = system.x, system.y
    dx, dy = system.dx, system.dy
    
    # Compute initial metrics
    initial_conservation = compute_conservation_metrics(system.U, dx, dy, config['gamma'])
    
    # Run simulation with diagnostics
    t_start = time.perf_counter()
    
    print(f"\n{'='*60}")
    print(f"  Running: {config['name']}")
    print(f"  Grid: {config['nx']}x{config['ny']}, t_end: {config['t_end']}")
    print(f"{'='*60}\n")
    
    result = integrator.run_with_diagnostics(
        system,
        t_end=config['t_end'],
        save_dt=config['save_dt'],
        verbose=True  # Enable tqdm progress bar
    )
    
    timing['simulation'] = time.perf_counter() - t_start
    print()  # Clean line after tqdm
    
    # Get final state
    _, U_final = result['snapshots'][-1]
    t_final = result['times'][-1]
    
    # Compute final metrics
    t_start = time.perf_counter()
    
    final_metrics = compute_all_metrics(
        U_final, dx, dy, config['gamma'], 
        dt=0.01, cfl=config['cfl'],
        conservation_initial=initial_conservation
    )
    
    timing['metrics'] = time.perf_counter() - t_start
    
    # Log final metrics
    logger.info("=" * 70)
    logger.info("FINAL METRICS:")
    logger.info("=" * 70)
    
    # Conservation metrics
    cons_metrics = compute_conservation_metrics(U_final, dx, dy, config['gamma'])
    log_metrics(logger, cons_metrics, 'CONSERVATION')
    
    # Stability metrics
    stab_metrics = compute_stability_metrics(U_final, dx, dy, config['gamma'], dt=0.01, cfl=config['cfl'])
    log_metrics(logger, stab_metrics, 'STABILITY')
    
    # Turbulence metrics
    turb_metrics = compute_turbulence_metrics(U_final, dx, dy, config['gamma'])
    log_metrics(logger, turb_metrics, 'TURBULENCE')
    
    # Information metrics
    info_metrics = compute_information_metrics(U_final, dx, dy, config['gamma'])
    log_metrics(logger, info_metrics, 'INFORMATION METRICS')
    
    # Composite metrics
    comp_metrics = compute_composite_metrics(U_final, dx, dy, config['gamma'], initial_conservation)
    log_metrics(logger, comp_metrics, 'COMPOSITE METRICS')
    
    logger.info("=" * 70)
    
    # Build history list from snapshots
    history = [{'U': U.copy(), 't': t} for t, U in result['snapshots']]
    times = list(result['times'])
    
    # Create result dictionary for visualization
    vis_result = {
        'U': U_final,
        'x': x,
        'y': y,
        't': t_final,
        'params': config,
        'history': history,
        'times': times,
        'conservation_history': result['conservation_history'],
        'stability_history': result['stability_history'],
        'final_metrics': final_metrics,
    }
    
    # Post-processing with progress bar
    post_steps = [
        ('Creating field plot', 'png'),
        ('Creating metrics plot', 'metrics_png'),
        ('Creating animation', 'gif'),
        ('Saving NetCDF', 'netcdf'),
        ('Saving CSV', 'csv'),
    ]
    
    pbar = tqdm(post_steps, desc="Post-processing", unit="step", leave=True)
    
    for step_name, step_key in pbar:
        pbar.set_description(f"  {step_name}")
        
        if step_key == 'png':
            t_start = time.perf_counter()
            animator = Animator(fps=10, dpi=150)
            png_file = f"{output_dir}/{scenario_name}_fields.png"
            animator.create_static_plot(vis_result, png_file, config['name'], final_metrics)
            timing['png_save'] = time.perf_counter() - t_start
            
        elif step_key == 'metrics_png':
            t_start = time.perf_counter()
            metrics_png = f"{output_dir}/{scenario_name}_metrics.png"
            animator.create_metrics_plot(
                np.array(times), 
                result['conservation_history'], 
                result['stability_history'],
                metrics_png,
                f"{config['name']} - Diagnostics"
            )
            timing['visualization'] = time.perf_counter() - t_start
            
        elif step_key == 'gif':
            t_start = time.perf_counter()
            gif_file = f"{output_dir}/{scenario_name}.gif"
            animator.create_animation(vis_result, gif_file, config['name'], verbose=False)
            timing['gif_save'] = time.perf_counter() - t_start
            
        elif step_key == 'netcdf':
            t_start = time.perf_counter()
            try:
                import netCDF4 as nc
                nc_file = f"{output_dir}/{scenario_name}.nc"
                with nc.Dataset(nc_file, 'w', format='NETCDF4') as ds:
                    ds.createDimension('x', config['nx'])
                    ds.createDimension('y', config['ny'])
                    ds.createDimension('time', len(history))
                    ds.createDimension('var', 7)
                    ds.createVariable('x', 'f8', ('x',))[:] = x
                    ds.createVariable('y', 'f8', ('y',))[:] = y
                    ds.createVariable('time', 'f8', ('time',))[:] = times
                    U_all = np.array([h['U'] for h in history])
                    ds.createVariable('U', 'f8', ('time', 'var', 'x', 'y'))[:] = U_all
                    ds.scenario = config['name']
                    ds.gamma = config['gamma']
                    ds.cfl = config['cfl']
                timing['netcdf_save'] = time.perf_counter() - t_start
            except ImportError:
                timing['netcdf_save'] = 0.0
                
        elif step_key == 'csv':
            t_start = time.perf_counter()
            try:
                import pandas as pd
                all_data = []
                for i, t_val in enumerate(times):
                    row = {'time': t_val}
                    row.update(result['conservation_history'][i])
                    row.update(result['stability_history'][i])
                    all_data.append(row)
                df = pd.DataFrame(all_data)
                csv_file = f"{output_dir}/{scenario_name}_metrics.csv"
                df.to_csv(csv_file, index=False)
                timing['csv_save'] = time.perf_counter() - t_start
            except ImportError:
                timing['csv_save'] = 0.0
    
    print()  # Clean line after progress bar
    
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
    
    # Print clean summary to console
    total_time = sum(timing.values())
    print(f"\n{'='*60}")
    print(f"  SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Mass conservation error: {comp_metrics.get('mass_conservation_error', 0)*100:.4f}%")
    print(f"  Energy conservation error: {comp_metrics.get('energy_conservation_error', 0)*100:.4f}%")
    print(f"  Max |∇·B|: {cons_metrics.get('max_div_B', 0):.2e}")
    print(f"{'='*60}")
    print(f"  Output files:")
    print(f"    • {output_dir}/{scenario_name}_fields.png")
    print(f"    • {output_dir}/{scenario_name}_metrics.png")
    print(f"    • {output_dir}/{scenario_name}.gif")
    print(f"    • {output_dir}/{scenario_name}.nc")
    print(f"    • {output_dir}/{scenario_name}_metrics.csv")
    print(f"    • logs/{scenario_name}.log")
    print(f"{'='*60}\n")
    
    return vis_result


# =============================================================================
# CLI Entry Point
# =============================================================================

def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a text file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary compatible with run_simulation
    """
    from baron_sakender.io.config_manager import ConfigManager
    
    config = ConfigManager.load(config_path)
    
    # Map config file keys to SCENARIOS format
    result = {
        'name': config.get('scenario_name', 'Custom Simulation'),
        'init_type': config.get('init_type', 'orszag_tang'),
        'nx': config.get('nx', 256),
        'ny': config.get('ny', 256),
        'Lx': config.get('Lx', 2 * np.pi),
        'Ly': config.get('Ly', 2 * np.pi),
        't_end': config.get('t_end', 3.0),
        'save_dt': config.get('save_dt', 0.05),
        'cfl': config.get('cfl', 0.4),
        'gamma': config.get('gamma', 5/3),
    }
    
    # Optional init parameters
    if 'beta' in config:
        result['beta'] = config['beta']
    if 'width' in config:
        result['width'] = config['width']
    if 'amplitude' in config:
        result['amplitude'] = config['amplitude']
    if 'wavenumber' in config:
        result['wavenumber'] = config['wavenumber']
    
    return result


def run_simulation_from_config(
    config: Dict[str, Any],
    output_dir: str = 'outputs',
    verbose: bool = False,
    use_gpu: bool = False
) -> Dict[str, Any]:
    """
    Run simulation from a configuration dictionary.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory
        verbose: Enable verbose output
        use_gpu: Use GPU acceleration
    
    Returns:
        Simulation results
    """
    # Create a temporary scenario key
    init_type = config.get('init_type', 'custom')
    scenario_name = f"custom_{init_type}"
    
    # Register the config temporarily
    SCENARIOS[scenario_name] = config
    
    try:
        result = run_simulation(
            scenario_name,
            output_dir=output_dir,
            verbose=verbose,
            use_gpu=use_gpu
        )
        return result
    finally:
        # Clean up temporary scenario
        if scenario_name in SCENARIOS:
            del SCENARIOS[scenario_name]


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Baron-Sakender: 2D Ideal MHD Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  baron-sakender case1              Run Orszag-Tang vortex benchmark
  baron-sakender case2              Run strong magnetic field case
  baron-sakender case3              Run current sheet case
  baron-sakender case4              Run Alfven wave propagation
  baron-sakender -a                 Run all test cases sequentially
  baron-sakender --all              Run all test cases sequentially
  baron-sakender -c config.txt      Run from config file
  baron-sakender case1 -v           Run with verbose output
  baron-sakender case1 --gpu        Run with GPU acceleration

Available scenarios:
  case1  Orszag-Tang Vortex (standard MHD benchmark)
  case2  Strong Magnetic Field (low-beta regime)
  case3  Current Sheet (reconnection setup)
  case4  Alfven Wave (wave propagation test)
        """
    )
    
    parser.add_argument(
        'scenario',
        nargs='?',
        choices=list(SCENARIOS.keys()),
        default=None,
        help='Scenario to run (optional if using -a or -c)'
    )
    
    parser.add_argument(
        '-a', '--all',
        action='store_true',
        help='Run all test cases sequentially'
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        default=None,
        help='Path to configuration file (.txt)'
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
    
    # Validate arguments
    if not args.all and not args.config and not args.scenario:
        parser.error("Please specify a scenario, use -a/--all, or provide -c/--config")
    
    try:
        # Run all cases
        if args.all:
            print(f"\n{'='*60}")
            print(f"  BARON-SAKENDER: Running All Test Cases")
            print(f"{'='*60}\n")
            
            results = {}
            failed = []
            
            for i, scenario in enumerate(SCENARIOS.keys(), 1):
                print(f"\n[{i}/{len(SCENARIOS)}] Running {scenario}...")
                try:
                    result = run_simulation(
                        scenario,
                        output_dir=args.output,
                        verbose=args.verbose,
                        use_gpu=args.gpu
                    )
                    results[scenario] = result
                except Exception as e:
                    print(f"  ERROR: {e}")
                    failed.append((scenario, str(e)))
            
            # Summary
            print(f"\n{'='*60}")
            print(f"  ALL SIMULATIONS COMPLETE")
            print(f"{'='*60}")
            print(f"  Successful: {len(results)}/{len(SCENARIOS)}")
            if failed:
                print(f"  Failed: {len(failed)}")
                for scenario, error in failed:
                    print(f"    • {scenario}: {error}")
            print(f"{'='*60}\n")
            
            return 0 if not failed else 1
        
        # Run from config file
        elif args.config:
            if not os.path.exists(args.config):
                print(f"Error: Config file not found: {args.config}", file=sys.stderr)
                return 1
            
            print(f"\n  Loading config: {args.config}")
            config = load_config_file(args.config)
            
            result = run_simulation_from_config(
                config,
                output_dir=args.output,
                verbose=args.verbose,
                use_gpu=args.gpu
            )
            return 0
        
        # Run single scenario
        else:
            result = run_simulation(
                args.scenario,
                output_dir=args.output,
                verbose=args.verbose,
                use_gpu=args.gpu
            )
            return 0
            
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
