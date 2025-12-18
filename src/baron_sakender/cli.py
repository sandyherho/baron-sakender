#!/usr/bin/env python
"""
Command Line Interface for baron-sakender 2D Ideal MHD Solver.

Usage:
    baron-sakender case1              # Standard Orszag-Tang
    baron-sakender case2              # Strong magnetic field
    baron-sakender case3              # Current sheet
    baron-sakender case4              # Alfvén wave
    baron-sakender --all              # Run all cases
    baron-sakender case1 --gpu        # Use GPU acceleration
    baron-sakender --config path.txt  # Custom config
"""

import argparse
import sys
from pathlib import Path
import numpy as np

from .core.mhd_system import MHDSystem
from .core.integrator import MHDSolver
from .core.metrics import (
    compute_conservation_metrics,
    compute_stability_metrics,
    compute_turbulence_metrics,
    compute_information_metrics,
    compute_composite_metrics,
    compute_all_metrics,
)
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler
from .visualization.animator import Animator
from .utils.logger import SimulationLogger
from .utils.timer import Timer


def print_header():
    """Print ASCII art header."""
    print("\n" + "=" * 70)
    print(" " * 12 + "Baron-Sakender: 2D Ideal MHD Solver")
    print(" " * 25 + "Version 0.0.1")
    print("=" * 70)
    print("\n  JAX-Accelerated Magnetohydrodynamics")
    print("  Conservation | Stability | Turbulence | Information Metrics")
    print("  License: MIT")
    print("=" * 70 + "\n")


def normalize_scenario_name(scenario_name: str) -> str:
    """Convert scenario name to clean filename format."""
    clean = scenario_name.lower()
    clean = clean.replace(' - ', '_')
    clean = clean.replace('-', '_')
    clean = clean.replace(' ', '_')
    
    while '__' in clean:
        clean = clean.replace('__', '_')
    
    clean = clean.rstrip('_')
    return clean


def run_scenario(
    config: dict,
    output_dir: str = "outputs",
    verbose: bool = True,
    use_gpu: bool = False
):
    """Run a complete MHD simulation scenario."""
    
    scenario_name = config.get('scenario_name', 'simulation')
    clean_name = normalize_scenario_name(scenario_name)
    
    # Override GPU setting from command line
    if use_gpu:
        config['use_gpu'] = True
    
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'=' * 70}")
    
    logger = SimulationLogger(clean_name, "logs", verbose)
    timer = Timer()
    timer.start("total")
    
    try:
        logger.log_parameters(config)
        
        # [1/9] Initialize system
        with timer.time_section("system_init"):
            if verbose:
                print("\n[1/9] Initializing MHD system...")
            
            system = MHDSystem(
                nx=config.get('nx', 256),
                ny=config.get('ny', 256),
                gamma=config.get('gamma', 5.0 / 3.0),
                Lx=config.get('Lx', 2 * np.pi),
                Ly=config.get('Ly', 2 * np.pi),
                rho_0=config.get('rho_0', 1.0e-12),
                B_0=config.get('B_0', 1.0e-9),
                L_0=config.get('L_0', 1.0e6),
            )
            
            # Initialize based on type
            init_type = config.get('init_type', 'orszag_tang')
            if init_type == 'orszag_tang':
                system.init_orszag_tang(amplitude=config.get('amplitude', 1.0))
            elif init_type == 'strong_magnetic':
                system.init_strong_magnetic(beta=config.get('beta', 0.1))
            elif init_type == 'current_sheet':
                system.init_current_sheet(width=config.get('width', 0.2))
            elif init_type == 'alfven_wave':
                system.init_alfven_wave(
                    amplitude=config.get('amplitude', 0.1),
                    k=config.get('wavenumber', 1)
                )
            else:
                system.init_orszag_tang()
            
            if verbose:
                print(f"      {system}")
                print(f"      Init type: {init_type}")
        
        # [2/9] Initialize solver
        with timer.time_section("solver_init"):
            if verbose:
                print("\n[2/9] Initializing JAX solver...")
            
            solver = MHDSolver(
                cfl=config.get('cfl', 0.4),
                use_gpu=config.get('use_gpu', False)
            )
            
            if verbose:
                print(f"      CFL={solver.cfl}, Backend={solver.backend}")
        
        # [3/9] Run simulation
        with timer.time_section("simulation"):
            if verbose:
                print("\n[3/9] Running simulation...")
            
            result = solver.run_with_diagnostics(
                system=system,
                t_end=config.get('t_end', 3.0),
                save_dt=config.get('save_dt', 0.05),
                verbose=verbose
            )
            
            if verbose:
                print(f"      ✓ {result['n_snapshots']} snapshots, {result['total_steps']} steps")
        
        # [4/9] Initial conservation metrics (for comparison)
        initial_cons = result['conservation_history'][0]
        
        # [5/9] Compute final comprehensive metrics
        final_metrics = None
        if config.get('compute_metrics', True):
            with timer.time_section("metrics"):
                if verbose:
                    print("\n[4/9] Computing final comprehensive metrics...")
                
                _, U_final = result['snapshots'][-1]
                dx = system.params.Lx / system.params.nx
                dy = system.params.Ly / system.params.ny
                dt_final = result['stability_history'][-1].get('dt', 0.01)
                
                final_metrics = compute_all_metrics(
                    U=U_final,
                    dx=dx,
                    dy=dy,
                    gamma=system.params.gamma,
                    conservation_initial=initial_cons,
                    dt=dt_final,
                    cfl=solver.cfl,
                    verbose=verbose
                )
                
                logger.log_final_metrics(final_metrics)
                
                if verbose:
                    print(f"\n      === KEY METRICS ===")
                    print(f"      Energy conservation error: {final_metrics.get('comp_energy_conservation_error', 0)*100:.4f}%")
                    print(f"      Mass conservation error: {final_metrics.get('comp_mass_conservation_error', 0)*100:.4f}%")
                    print(f"      Max |∇·B|: {final_metrics.get('cons_max_div_B', 0):.2e}")
                    print(f"      Dynamo efficiency: {final_metrics.get('comp_dynamo_efficiency_index', 0):.4f}")
                    print(f"      MHD complexity: {final_metrics.get('comp_mhd_complexity_index', 0):.4f}")
        else:
            if verbose:
                print("\n[4/9] Skipping metrics (disabled)")
        
        # [5/9] Save CSV data
        if config.get('save_csv', True):
            with timer.time_section("csv_save"):
                if verbose:
                    print("\n[5/9] Saving CSV data...")
                
                csv_dir = Path(output_dir) / "csv"
                csv_dir.mkdir(parents=True, exist_ok=True)
                
                # Conservation time series
                cons_file = csv_dir / f"{clean_name}_conservation.csv"
                DataHandler.save_metrics_csv(
                    cons_file, result['conservation_history'], result['times']
                )
                if verbose:
                    print(f"      Saved: {cons_file}")
                
                # Stability time series
                stab_file = csv_dir / f"{clean_name}_stability.csv"
                DataHandler.save_metrics_csv(
                    stab_file, result['stability_history'], result['times']
                )
                if verbose:
                    print(f"      Saved: {stab_file}")
                
                # Final metrics
                if final_metrics is not None:
                    metrics_file = csv_dir / f"{clean_name}_final_metrics.csv"
                    DataHandler.save_final_metrics_csv(metrics_file, final_metrics)
                    if verbose:
                        print(f"      Saved: {metrics_file}")
        
        # [6/9] Save NetCDF
        if config.get('save_netcdf', True):
            with timer.time_section("netcdf_save"):
                if verbose:
                    print("\n[6/9] Saving NetCDF data...")
                
                nc_dir = Path(output_dir) / "netcdf"
                nc_dir.mkdir(parents=True, exist_ok=True)
                
                nc_file = nc_dir / f"{clean_name}.nc"
                
                # Combine conservation and stability into all_metrics
                all_metrics = []
                for cons, stab in zip(result['conservation_history'], result['stability_history']):
                    combined = {}
                    combined.update(cons)
                    combined.update(stab)
                    all_metrics.append(combined)
                
                DataHandler.save_netcdf(
                    nc_file, result, config, all_metrics, final_metrics
                )
                
                if verbose:
                    print(f"      Saved: {nc_file}")
        
        # [7/9] Generate static visualization
        with timer.time_section("visualization"):
            if verbose:
                print("\n[7/9] Generating visualizations...")
            
            animator = Animator(
                fps=config.get('animation_fps', 20),
                dpi=config.get('png_dpi', 150)
            )
            
            if config.get('save_png', True):
                with timer.time_section("png_save"):
                    if verbose:
                        print("      Creating static plots...")
                    
                    fig_dir = Path(output_dir) / "figs"
                    fig_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Main field plot
                    png_file = fig_dir / f"{clean_name}_fields.png"
                    animator.create_static_plot(
                        result, png_file, scenario_name, final_metrics
                    )
                    if verbose:
                        print(f"      Saved: {png_file}")
                    
                    # Metrics evolution plot
                    metrics_png = fig_dir / f"{clean_name}_metrics.png"
                    animator.create_metrics_plot(
                        result['times'],
                        result['conservation_history'],
                        result['stability_history'],
                        metrics_png,
                        f"{scenario_name} - Diagnostics"
                    )
                    if verbose:
                        print(f"      Saved: {metrics_png}")
        
        # [8/9] Generate animation
        if config.get('save_gif', True):
            with timer.time_section("gif_save"):
                if verbose:
                    print("\n[8/9] Creating animation...")
                
                gif_dir = Path(output_dir) / "gifs"
                gif_dir.mkdir(parents=True, exist_ok=True)
                
                gif_file = gif_dir / f"{clean_name}.gif"
                animator.create_animation(
                    result, gif_file, scenario_name, verbose=verbose
                )
                
                if verbose:
                    print(f"      Saved: {gif_file}")
        
        timer.stop("total")
        logger.log_timing(timer.get_times())
        
        # [9/9] Summary
        sim_time = timer.times.get('simulation', 0)
        metrics_time = timer.times.get('metrics', 0)
        total_time = timer.times.get('total', 0)
        
        if verbose:
            print(f"\n[9/9] SIMULATION COMPLETED")
            print(f"{'=' * 70}")
            
            if final_metrics is not None:
                print(f"  Conservation quality: {final_metrics.get('comp_conservation_quality', 0):.4f}")
                print(f"  Spectral index (kinetic): {final_metrics.get('turb_spectral_index_kinetic', np.nan):.2f}")
                print(f"  Spectral index (magnetic): {final_metrics.get('turb_spectral_index_magnetic', np.nan):.2f}")
            
            print(f"  Simulation time: {sim_time:.2f} s")
            if metrics_time > 0:
                print(f"  Metrics computation: {metrics_time:.2f} s")
            print(f"  Total time: {total_time:.2f} s")
            
            if logger.warnings:
                print(f"  Warnings: {len(logger.warnings)}")
            if logger.errors:
                print(f"  Errors: {len(logger.errors)}")
            
            print(f"{'=' * 70}\n")
    
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"SIMULATION FAILED")
            print(f"  Error: {str(e)}")
            print(f"{'=' * 70}\n")
        
        raise
    
    finally:
        logger.finalize()


def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Baron-Sakender: 2D Ideal MHD Solver',
        epilog='Example: baron-sakender case1 --gpu'
    )
    
    parser.add_argument(
        'case',
        nargs='?',
        choices=['case1', 'case2', 'case3', 'case4'],
        help='Test case to run (case1-4)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to custom configuration file'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run all test cases sequentially'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='outputs',
        help='Output directory for results (default: outputs)'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU acceleration (requires JAX with CUDA)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode (minimal output)'
    )
    
    parser.add_argument(
        '--no-gif',
        action='store_true',
        help='Skip GIF animation generation'
    )
    
    parser.add_argument(
        '--no-metrics',
        action='store_true',
        help='Skip metrics computation'
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if verbose:
        print_header()
    
    # Custom config
    if args.config:
        config = ConfigManager.load(args.config)
        if args.no_gif:
            config['save_gif'] = False
        if args.no_metrics:
            config['compute_metrics'] = False
        run_scenario(config, args.output_dir, verbose, args.gpu)
    
    # All cases
    elif args.all:
        configs_dir = Path(__file__).parent.parent.parent / 'configs'
        
        if not configs_dir.exists():
            configs_dir = Path(__file__).parent.parent.parent.parent / 'configs'
        
        config_files = sorted(configs_dir.glob('case*.txt'))
        
        if not config_files:
            print("ERROR: No configuration files found in configs/")
            print(f"Searched in: {configs_dir}")
            sys.exit(1)
        
        for i, cfg_file in enumerate(config_files, 1):
            if verbose:
                print(f"\n[Case {i}/{len(config_files)}] Running {cfg_file.stem}...")
            
            config = ConfigManager.load(str(cfg_file))
            if args.no_gif:
                config['save_gif'] = False
            if args.no_metrics:
                config['compute_metrics'] = False
            run_scenario(config, args.output_dir, verbose, args.gpu)
    
    # Single case
    elif args.case:
        case_map = {
            'case1': 'case1_orszag_tang',
            'case2': 'case2_strong_magnetic',
            'case3': 'case3_current_sheet',
            'case4': 'case4_alfven_wave',
        }
        
        cfg_name = case_map[args.case]
        
        search_paths = [
            Path(__file__).parent.parent.parent / 'configs',
            Path(__file__).parent.parent.parent.parent / 'configs',
            Path.cwd() / 'configs',
        ]
        
        cfg_file = None
        for search_path in search_paths:
            potential_file = search_path / f'{cfg_name}.txt'
            if potential_file.exists():
                cfg_file = potential_file
                break
        
        if cfg_file is not None:
            config = ConfigManager.load(str(cfg_file))
            if args.no_gif:
                config['save_gif'] = False
            if args.no_metrics:
                config['compute_metrics'] = False
            run_scenario(config, args.output_dir, verbose, args.gpu)
        else:
            print(f"ERROR: Configuration file not found: {cfg_name}.txt")
            print(f"Searched in: {[str(p) for p in search_paths]}")
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == '__main__':
    main()
