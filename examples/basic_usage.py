#!/usr/bin/env python
"""
Example: Basic usage of baron-sakender library.

This script demonstrates how to use the baron-sakender library
to simulate the Orszag-Tang vortex.

Run with:
    python examples/basic_usage.py
"""

import numpy as np
from pathlib import Path

# Import baron-sakender components
from baron_sakender import MHDSystem, MHDSolver
from baron_sakender import compute_all_metrics
from baron_sakender.io.data_handler import DataHandler
from baron_sakender.visualization.animator import Animator


def main():
    print("=" * 60)
    print("baron-sakender: 2D Ideal MHD Simulation")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("example_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Define the MHD system
    print("\n[1] Creating MHD system (Orszag-Tang vortex)...")
    system = MHDSystem(
        nx=128,
        ny=128,
        gamma=5.0/3.0,
        Lx=2*np.pi,
        Ly=2*np.pi,
    )
    system.init_orszag_tang()
    print(f"    {system}")
    
    # 2. Initialize the JAX-accelerated solver
    print("\n[2] Initializing solver...")
    solver = MHDSolver(
        cfl=0.4,
        use_gpu=False  # Set to True if you have GPU
    )
    print(f"    CFL={solver.cfl}, Backend={solver.backend}")
    
    # 3. Run simulation with diagnostics
    print("\n[3] Running simulation...")
    result = solver.run_with_diagnostics(
        system=system,
        t_end=1.0,
        save_dt=0.1,
        verbose=True
    )
    
    print(f"\n    Simulation complete!")
    print(f"    Snapshots: {result['n_snapshots']}")
    print(f"    Total steps: {result['total_steps']}")
    
    # 4. Compute comprehensive metrics
    print("\n[4] Computing metrics...")
    _, U_final = result['snapshots'][-1]
    initial_cons = result['conservation_history'][0]
    
    metrics = compute_all_metrics(
        U=U_final,
        dx=system.dx,
        dy=system.dy,
        gamma=system.gamma,
        conservation_initial=initial_cons,
        dt=0.01,
        cfl=0.4,
        verbose=True
    )
    
    print("\n    === KEY METRICS ===")
    print(f"    Mass conservation error: {metrics.get('comp_mass_conservation_error', 0)*100:.4f}%")
    print(f"    Energy conservation error: {metrics.get('comp_energy_conservation_error', 0)*100:.4f}%")
    print(f"    Max |∇·B|: {metrics.get('cons_max_div_B', 0):.2e}")
    print(f"    Dynamo efficiency: {metrics.get('comp_dynamo_efficiency_index', 0):.4f}")
    
    # 5. Save results
    print("\n[5] Saving results...")
    
    # Save metrics CSV
    metrics_file = output_dir / "orszag_tang_metrics.csv"
    DataHandler.save_final_metrics_csv(metrics_file, metrics)
    print(f"    Saved: {metrics_file}")
    
    # Save NetCDF
    nc_file = output_dir / "orszag_tang.nc"
    config = {'scenario_name': 'Orszag-Tang Example'}
    DataHandler.save_netcdf(nc_file, result, system, config, metrics)
    print(f"    Saved: {nc_file}")
    
    # 6. Create visualization
    print("\n[6] Creating visualizations...")
    animator = Animator(fps=10, dpi=150)
    
    # Static plot
    png_file = output_dir / "orszag_tang_final.png"
    animator.create_static_plot(result, png_file, "Orszag-Tang Vortex", metrics)
    print(f"    Saved: {png_file}")
    
    # Animation (optional)
    create_gif = input("\n    Create GIF animation? (y/n): ").lower().strip() == 'y'
    if create_gif:
        gif_file = output_dir / "orszag_tang_animation.gif"
        animator.create_animation(result, gif_file, "Orszag-Tang Vortex")
        print(f"    Saved: {gif_file}")
    
    print("\n" + "=" * 60)
    print("Done! Check 'example_outputs' directory for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
