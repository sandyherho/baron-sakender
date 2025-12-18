#!/usr/bin/env python
"""
Example: Advanced analysis with baron-sakender library.

This script demonstrates advanced features including:
- Different initial conditions
- Comprehensive turbulence analysis
- Information-theoretic metrics
- Energy spectra computation

Run with:
    python examples/advanced_analysis.py
"""

import numpy as np
from pathlib import Path
import pandas as pd

from baron_sakender import MHDSystem, MHDSolver
from baron_sakender import (
    compute_conservation_metrics,
    compute_stability_metrics,
    compute_turbulence_metrics,
    compute_information_metrics,
    compute_composite_metrics,
)
from baron_sakender.visualization.animator import Animator


def compare_initializations():
    """Compare different initialization methods."""
    print("\n" + "=" * 60)
    print("COMPARING INITIAL CONDITIONS")
    print("=" * 60)
    
    init_configs = [
        ('Orszag-Tang', 'orszag_tang', {}),
        ('Strong B-field', 'strong_magnetic', {'beta': 0.2}),
        ('Current Sheet', 'current_sheet', {'width': 0.25}),
        ('Alfvén Wave', 'alfven_wave', {'amplitude': 0.1}),
    ]
    
    results = []
    solver = MHDSolver(cfl=0.4, use_gpu=False)
    
    for name, init_type, kwargs in init_configs:
        print(f"\n  Running: {name}...")
        
        system = MHDSystem(nx=64, ny=64)
        
        if init_type == 'orszag_tang':
            system.init_orszag_tang(**kwargs)
        elif init_type == 'strong_magnetic':
            system.init_strong_magnetic(**kwargs)
        elif init_type == 'current_sheet':
            system.init_current_sheet(**kwargs)
        elif init_type == 'alfven_wave':
            system.init_alfven_wave(**kwargs)
        
        # Run short simulation
        result = solver.run_with_diagnostics(
            system, t_end=0.5, save_dt=0.1, verbose=False
        )
        
        # Get final metrics
        _, U_final = result['snapshots'][-1]
        cons_init = result['conservation_history'][0]
        cons_final = result['conservation_history'][-1]
        
        # Energy conservation error
        mass_err = abs(cons_final['total_mass'] - cons_init['total_mass']) / cons_init['total_mass']
        energy_err = abs(cons_final['total_energy'] - cons_init['total_energy']) / cons_init['total_energy']
        
        # Compute turbulence metrics
        turb = compute_turbulence_metrics(U_final, system.dx, system.dy, system.gamma)
        
        results.append({
            'Name': name,
            'Mass Error': f"{mass_err:.2e}",
            'Energy Error': f"{energy_err:.2e}",
            'Enstrophy': f"{turb['enstrophy']:.4f}",
            'Current J²': f"{turb['current_density_rms']**2:.4f}",
            'E_kin/E_mag': f"{turb['energy_ratio']:.4f}",
        })
    
    # Display results
    df = pd.DataFrame(results)
    print("\n  === COMPARISON ===")
    print(df.to_string(index=False))
    
    return df


def analyze_turbulence():
    """Detailed turbulence analysis of Orszag-Tang vortex."""
    print("\n" + "=" * 60)
    print("TURBULENCE ANALYSIS")
    print("=" * 60)
    
    # Initialize system
    system = MHDSystem(nx=128, ny=128)
    system.init_orszag_tang()
    
    solver = MHDSolver(cfl=0.4, use_gpu=False)
    
    print("\n  Running simulation to t=2.0...")
    result = solver.run_with_diagnostics(
        system, t_end=2.0, save_dt=0.2, verbose=False
    )
    
    print("  Computing turbulence evolution...")
    
    # Analyze turbulence at different times
    times_to_analyze = [0.0, 0.5, 1.0, 1.5, 2.0]
    turb_evolution = []
    
    for t, U in result['snapshots']:
        if any(abs(t - tt) < 0.01 for tt in times_to_analyze):
            turb = compute_turbulence_metrics(U, system.dx, system.dy, system.gamma)
            turb_evolution.append({
                'time': t,
                'E_kinetic': turb['kinetic_energy'],
                'E_magnetic': turb['magnetic_energy'],
                'enstrophy': turb['enstrophy'],
                'cross_helicity': turb.get('cross_helicity', 0),
                'spectral_index_k': turb.get('spectral_index_kinetic', np.nan),
            })
    
    df = pd.DataFrame(turb_evolution)
    print("\n  === TURBULENCE EVOLUTION ===")
    print(df.to_string(index=False))
    
    # Final state information metrics
    print("\n  Computing information-theoretic metrics for final state...")
    _, U_final = result['snapshots'][-1]
    info = compute_information_metrics(U_final, system.dx, system.dy, system.gamma)
    
    print("\n  === INFORMATION METRICS (t=2.0) ===")
    print(f"    Density entropy: {info['shannon_entropy_density']:.4f}")
    print(f"    Vorticity entropy: {info['shannon_entropy_vorticity']:.4f}")
    print(f"    Current entropy: {info['shannon_entropy_current']:.4f}")
    print(f"    Mutual information (v,B): {info['mutual_information_v_B']:.4f}")
    print(f"    Statistical complexity: {info['statistical_complexity']:.4f}")
    
    return turb_evolution


def analyze_energy_spectra():
    """Analyze energy spectra development."""
    print("\n" + "=" * 60)
    print("ENERGY SPECTRA ANALYSIS")
    print("=" * 60)
    
    system = MHDSystem(nx=256, ny=256)
    system.init_orszag_tang()
    
    solver = MHDSolver(cfl=0.4, use_gpu=False)
    
    print("\n  Running high-resolution simulation...")
    result = solver.run_with_diagnostics(
        system, t_end=1.0, save_dt=0.5, verbose=False
    )
    
    print("\n  === SPECTRAL ANALYSIS ===")
    
    for t, U in result['snapshots']:
        turb = compute_turbulence_metrics(U, system.dx, system.dy, system.gamma)
        
        k = turb['k_spectrum']
        E_k = turb['E_kinetic_spectrum']
        E_m = turb['E_magnetic_spectrum']
        
        # Find spectral indices in inertial range
        idx_inertial = (k > 5) & (k < 50)
        
        if np.sum(idx_inertial) > 3:
            # Power-law fit
            log_k = np.log10(k[idx_inertial])
            log_Ek = np.log10(E_k[idx_inertial] + 1e-20)
            log_Em = np.log10(E_m[idx_inertial] + 1e-20)
            
            alpha_k = np.polyfit(log_k, log_Ek, 1)[0]
            alpha_m = np.polyfit(log_k, log_Em, 1)[0]
            
            print(f"\n  t = {t:.2f}:")
            print(f"    Kinetic spectral index: {alpha_k:.2f}")
            print(f"    Magnetic spectral index: {alpha_m:.2f}")
            print(f"    (Kolmogorov: -5/3 ≈ -1.67, Iroshnikov-Kraichnan: -3/2 = -1.50)")


def main():
    print("=" * 60)
    print("baron-sakender: Advanced Analysis Examples")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("example_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Run analyses
    df_compare = compare_initializations()
    
    # Save comparison
    df_compare.to_csv(output_dir / "initialization_comparison.csv", index=False)
    print(f"\n  Saved comparison to: {output_dir / 'initialization_comparison.csv'}")
    
    turb_evolution = analyze_turbulence()
    
    # Save turbulence evolution
    df_turb = pd.DataFrame(turb_evolution)
    df_turb.to_csv(output_dir / "turbulence_evolution.csv", index=False)
    print(f"\n  Saved turbulence evolution to: {output_dir / 'turbulence_evolution.csv'}")
    
    analyze_energy_spectra()
    
    print("\n" + "=" * 60)
    print("Advanced analysis complete!")
    print("Check 'example_outputs' directory for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
