# Baron-Sakender

**JAX-Accelerated 2D Ideal MHD Solver**

A high-performance magnetohydrodynamics (MHD) simulation library using JAX for GPU/CPU acceleration.

Named after *Baron Sakender*, a character from Javanese literary tradition.

## Features

- **JAX Acceleration**: GPU/CPU computation with automatic differentiation
- **HLL Riemann Solver**: Robust shock-capturing scheme
- **Dedner Divergence Cleaning**: Maintains ∇·B ≈ 0 numerically
- **Comprehensive Diagnostics**: Conservation, stability, and turbulence metrics
- **Publication-Quality Visualization**: Dark-themed plots and GIF animations
- **CF-Compliant NetCDF Output**: Standard climate/forecast conventions

## Installation

```bash
pip install baron-sakender
```

Or from source:

```bash
git clone https://github.com/username/baron-sakender.git
cd baron-sakender
pip install -e .
```

### Dependencies

- Python ≥ 3.9
- JAX ≥ 0.4.0
- NumPy ≥ 1.21
- SciPy ≥ 1.7
- Matplotlib ≥ 3.5
- netCDF4 ≥ 1.5

## Quick Start

```python
from baron_sakender import MHDSystem, run_simulation

# Create Orszag-Tang vortex simulation
system = MHDSystem(
    nx=256, ny=256,
    Lx=2*np.pi, Ly=2*np.pi,
    gamma=5/3,
    init_type='orszag_tang'
)

# Run simulation
results = run_simulation(
    system,
    t_end=3.0,
    cfl=0.4,
    save_dt=0.1
)
```

## Physics

Baron-Sakender solves the 2D ideal MHD equations in conservative form:

| Equation | Description |
|----------|-------------|
| ∂ρ/∂t + ∇·(ρ**v**) = 0 | Mass conservation |
| ∂(ρ**v**)/∂t + ∇·(ρ**vv** + P\*I - **BB**) = 0 | Momentum conservation |
| ∂**B**/∂t - ∇×(**v**×**B**) + ∇ψ = 0 | Magnetic induction + cleaning |
| ∂E/∂t + ∇·((E+P\*)**v** - **B**(**v**·**B**)) = 0 | Energy conservation |
| ∂ψ/∂t + c_h²∇·**B** = -(c_h²/c_p²)ψ | Divergence cleaning |

Where:
- P\* = p + B²/2 (total pressure)
- E = ρv²/2 + p/(γ-1) + B²/2 (total energy density)
- ψ is the Dedner cleaning potential

### Characteristic Speeds

- **Alfvén**: v_A = |**B**|/√ρ
- **Sound**: c_s = √(γp/ρ)
- **Fast magnetosonic**: c_f = √(c_s² + v_A²)

## Test Cases

### 1. Orszag-Tang Vortex
Classic 2D MHD turbulence benchmark showcasing shock formation and nonlinear interactions.

```python
system = MHDSystem(init_type='orszag_tang')
```

### 2. Strong Magnetic Field
Low plasma-β regime (β = 0.1) for magnetically-dominated dynamics.

```python
system = MHDSystem(init_type='strong_magnetic')
```

### 3. Current Sheet
Harris current sheet configuration for magnetic reconnection studies.

```python
system = MHDSystem(init_type='current_sheet')
```

### 4. Alfvén Wave
Linear wave propagation test for code verification.

```python
system = MHDSystem(init_type='alfven_wave')
```

## Diagnostics

Baron-Sakender computes essential MHD diagnostics:

### Conservation Metrics
- Total mass, momentum, energy
- Cross helicity H_c = ∫**v**·**B** dV
- Magnetic flux
- Divergence constraint (max |∇·B|)

### Stability Metrics
- Mach numbers (sonic, Alfvén, fast)
- Plasma β = 2p/B²
- CFL numbers
- Positivity checks

### Turbulence Metrics
- Energy partition (kinetic, magnetic, thermal)
- Enstrophy ∫ω² dV
- Current density statistics
- Elsässer energies (z± = **v** ± **B**/√ρ)
- Intermittency (kurtosis)

## Output Formats

### NetCDF (CF-Compliant)
```python
# Save to NetCDF
handler.save_netcdf(U, t, metrics, 'output.nc')
```

### CSV Time Series
```python
# Save metrics history
handler.save_csv(metrics_history, 'metrics.csv')
```

## Visualization

Dark-themed, publication-quality plots:

```python
from baron_sakender.visualization import plot_fields, plot_diagnostics

# Field snapshots
fig = plot_fields(U, dx, dy, gamma, t, save_path='fields.png')

# Diagnostic time series
fig = plot_diagnostics(times, metrics_history, save_path='diagnostics.png')

# Create animation
create_animation(states, times, dx, dy, gamma, save_path='animation.gif')
```

## Configuration

Simulations can be configured via text files:

```ini
# config.txt
nx = 256
ny = 256
Lx = 6.2832
Ly = 6.2832
gamma = 1.6667
t_end = 3.0
cfl = 0.4
save_dt = 0.1
init_type = orszag_tang
```

## Physical Units

Default normalization (solar wind conditions):

| Quantity | Symbol | Reference Value |
|----------|--------|-----------------|
| Length | L₀ | 10⁶ m (1000 km) |
| Density | ρ₀ | 10⁻¹² kg/m³ |
| Magnetic field | B₀ | 10⁻⁹ T (1 nT) |
| Alfvén velocity | v_A | ~890 m/s |
| Alfvén time | τ_A | ~1124 s |

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run specific test categories:

```bash
# Unit tests only
pytest tests/test_metrics.py -v

# Integration tests
pytest tests/ -m integration -v
```

## Project Structure

```
baron-sakender/
├── src/baron_sakender/
│   ├── core/
│   │   ├── mhd_system.py      # Physical system & initial conditions
│   │   ├── integrator.py      # JAX-accelerated numerics
│   │   └── metrics.py         # Diagnostic computations
│   ├── io/
│   │   ├── config_manager.py  # Configuration parsing
│   │   └── data_handler.py    # NetCDF/CSV output
│   └── visualization/
│       └── animator.py        # Plotting & animation
├── tests/
│   ├── test_metrics.py
│   └── test_integrator.py
├── configs/                   # Example configurations
├── README.md
└── pyproject.toml
```

## References

1. Dedner, A., et al. (2002). "Hyperbolic Divergence Cleaning for the MHD Equations." *J. Comput. Phys.* 175, 645-673.

2. Orszag, S. A., & Tang, C.-M. (1979). "Small-scale structure of two-dimensional magnetohydrodynamic turbulence." *J. Fluid Mech.* 90, 129-143.

3. Biskamp, D. (2003). *Magnetohydrodynamic Turbulence*. Cambridge University Press.

4. Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*. Springer.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use Baron-Sakender in your research, please cite:

```bibtex
@software{baron_sakender,
  title = {Baron-Sakender: JAX-Accelerated 2D Ideal MHD Solver},
  author = {Herho, Sandy H. S.},
  year = {2024},
  url = {https://github.com/username/baron-sakender}
}
```

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- JAX team at Google for the excellent autodiff framework
- The plasma physics community for benchmark test cases
- Baron Sakender from Javanese literary tradition for the name inspiration
