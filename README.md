# `baron-sakender`: JAX-Accelerated 2D Ideal MHD Solver

[![Tests](https://github.com/sandyherho/baron-sakender/actions/workflows/tests.yml/badge.svg)](https://github.com/sandyherho/baron-sakender/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![JAX](https://img.shields.io/badge/JAX-%23FF6F00.svg?logo=google&logoColor=white)](https://github.com/google/jax)
[![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![netCDF4](https://img.shields.io/badge/netCDF4-%23004B87.svg)](https://unidata.github.io/netcdf4-python/)

JAX-accelerated Python library for simulating 2D ideal magnetohydrodynamics (MHD) with comprehensive conservation, stability, turbulence, and information-theoretic metrics.

> *This library is named after **Baron Sakéndér**, a legendary hero from the Bugis-Makassar epic traditions of South Sulawesi, Indonesia. Known in traditional literature as I La Galigo, the Baron Sakéndér tales represent one of the world's longest epic cycles, predating the Iliad and Odyssey. Baron Sakéndér embodies extraordinary strength, wisdom, and the harmonious balance between opposing forces—a fitting namesake for software that models the dynamic interplay of fluid motion and magnetic fields in plasma physics.*

## Governing Equations

The 2D ideal MHD equations in conservative form:

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0$$

$$\frac{\partial (\rho \mathbf{v})}{\partial t} + \nabla \cdot \left(\rho \mathbf{v}\mathbf{v} + P^* \mathbf{I} - \mathbf{B}\mathbf{B}\right) = 0$$

$$\frac{\partial \mathbf{B}}{\partial t} - \nabla \times (\mathbf{v} \times \mathbf{B}) = 0$$

$$\frac{\partial E}{\partial t} + \nabla \cdot \left[(E + P^*)\mathbf{v} - \mathbf{B}(\mathbf{v} \cdot \mathbf{B})\right] = 0$$

where $P^* = p + B^2/2$ is the total pressure and $E = \rho v^2/2 + p/(\gamma-1) + B^2/2$ is the total energy density.

## Features

- **JAX-Accelerated**: Automatic GPU/CPU backend selection with JIT compilation
- **HLL Riemann Solver**: Robust shock-capturing for MHD discontinuities
- **Comprehensive Metrics**:
  - Conservation: mass, momentum, energy, cross-helicity, div B
  - Stability: Mach numbers, plasma beta, CFL condition
  - Turbulence: energy spectra, structure functions, intermittency
  - Information-theoretic: Shannon entropy, mutual information, complexity
  - Novel composite: dynamo efficiency, cascade efficiency, alignment indices
- **Physical Units**: CF-compliant NetCDF output with SI reference units
- **Beautiful Visualizations**: Dark-themed plots and animations

## Installation

```bash
pip install baron-sakender          # From PyPI (when published)
pip install jax[cuda12]             # Optional: GPU support
```

Or install from source:

```bash
git clone https://github.com/sandyherho/baron-sakender.git
cd baron-sakender
pip install -e .
```

## Quick Start

**CLI:**
```bash
baron-sakender case1              # Standard Orszag-Tang
baron-sakender --all              # Run all 4 cases
baron-sakender case1 --gpu        # GPU acceleration
```

**Python API:**
```python
from baron_sakender import MHDSystem, MHDSolver, compute_all_metrics

# Initialize system
system = MHDSystem(nx=256, ny=256, gamma=5/3)
system.init_orszag_tang()

# Run simulation
solver = MHDSolver(cfl=0.4, use_gpu=False)
result = solver.run_with_diagnostics(system, t_end=3.0, save_dt=0.05)

# Compute comprehensive metrics
U_final = result['snapshots'][-1][1]
metrics = compute_all_metrics(
    U_final, system.dx, system.dy, system.gamma,
    conservation_initial=result['conservation_history'][0]
)

print(f"Energy conservation error: {metrics['comp_energy_conservation_error']*100:.4f}%")
print(f"Spectral index (kinetic): {metrics['turb_spectral_index_kinetic']:.2f}")
```

## Test Cases

| Case | Description | Initial Condition | Focus |
|:----:|:------------|:------------------|:------|
| 1 | Orszag-Tang Vortex | Classic turbulence test | Shock formation, spectral cascade |
| 2 | Strong Magnetic Field | Low beta (magnetically dominated) | Alfven dynamics |
| 3 | Current Sheet | Harris-like configuration | Reconnection, current structures |
| 4 | Alfven Wave | Circularly polarized wave | Wave propagation, conservation |

## Metrics Overview

### Conservation Metrics
| Metric | Description | Units |
|:-------|:------------|:------|
| Total Mass | Integral of rho dV | rho_0 L_0^2 |
| Total Energy | Integral of E dV | P_0 L_0^2 |
| Cross Helicity | Integral of v dot B dV | v_A B_0 L_0^2 |
| Max div B | Maximum divergence error | B_0/L_0 |

### Stability Metrics
| Metric | Description |
|:-------|:------------|
| Sonic Mach | v/c_s |
| Alfven Mach | v/v_A |
| Plasma beta | 2p/B^2 |
| CFL Number | (v+c_f) dt/dx |

### Turbulence Metrics
| Metric | Description |
|:-------|:------------|
| Energy Spectra | E(k) for kinetic and magnetic |
| Spectral Index | Power-law slope alpha in E(k) ~ k^alpha |
| Enstrophy | Integral of omega^2 dV |
| Cross Helicity | Normalized v dot B correlation |
| Elsasser Energies | z_pm = v pm B/sqrt(rho) |

### Information-Theoretic Metrics
| Metric | Description |
|:-------|:------------|
| Shannon Entropy | H = -Sum p log(p) for each field |
| Mutual Information | I(v;B) between velocity and magnetic |
| Statistical Complexity | C = H x D (entropy x disequilibrium) |
| Permutation Entropy | Pattern-based complexity measure |

### Novel Composite Metrics
| Metric | Description |
|:-------|:------------|
| Dynamo Efficiency | Cross-helicity normalized by energies |
| MHD Complexity | Combined entropy-complexity measure |
| Cascade Efficiency | Small-scale/large-scale energy ratio |
| Alignment Index | v dot B/(|v||B|) |
| Intermittency Index | Scale-dependent flatness variation |

## Output Files

- **CSV**: Conservation/stability time series, final metrics
- **NetCDF**: CF-compliant with all fields, metrics, and physical units
- **PNG**: High-resolution field plots and diagnostic evolution
- **GIF**: Animated simulation visualization

## Physical Units

The code uses normalized units with SI reference values:

| Quantity | Symbol | Default SI Value |
|:---------|:-------|:-----------------|
| Length | L_0 | 10^6 m (1000 km) |
| Density | rho_0 | 10^-12 kg/m^3 |
| Magnetic Field | B_0 | 10^-9 T (1 nT) |
| Alfven Velocity | v_A | B_0/sqrt(mu_0 rho_0) |
| Alfven Time | tau_A | L_0/v_A |

## License

MIT License

Copyright (c) 2025 Sandy H. S. Herho, Faiz R. Fajary, Dasapta E. Irawan

## Citation

```bibtex
@software{baron_sakender_2025,
  title   = {{baron-sakender}: JAX-Accelerated 2D Ideal MHD Solver},
  author  = {Herho, Sandy H. S. and Fajary, Faiz R. and Irawan, Dasapta E.},
  year    = {2025},
  url     = {https://github.com/sandyherho/baron-sakender}
}
```

## References

- Orszag, S. A., & Tang, C.-M. (1979). Small-scale structure of two-dimensional magnetohydrodynamic turbulence. *J. Fluid Mech.*, 90(1), 129-143.
- Stone, J. M., et al. (2008). Athena: A new code for astrophysical MHD. *ApJS*, 178(1), 137.
- Biskamp, D. (2003). *Magnetohydrodynamic Turbulence*. Cambridge University Press.
- Bruno, R., & Carbone, V. (2013). The solar wind as a turbulence laboratory. *Living Rev. Sol. Phys.*, 10, 2.
