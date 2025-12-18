# `baron-sakender`: JAX-Accelerated 2D Ideal MHD Solver

[![PyPI version](https://badge.fury.io/py/baron-sakender.svg)](https://badge.fury.io/py/baron-sakender)
[![Tests](https://github.com/sandyherho/baron-sakender/actions/workflows/tests.yml/badge.svg)](https://github.com/sandyherho/baron-sakender/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![JAX](https://img.shields.io/badge/JAX-%23FF6F00.svg?logo=google&logoColor=white)](https://github.com/google/jax)
[![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-%23150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![Pillow](https://img.shields.io/badge/Pillow-%23776AB0.svg?logo=python&logoColor=white)](https://pillow.readthedocs.io/)
[![netCDF4](https://img.shields.io/badge/netCDF4-%23004B87.svg)](https://unidata.github.io/netcdf4-python/)
[![tqdm](https://img.shields.io/badge/tqdm-%23FFC107.svg?logo=tqdm&logoColor=black)](https://tqdm.github.io/)
[![pytest](https://img.shields.io/badge/pytest-%230A9EDC.svg?logo=pytest&logoColor=white)](https://pytest.org/)

JAX-accelerated Python library for simulating 2D ideal magnetohydrodynamics (MHD) with comprehensive conservation, stability, turbulence, and information-theoretic metrics.

> *This library is named after **Baron Sakender**, the protagonist of *Serat Baron Sakender*, a classical Javanese literary work blending history, legend, and myth. Depicted as a noble European knight with supernatural powers, Baron Sakender journeyed to Java where he eventually submitted to Panembahan Senopati of Mataram and served as a royal garden keeper. The text symbolically incorporates European colonial presence into Javanese cosmology while affirming local sovereignty—a fitting namesake for software that harmonizes the dynamic interplay between fluid motion and magnetic fields in plasma physics.*

## Governing Equations

The 2D ideal MHD system is expressed as conservation laws for the state vector $\mathbf{U} = (\rho, \rho v_{i}, B_{i}, E)^{\top}$:

$$\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}^{j}}{\partial x^{j}} = \mathbf{0}$$

where the flux tensor $\mathbf{F}^{j}$ contains the following components:

**Mass conservation:**

$$\frac{\partial \rho}{\partial t} + \frac{\partial (\rho v^{j})}{\partial x^{j}} = 0$$

**Momentum conservation:**

$$\frac{\partial (\rho v_{i})}{\partial t} + \frac{\partial}{\partial x^{j}} \left( \rho v_{i} v^{j} + P^{*} \delta_{i}^{j} - B_{i} B^{j} \right) = 0$$

**Magnetic induction:**

$$\frac{\partial B_{i}}{\partial t} + \frac{\partial}{\partial x^{j}} \left( v^{j} B_{i} - v_{i} B^{j} \right) = 0$$

**Energy conservation:**

$$\frac{\partial E}{\partial t} + \frac{\partial}{\partial x^{j}} \left[ (E + P^{*}) v^{j} - B^{j} (v_{k} B^{k}) \right] = 0$$

with the total pressure:

$$P^{*} = p + \frac{1}{2} B_{k} B^{k}$$

and total energy density:

$$E = \frac{1}{2} \rho v_{k} v^{k} + \frac{p}{\gamma - 1} + \frac{1}{2} B_{k} B^{k}$$

### Divergence Cleaning

To enforce the solenoidal constraint $\frac{\partial B^{i}}{\partial x^{i}} = 0$, we employ hyperbolic-parabolic divergence cleaning via an auxiliary scalar field $\psi$:

$$\frac{\partial B_{i}}{\partial t} + \frac{\partial}{\partial x^{j}}(v^{j} B_{i} - v_{i} B^{j}) + \frac{\partial \psi}{\partial x^{i}} = 0$$

$$\frac{\partial \psi}{\partial t} + c_{h}^{2} \frac{\partial B^{i}}{\partial x^{i}} = -\frac{c_{h}^{2}}{c_{p}^{2}} \psi$$

where $c_{h}$ is the hyperbolic cleaning speed and $c_{p}$ controls parabolic damping.

### Numerical Scheme

The spatial discretization employs a finite volume method with the HLL approximate Riemann solver:

$$\mathbf{F}^{\mathrm{HLL}} = \frac{S^{+} \mathbf{F}^{L} - S^{-} \mathbf{F}^{R} + S^{+} S^{-} (\mathbf{U}^{R} - \mathbf{U}^{L})}{S^{+} - S^{-}}$$

where $S^{\pm}$ are the fastest left/right-going wave speeds estimated from the fast magnetosonic velocity:

$$c_{f} = \sqrt{c_{s}^{2} + v_{A}^{2}}, \quad c_{s} = \sqrt{\frac{\gamma p}{\rho}}, \quad v_{A} = \sqrt{\frac{B_{k} B^{k}}{\rho}}$$

Time integration uses second-order Runge-Kutta (Heun's method):

$$\mathbf{U}^{n+1} = \mathbf{U}^{n} + \frac{\Delta t}{2}\left[\mathbf{L}(\mathbf{U}^{n}) + \mathbf{L}\left(\mathbf{U}^{n} + \Delta t \, \mathbf{L}(\mathbf{U}^{n})\right)\right]$$

with adaptive $\Delta t$ satisfying the CFL condition:

$$\Delta t \leq C \cdot \frac{\min(\Delta x, \Delta y)}{\max(|v^{i}| + c_{f})}$$

## Features

- **JAX-Accelerated**: Automatic GPU/CPU backend selection with JIT compilation
- **HLL Riemann Solver**: Robust shock-capturing for MHD discontinuities
- **Dedner Divergence Cleaning**: Hyperbolic/parabolic cleaning to maintain $\frac{\partial B^{i}}{\partial x^{i}} \approx 0$
- **Comprehensive Metrics**:
  - Conservation: mass, momentum, energy, cross-helicity, $\max\left|\frac{\partial B^{i}}{\partial x^{i}}\right|$
  - Stability: Mach numbers, plasma $\beta$, CFL condition
  - Turbulence: energy spectra $E(k)$, structure functions $S_{p}(\ell)$, intermittency
  - Information-theoretic: Shannon entropy $H$, mutual information $I(v; B)$, complexity $C$
  - Novel composite: dynamo efficiency, cascade efficiency, alignment indices
- **Physical Units**: CF-compliant NetCDF output with SI reference units
- **Beautiful Visualizations**: Dark-themed plots and animations

## Installation

```bash
pip install baron-sakender          # From PyPI
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
| 1 | Orszag-Tang Vortex | $\rho = \gamma^{2}$, $v_{i} = (-\sin y, \sin x)$, $B_{i} = (-\sin y, \sin 2x)$ | Shock formation, spectral cascade |
| 2 | Strong Magnetic Field | Low $\beta = 0.1$ (magnetically dominated) | Alfvén dynamics |
| 3 | Current Sheet | Harris-like: $B_{x} = \tanh\left[\frac{x - L/2}{w}\right]$ | Reconnection, current structures |
| 4 | Alfvén Wave | Circularly polarized: $\delta B_{y} = \delta v_{y} = A \sin(kx)$ | Wave propagation, conservation |

## Metrics Overview

### Conservation Metrics

| Metric | Expression | Units |
|:-------|:-----------|:------|
| Total Mass | $M = \int \rho \, dV$ | $\rho_{0} L_{0}^{2}$ |
| Total Momentum | $P_{i} = \int \rho v_{i} \, dV$ | $\rho_{0} v_{A} L_{0}^{2}$ |
| Total Energy | $\mathcal{E} = \int E \, dV$ | $P_{0} L_{0}^{2}$ |
| Cross Helicity | $H_{c} = \int v_{i} B^{i} \, dV$ | $v_{A} B_{0} L_{0}^{2}$ |
| Divergence Error | $\max\left\lvert\frac{\partial B^{i}}{\partial x^{i}}\right\rvert$ | $B_{0} / L_{0}$ |

### Stability Metrics

| Metric | Expression |
|:-------|:-----------|
| Sonic Mach | $M_{s} = \frac{\lVert v \rVert}{c_{s}}$ |
| Alfvén Mach | $M_{A} = \frac{\lVert v \rVert}{v_{A}}$ |
| Fast Mach | $M_{f} = \frac{\lVert v \rVert}{c_{f}}$ |
| Plasma Beta | $\beta = \frac{2p}{B_{k} B^{k}}$ |
| CFL Number | $\nu = \frac{(\lvert v^{i} \rvert + c_{f}) \Delta t}{\Delta x^{i}}$ |

### Turbulence Metrics

| Metric | Expression |
|:-------|:-----------|
| Energy Spectrum | $E(k) = \frac{1}{2} \sum_{\lvert \mathbf{k}' \rvert = k} \lvert \hat{v}_{i}(\mathbf{k}') \rvert^{2}$ |
| Spectral Index | $\alpha$ where $E(k) \sim k^{\alpha}$ |
| Enstrophy | $\Omega = \int \omega_{i} \omega^{i} \, dV$ where $\omega_{i} = \epsilon_{ijk} \frac{\partial v^{k}}{\partial x^{j}}$ |
| Structure Function | $S_{p}(\ell) = \langle \lvert \delta v(\ell) \rvert^{p} \rangle$ |
| Elsässer Variables | $z_{i}^{\pm} = v_{i} \pm \frac{B_{i}}{\sqrt{\rho}}$ |
| Energy Imbalance | $\sigma_{c} = \frac{E^{+} - E^{-}}{E^{+} + E^{-}}$ |

### Information-Theoretic Metrics

| Metric | Expression |
|:-------|:-----------|
| Shannon Entropy | $H[f] = -\int p(f) \log p(f) \, df$ |
| Mutual Information | $I(v; B) = H[v] + H[B] - H[v, B]$ |
| Statistical Complexity | $C = H \cdot D_{\mathrm{JS}}$ (entropy $\times$ Jensen-Shannon disequilibrium) |
| Permutation Entropy | $H_{\pi} = -\sum_{\pi} p(\pi) \log p(\pi)$ |

### Composite Metrics

| Metric | Expression |
|:-------|:-----------|
| Dynamo Efficiency | $\eta_{d} = \frac{\lvert H_{c} \rvert}{\sqrt{E_{k} E_{m}}}$ |
| Cascade Efficiency | $\eta_{c} = \frac{E_{k > k_{*}}}{E_{k < k_{*}}}$ |
| Alignment Index | $\sigma_{A} = \frac{\langle v_{i} B^{i} \rangle}{\langle \lVert v \rVert \lVert B \rVert \rangle}$ |
| Intermittency Index | $\mathcal{I} = \frac{\partial \log F(\ell)}{\partial \log \ell}$ where $F = \frac{S_{4}}{S_{2}^{2}}$ |

## Output Files

- **CSV**: Conservation/stability time series, final metrics
- **NetCDF**: CF-compliant with all fields, metrics, and physical units
- **PNG**: High-resolution field plots and diagnostic evolution
- **GIF**: Animated simulation visualization

## Physical Units

The code uses normalized units with SI reference values:

| Quantity | Symbol | Default SI Value |
|:---------|:-------|:-----------------|
| Length | $L_{0}$ | $10^{6}$ m |
| Density | $\rho_{0}$ | $10^{-12}$ kg/m³ |
| Magnetic Field | $B_{0}$ | $10^{-9}$ T |
| Alfvén Velocity | $v_{A} = \frac{B_{0}}{\sqrt{\mu_{0} \rho_{0}}}$ | — |
| Alfvén Time | $\tau_{A} = \frac{L_{0}}{v_{A}}$ | — |
| Pressure | $P_{0} = \frac{B_{0}^{2}}{\mu_{0}}$ | — |

## License

MIT License

Copyright (c) 2025 Sandy H. S. Herho, Faiz R. Fajary, Nurjanna J. Trilaksono, Dasapta E. Irawan

## Citation

```bibtex
@software{baron_sakender_2025,
  title   = {{baron-sakender}: JAX-Accelerated 2D Ideal MHD Solver},
  author  = {Herho, Sandy H. S. and Fajary, Faiz R. and Trilaksono, Nurjanna J. and Irawan, Dasapta E.},
  year    = {2025},
  url     = {https://github.com/sandyherho/baron-sakender}
}
```
