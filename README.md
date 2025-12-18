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

> *This library is named after **Baron Sakender**, the protagonist of *Serat Baron Sakender*, a classical Javanese literary work blending history, legend, and myth. Depicted as a noble European knight with supernatural powers, Baron Sakender journeyed to Java where he eventually submitted to Panembahan Senopati of Mataram and served as a royal garden keeper. The text symbolically incorporates European colonial presence into Javanese cosmology while affirming local sovereignty—a fitting namesake for software that harmonizes the dynamic interplay between fluid motion and magnetic fields in plasma physics.*

## Governing Equations

The 2D ideal MHD system is expressed as conservation laws for the state vector $\mathbf{U} = (\rho, \rho v_i, B_i, E)^\top$:

$$\partial_t \mathbf{U} + \partial_j \mathbf{F}^j = \mathbf{0}$$

where the flux tensor $\mathbf{F}^j$ contains the following components:

**Mass conservation:**
$$\partial_t \rho + \partial_j (\rho v^j) = 0$$

**Momentum conservation:**
$$\partial_t (\rho v_i) + \partial_j \left( \rho v_i v^j + P^* \delta_i^j - B_i B^j \right) = 0$$

**Magnetic induction:**
$$\partial_t B_i + \partial_j \left( v^j B_i - v_i B^j \right) = 0$$

**Energy conservation:**
$$\partial_t E + \partial_j \left[ (E + P^*) v^j - B^j (v_k B^k) \right] = 0$$

with the total pressure $P^* = p + \frac{1}{2}B_k B^k$ and total energy density $E = \frac{1}{2}\rho v_k v^k + \frac{p}{\gamma - 1} + \frac{1}{2}B_k B^k$.

### Divergence Cleaning

To enforce the solenoidal constraint $\partial_i B^i = 0$, we employ hyperbolic-parabolic divergence cleaning via an auxiliary scalar field $\psi$:

$$\partial_t B_i + \partial_j(v^j B_i - v_i B^j) + \partial_i \psi = 0$$

$$\partial_t \psi + c_h^2 \partial_i B^i = -\frac{c_h^2}{c_p^2} \psi$$

where $c_h$ is the hyperbolic cleaning speed and $c_p$ controls parabolic damping.

### Numerical Scheme

The spatial discretization employs a finite volume method with the HLL approximate Riemann solver:

$$\mathbf{F}^{\text{HLL}} = \frac{S^+ \mathbf{F}^L - S^- \mathbf{F}^R + S^+ S^- (\mathbf{U}^R - \mathbf{U}^L)}{S^+ - S^-}$$

where $S^\pm$ are the fastest left/right-going wave speeds estimated from the fast magnetosonic velocity:

$$c_f = \sqrt{c_s^2 + v_A^2}, \quad c_s = \sqrt{\gamma p / \rho}, \quad v_A = \sqrt{B_k B^k / \rho}$$

Time integration uses second-order Runge-Kutta (Heun's method):

$$\mathbf{U}^{n+1} = \mathbf{U}^n + \frac{\Delta t}{2}\left[\mathbf{L}(\mathbf{U}^n) + \mathbf{L}(\mathbf{U}^n + \Delta t \, \mathbf{L}(\mathbf{U}^n))\right]$$

with adaptive $\Delta t$ satisfying the CFL condition $\Delta t \leq C \cdot \min(\Delta x, \Delta y) / \max(|v^i| + c_f)$.

## Features

- **JAX-Accelerated**: Automatic GPU/CPU backend selection with JIT compilation
- **HLL Riemann Solver**: Robust shock-capturing for MHD discontinuities
- **Dedner Divergence Cleaning**: Hyperbolic/parabolic cleaning to maintain $\partial_i B^i \approx 0$
- **Comprehensive Metrics**:
  - Conservation: mass, momentum, energy, cross-helicity, $\max|\partial_i B^i|$
  - Stability: Mach numbers, plasma $\beta$, CFL condition
  - Turbulence: energy spectra $E(k)$, structure functions $S_p(\ell)$, intermittency
  - Information-theoretic: Shannon entropy $H$, mutual information $I(v;B)$, complexity $C$
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
| 1 | Orszag-Tang Vortex | $\rho = \gamma^2$, $v_i = (-\sin y, \sin x)$, $B_i = (-\sin y, \sin 2x)$ | Shock formation, spectral cascade |
| 2 | Strong Magnetic Field | Low $\beta = 0.1$ (magnetically dominated) | Alfvén dynamics |
| 3 | Current Sheet | Harris-like: $B_x = \tanh[(x - L/2)/w]$ | Reconnection, current structures |
| 4 | Alfvén Wave | Circularly polarized: $\delta B_y = \delta v_y = A\sin(kx)$ | Wave propagation, conservation |

## Metrics Overview

### Conservation Metrics

| Metric | Expression | Units |
|:-------|:-----------|:------|
| Total Mass | $M = \int \rho \, dV$ | $\rho_0 L_0^2$ |
| Total Momentum | $P_i = \int \rho v_i \, dV$ | $\rho_0 v_A L_0^2$ |
| Total Energy | $\mathcal{E} = \int E \, dV$ | $P_0 L_0^2$ |
| Cross Helicity | $H_c = \int v_i B^i \, dV$ | $v_A B_0 L_0^2$ |
| Divergence Error | $\max|\partial_i B^i|$ | $B_0 / L_0$ |

### Stability Metrics

| Metric | Expression |
|:-------|:-----------|
| Sonic Mach | $M_s = \|v\| / c_s$ |
| Alfvén Mach | $M_A = \|v\| / v_A$ |
| Fast Mach | $M_f = \|v\| / c_f$ |
| Plasma $\beta$ | $\beta = 2p / (B_k B^k)$ |
| CFL Number | $\nu = (|v^i| + c_f) \Delta t / \Delta x^i$ |

### Turbulence Metrics

| Metric | Expression |
|:-------|:-----------|
| Energy Spectrum | $E(k) = \frac{1}{2} \sum_{|\mathbf{k}'|=k} |\hat{v}_i(\mathbf{k}')|^2$ |
| Spectral Index | $\alpha$ where $E(k) \sim k^\alpha$ |
| Enstrophy | $\Omega = \int \omega_i \omega^i \, dV$ where $\omega_i = \epsilon_{ijk} \partial^j v^k$ |
| Structure Function | $S_p(\ell) = \langle |\delta v(\ell)|^p \rangle$ |
| Elsässer Variables | $z_i^\pm = v_i \pm B_i / \sqrt{\rho}$ |
| Energy Imbalance | $\sigma_c = (E^+ - E^-) / (E^+ + E^-)$ |

### Information-Theoretic Metrics

| Metric | Expression |
|:-------|:-----------|
| Shannon Entropy | $H[f] = -\int p(f) \log p(f) \, df$ |
| Mutual Information | $I(v; B) = H[v] + H[B] - H[v, B]$ |
| Statistical Complexity | $C = H \cdot D_{JS}$ (entropy × Jensen-Shannon disequilibrium) |
| Permutation Entropy | $H_\pi = -\sum_\pi p(\pi) \log p(\pi)$ |

### Composite Metrics

| Metric | Expression |
|:-------|:-----------|
| Dynamo Efficiency | $\eta_d = |H_c| / \sqrt{E_k E_m}$ |
| Cascade Efficiency | $\eta_c = E_{k > k_*} / E_{k < k_*}$ |
| Alignment Index | $\sigma_A = \langle v_i B^i \rangle / \langle\|v\| \|B\|\rangle$ |
| Intermittency Index | $\mathcal{I} = \partial_{\log \ell} \log F(\ell)$ where $F = S_4 / S_2^2$ |

## Output Files

- **CSV**: Conservation/stability time series, final metrics
- **NetCDF**: CF-compliant with all fields, metrics, and physical units
- **PNG**: High-resolution field plots and diagnostic evolution
- **GIF**: Animated simulation visualization

## Physical Units

The code uses normalized units with SI reference values:

| Quantity | Symbol | Default SI Value |
|:---------|:-------|:-----------------|
| Length | $L_0$ | $10^6$ m |
| Density | $\rho_0$ | $10^{-12}$ kg/m³ |
| Magnetic Field | $B_0$ | $10^{-9}$ T |
| Alfvén Velocity | $v_A = B_0 / \sqrt{\mu_0 \rho_0}$ | — |
| Alfvén Time | $\tau_A = L_0 / v_A$ | — |
| Pressure | $P_0 = B_0^2 / \mu_0$ | — |

## License

MIT License

Copyright (c) 2026 Sandy H. S. Herho, Faiz R. Fajary, Nurjanna J. Trilaksono, Dasapta E. Irawan

## Citation

```bibtex
@software{baron_sakender_2026,
  title   = {\texttt{baron-sakender}: JAX-Accelerated 2D Ideal MHD Solver},
  author  = {Herho, Sandy H. S. and Fajary, Faiz R. and Trilaksono, Nurjanna J. and Irawan, Dasapta E.},
  year    = {2026},
  url     = {https://github.com/sandyherho/baron-sakender}
}
```
