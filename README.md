# `baron-sakender`: JAX-Accelerated 2D Ideal MHD Solver

[![DOI](https://zenodo.org/badge/1118564602.svg)](https://doi.org/10.5281/zenodo.18059201)
[![Tests](https://github.com/sandyherho/baron-sakender/actions/workflows/tests.yml/badge.svg)](https://github.com/sandyherho/baron-sakender/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/baron-sakender.svg)](https://pypi.org/project/baron-sakender/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![JAX](https://img.shields.io/badge/JAX-accelerated-9cf.svg)](https://github.com/google/jax)
[![NumPy](https://img.shields.io/badge/NumPy-%3C2.0-013243.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7%2B-8CAAE6.svg)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-11557c.svg)](https://matplotlib.org/)

---

> *Baron Sakender is a legendary figure in Javanese folklore, depicted as a European nobleman (often Spanish or Dutch) who voyages to Java to challenge local rulers. His significance lies in the Serat Baron Sakender, a semi-historical narrative interweaving European presence with Javanese cultural legitimacy. By portraying Sakender as a formidable outsider who ultimately submits to Javanese authority, these chronicles served as a sophisticated political instrument for Javanese monarchs to negotiate their sovereignty and preserve cultural primacy during the Dutch colonial era.*

<p align="center">
  <img src=".assets/case1_orszag_tang.gif" alt="Orszag-Tang Vortex Simulation" width="600">
</p>

---

A finite volume solver for the two-dimensional ideal magnetohydrodynamic (MHD) equations with hyperbolic-parabolic divergence cleaning. This implementation employs the HLL approximate Riemann solver and second-order Runge-Kutta time integration, with optional GPU acceleration via JAX.

## Governing Equations

The ideal MHD system describes the dynamics of electrically conducting fluids permeated by magnetic fields. In conservative form with the Dedner divergence cleaning formulation, the equations read:

$$
\partial_t \mathbf{U} + \partial_i \mathbf{F}^i = \mathbf{S}
$$

where $\partial_t \equiv \partial/\partial t$ denotes the temporal derivative, $\partial_i \equiv \partial/\partial x^i$ the spatial derivative with respect to coordinate $x^i$, $\mathbf{U}$ the state vector, $\mathbf{F}^i$ the flux tensor, and $\mathbf{S}$ the source term.

### State Vector

The conservative state vector $\mathbf{U} \in \mathbb{R}^7$ is defined as:

$$
\mathbf{U} = \begin{pmatrix} \rho \\ \rho v^x \\ \rho v^y \\ B^x \\ B^y \\ E \\ \psi \end{pmatrix}
$$

### Conservation Laws

**Mass conservation:**

$$
\partial_t \rho + \partial_i (\rho v^i) = 0
$$

**Momentum conservation:**

$$
\partial_t (\rho v^j) + \partial_i \left( \rho v^i v^j + P^* \delta^{ij} - B^i B^j \right) = 0
$$

where $\delta^{ij}$ is the Kronecker delta and $P^*$ denotes the total pressure:

$$
P^* = p + \frac{1}{2} B^k B_k
$$

**Magnetic induction with divergence cleaning:**

$$
\partial_t B^j + \partial_i \left( v^i B^j - B^i v^j \right) + \partial^j \psi = 0
$$

**Energy conservation:**

$$
\partial_t E + \partial_i \left[ (E + P^*) v^i - B^i (v_k B^k) \right] = 0
$$

**Divergence cleaning (Dedner et al. 2002):**

$$
\partial_t \psi + c_h^2 \, \partial_i B^i = -\frac{c_h^2}{c_p^2} \psi
$$

### Closure Relations

**Total energy density:**

$$
E = \frac{1}{2} \rho v^i v_i + \frac{p}{\gamma - 1} + \frac{1}{2} B^i B_i
$$

**Equation of state (ideal gas):**

$$
p = (\gamma - 1) \left( E - \frac{1}{2} \rho v^i v_i - \frac{1}{2} B^i B_i \right)
$$

### Characteristic Speeds

**Sound speed:**

$$
c_s = \sqrt{\frac{\gamma p}{\rho}}
$$

**Alfvén speed:**

$$
v_A = \frac{|B|}{\sqrt{\rho}} = \sqrt{\frac{B^i B_i}{\rho}}
$$

**Fast magnetosonic speed:**

$$
c_f = \sqrt{c_s^2 + v_A^2}
$$

## Variable Definitions

| Symbol | Description | SI Unit |
|--------|-------------|---------|
| $\rho$ | Mass density | [kg m⁻³] |
| $v^i$ | Velocity component | [m s⁻¹] |
| $B^i$ | Magnetic field component | [T] |
| $p$ | Thermal pressure | [Pa] |
| $P^*$ | Total pressure (thermal + magnetic) | [Pa] |
| $E$ | Total energy density | [J m⁻³] |
| $\psi$ | Divergence cleaning potential | [T m s⁻¹] |
| $\gamma$ | Adiabatic index | [—] |
| $c_s$ | Sound speed | [m s⁻¹] |
| $v_A$ | Alfvén speed | [m s⁻¹] |
| $c_f$ | Fast magnetosonic speed | [m s⁻¹] |
| $c_h$ | Hyperbolic cleaning speed | [m s⁻¹] |
| $c_p$ | Parabolic damping rate | [s⁻¹] |

### Normalization (Code Units)

The code employs dimensionless variables normalized by reference quantities:

| Reference | Symbol | Description |
|-----------|--------|-------------|
| $\rho_0$ | Reference density | [kg m⁻³] |
| $B_0$ | Reference magnetic field | [T] |
| $L_0$ | Reference length | [m] |
| $v_A = B_0 / \sqrt{\mu_0 \rho_0}$ | Alfvén velocity | [m s⁻¹] |
| $\tau_A = L_0 / v_A$ | Alfvén time | [s] |
| $P_0 = B_0^2 / \mu_0$ | Magnetic pressure | [Pa] |

where $\mu_0 = 4\pi \times 10^{-7}$ [H m⁻¹] is the vacuum permeability.

## Numerical Methods

- **Spatial discretization:** Finite volume method on uniform Cartesian grid
- **Riemann solver:** HLL (Harten-Lax-van Leer) approximate solver
- **Time integration:** RK2 (Heun's method), second-order accurate
- **Divergence cleaning:** Hyperbolic-parabolic formulation (Dedner et al. 2002)
- **Boundary conditions:** Periodic
- **Positivity preservation:** Density and pressure floors

## Installation

### Requirements

- Python 3.9 or later
- NumPy < 2.0 (for compatibility with compiled dependencies)

### From Source

```bash
git clone https://github.com/sandyherho/baron-sakender.git
cd baron-sakender
pip install -e ".[dev,netcdf]"
```

### Dependencies

**Core:**
- [NumPy](https://numpy.org/) — Array operations
- [SciPy](https://scipy.org/) — Scientific computing
- [JAX](https://github.com/google/jax) — Accelerated numerical computing
- [Matplotlib](https://matplotlib.org/) — Visualization
- [pandas](https://pandas.pydata.org/) — Data handling
- [tqdm](https://github.com/tqdm/tqdm) — Progress bars

**Optional:**
- [netCDF4](https://unidata.github.io/netcdf4-python/) — CF-compliant output
- [pytest](https://docs.pytest.org/) — Testing framework
- [black](https://github.com/psf/black) — Code formatting

## Usage

### Command Line Interface

```bash
# Run Orszag-Tang vortex (standard benchmark)
baron-sakender case1

# Run all test cases
baron-sakender --all

# Run with GPU acceleration
baron-sakender case1 --gpu

# Run from configuration file
baron-sakender -c configs/case1_orszag_tang.txt
```

### Python API

```python
from baron_sakender import MHDSystem, MHDSolver

# Initialize system
system = MHDSystem(nx=256, ny=256, gamma=5/3)
system.init_orszag_tang()

# Create solver and run
solver = MHDSolver(cfl=0.4, use_gpu=False)
result = solver.run_with_diagnostics(system, t_end=3.0, save_dt=0.1)

# Access results
times = result['times']
snapshots = result['snapshots']
conservation = result['conservation_history']
```

### Available Test Cases

| Case | Description | Configuration |
|------|-------------|---------------|
| `case1` | Orszag-Tang vortex | Standard MHD turbulence benchmark |
| `case2` | Strong magnetic field | Low plasma beta ($\beta = 0.1$) regime |
| `case3` | Current sheet | Harris-like configuration |
| `case4` | Alfvén wave | Wave propagation test |

## Output Formats

- **PNG:** Static field plots (density, magnetic pressure, current, vorticity)
- **GIF:** Animated density evolution
- **CSV:** Time series of conservation and stability metrics
- **NetCDF:** CF-1.8 compliant data with full metadata

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=baron_sakender --cov-report=html
```

## Project Structure

```
baron-sakender/
├── src/baron_sakender/
│   ├── core/
│   │   ├── mhd_system.py    # Physical system definition
│   │   ├── integrator.py    # JAX-accelerated solver
│   │   └── metrics.py       # Diagnostic computations
│   ├── io/
│   │   ├── config_manager.py
│   │   └── data_handler.py
│   ├── visualization/
│   │   └── animator.py
│   └── cli.py               # Command line interface
├── configs/                  # Configuration files
├── tests/                    # Test suite
└── pyproject.toml
```

## References

- Dedner, A., Kemm, F., Kröner, D., Munz, C.-D., Schnitzer, T., & Wesenberg, M. (2002). Hyperbolic divergence cleaning for the MHD equations. *Journal of Computational Physics*, 175(2), 645–673.
- Harten, A., Lax, P. D., & van Leer, B. (1983). On upstream differencing and Godunov-type schemes for hyperbolic conservation laws. *SIAM Review*, 25(1), 35–61.
- Orszag, S. A., & Tang, C.-M. (1979). Small-scale structure of two-dimensional magnetohydrodynamic turbulence. *Journal of Fluid Mechanics*, 90(1), 129–143.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Authors

Sandy H. S. Herho, Nurjanna J. Trilaksono

## Citation

If you use this software in your research, please cite:

```bibtex
@software{baron_sakender,
  author = {Herho, Sandy H. S. and Trilaksono, Nurjanna J.},
  title = {baron-sakender: JAX-Accelerated 2D Ideal MHD Solver},
  year = {2025},
  url = {https://github.com/sandyherho/baron-sakender}
}
```
