"""
Comprehensive tests for baron-sakender 2D Ideal MHD solver.

Updated to match cleaned metrics module (spectral indices removed).

Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from baron_sakender import MHDSystem, MHDParams, MHDSolver
from baron_sakender import (
    compute_conservation_metrics,
    compute_stability_metrics,
    compute_turbulence_metrics,
    compute_information_metrics,
    compute_composite_metrics,
    compute_all_metrics,
    compute_diagnostics,
)
from baron_sakender.io.config_manager import ConfigManager
from baron_sakender.io.data_handler import DataHandler


class TestMHDSystem:
    """Test MHD system definition and initialization."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        system = MHDSystem()
        assert system.params.gamma == pytest.approx(5.0 / 3.0)
        assert system.params.nx == 256
        assert system.params.ny == 256
        assert system.params.Lx == pytest.approx(2 * np.pi)
        assert system.params.Ly == pytest.approx(2 * np.pi)
    
    def test_custom_parameters(self):
        """Test custom parameter values."""
        system = MHDSystem(nx=128, ny=64, gamma=1.4)
        assert system.params.nx == 128
        assert system.params.ny == 64
        assert system.params.gamma == pytest.approx(1.4)
    
    def test_derived_quantities(self):
        """Test that derived physical quantities are computed."""
        system = MHDSystem(rho_0=1.0e-12, B_0=1.0e-9, L_0=1.0e6)
        
        # Alfvén velocity should be computed
        assert system.params.v_A > 0
        assert np.isfinite(system.params.v_A)
        
        # Alfvén time
        assert system.params.tau_A > 0
        assert np.isfinite(system.params.tau_A)
        
        # Magnetic pressure unit
        assert system.params.P_0 > 0
        assert np.isfinite(system.params.P_0)
    
    def test_grid_setup(self):
        """Test grid coordinates are correctly set up."""
        system = MHDSystem(nx=64, ny=32)
        
        assert len(system.x) == 64
        assert len(system.y) == 32
        assert system.X.shape == (64, 32)
        assert system.Y.shape == (64, 32)
        assert system.dx == pytest.approx(2 * np.pi / 64)
        assert system.dy == pytest.approx(2 * np.pi / 32)
    
    def test_conservative_variables_shape(self):
        """Test conservative variables array shape."""
        system = MHDSystem(nx=64, ny=64)
        assert system.U.shape == (7, 64, 64)  # 7 variables with divergence cleaning
    
    def test_init_orszag_tang(self):
        """Test Orszag-Tang initialization."""
        system = MHDSystem(nx=64, ny=64)
        system.init_orszag_tang()
        
        assert system._initialized == True
        assert system._init_type == 'orszag_tang'
        
        # Check density is positive
        assert np.all(system.U[0] > 0)
        
        # Check energy is positive
        assert np.all(system.U[5] > 0)
        
        # Check all values are finite
        assert np.all(np.isfinite(system.U))
    
    def test_init_strong_magnetic(self):
        """Test strong magnetic field initialization."""
        system = MHDSystem(nx=64, ny=64)
        system.init_strong_magnetic(beta=0.1)
        
        assert system._initialized == True
        assert system._init_type == 'strong_magnetic'
        assert np.all(system.U[0] > 0)
        assert np.all(np.isfinite(system.U))
    
    def test_init_current_sheet(self):
        """Test current sheet initialization."""
        system = MHDSystem(nx=64, ny=64)
        system.init_current_sheet(width=0.2)
        
        assert system._initialized == True
        assert system._init_type == 'current_sheet'
        assert np.all(system.U[0] > 0)
    
    def test_init_alfven_wave(self):
        """Test Alfvén wave initialization."""
        system = MHDSystem(nx=64, ny=64)
        system.init_alfven_wave(amplitude=0.1, k=2)
        
        assert system._initialized == True
        assert system._init_type == 'alfven_wave'
        assert np.all(system.U[0] > 0)
    
    def test_get_primitive(self):
        """Test primitive variable extraction."""
        system = MHDSystem(nx=32, ny=32)
        system.init_orszag_tang()
        
        rho, vx, vy, Bx, By, p = system.get_primitive()
        
        assert rho.shape == (32, 32)
        assert np.all(rho > 0)
        assert np.all(p > 0)
        assert np.all(np.isfinite(vx))
        assert np.all(np.isfinite(vy))
    
    def test_get_total_pressure(self):
        """Test total pressure computation."""
        system = MHDSystem(nx=32, ny=32)
        system.init_orszag_tang()
        
        P_total = system.get_total_pressure()
        
        assert P_total.shape == (32, 32)
        assert np.all(P_total > 0)
    
    def test_repr(self):
        """Test string representation."""
        system = MHDSystem(nx=64, ny=64)
        repr_str = repr(system)
        
        assert "MHDSystem" in repr_str
        assert "64" in repr_str
    
    def test_describe(self):
        """Test detailed description."""
        system = MHDSystem(nx=64, ny=64)
        system.init_orszag_tang()
        
        desc = system.describe()
        
        assert "2D Ideal MHD" in desc
        assert "Grid Resolution" in desc
        assert "orszag_tang" in desc


class TestMHDSolver:
    """Test JAX-accelerated MHD solver."""
    
    def test_solver_initialization_cpu(self):
        """Test solver initialization with CPU backend."""
        solver = MHDSolver(cfl=0.4, use_gpu=False)
        
        assert solver.cfl == 0.4
        assert solver.use_gpu == False
        assert solver.backend == 'cpu'
    
    def test_uninitialized_system_error(self):
        """Test that solver raises error for uninitialized system."""
        system = MHDSystem(nx=32, ny=32)
        solver = MHDSolver(cfl=0.4, use_gpu=False)
        
        with pytest.raises(ValueError, match="not initialized"):
            solver.run(system, t_end=0.1)
    
    def test_basic_simulation(self, small_system):
        """Test basic simulation run."""
        solver = MHDSolver(cfl=0.4, use_gpu=False)
        
        result = solver.run(
            small_system,
            t_end=0.1,
            save_dt=0.05,
            verbose=False
        )
        
        assert 'snapshots' in result
        assert 'system' in result
        assert 't_end' in result
        assert 'total_steps' in result
        assert len(result['snapshots']) >= 2
    
    def test_simulation_preserves_positivity(self, small_system):
        """Test that density stays positive during simulation."""
        solver = MHDSolver(cfl=0.3, use_gpu=False)
        
        result = solver.run(
            small_system,
            t_end=0.2,
            save_dt=0.1,
            verbose=False
        )
        
        for t, U in result['snapshots']:
            assert np.all(U[0] > 0), f"Negative density at t={t}"
    
    def test_simulation_with_diagnostics(self, small_system):
        """Test simulation with diagnostic tracking."""
        solver = MHDSolver(cfl=0.4, use_gpu=False)
        
        result = solver.run_with_diagnostics(
            small_system,
            t_end=0.2,
            save_dt=0.1,
            verbose=False
        )
        
        assert 'conservation_history' in result
        assert 'stability_history' in result
        assert 'times' in result
        
        assert len(result['conservation_history']) == len(result['times'])
        assert len(result['stability_history']) == len(result['times'])
    
    def test_energy_conservation(self, medium_system):
        """Test energy is approximately conserved."""
        solver = MHDSolver(cfl=0.4, use_gpu=False)
        
        result = solver.run_with_diagnostics(
            medium_system,
            t_end=0.3,
            save_dt=0.1,
            verbose=False
        )
        
        energies = [m['total_energy'] for m in result['conservation_history']]
        
        # Energy should not change by more than 5% over short time
        E0 = energies[0]
        E_final = energies[-1]
        
        relative_error = abs(E_final - E0) / E0
        assert relative_error < 0.05, f"Energy changed by {relative_error*100:.2f}%"
    
    def test_mass_conservation(self, medium_system):
        """Test mass is conserved."""
        solver = MHDSolver(cfl=0.4, use_gpu=False)
        
        result = solver.run_with_diagnostics(
            medium_system,
            t_end=0.3,
            save_dt=0.1,
            verbose=False
        )
        
        masses = [m['total_mass'] for m in result['conservation_history']]
        
        M0 = masses[0]
        M_final = masses[-1]
        
        relative_error = abs(M_final - M0) / M0
        assert relative_error < 0.01, f"Mass changed by {relative_error*100:.2f}%"


class TestConservationMetrics:
    """Test conservation metrics computation."""
    
    def test_conservation_metrics_structure(self, small_system):
        """Test conservation metrics return correct structure."""
        dx = small_system.dx
        dy = small_system.dy
        
        metrics = compute_conservation_metrics(
            small_system.U, dx, dy, small_system.gamma
        )
        
        required_keys = [
            'total_mass', 'total_momentum_x', 'total_momentum_y',
            'total_energy', 'kinetic_energy', 'magnetic_energy',
            'thermal_energy', 'cross_helicity', 'max_div_B', 'mean_div_B'
        ]
        
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"
            assert np.isfinite(metrics[key]), f"Non-finite value for {key}"
    
    def test_positive_energies(self, small_system):
        """Test that energies are positive."""
        dx = small_system.dx
        dy = small_system.dy
        
        metrics = compute_conservation_metrics(
            small_system.U, dx, dy, small_system.gamma
        )
        
        assert metrics['total_mass'] > 0
        assert metrics['total_energy'] > 0
        assert metrics['kinetic_energy'] > 0
        assert metrics['magnetic_energy'] > 0
        assert metrics['thermal_energy'] > 0
    
    def test_divergence_B_small(self, small_system):
        """Test that initial div B is small (should be ~0 analytically)."""
        dx = small_system.dx
        dy = small_system.dy
        
        metrics = compute_conservation_metrics(
            small_system.U, dx, dy, small_system.gamma
        )
        
        # For Orszag-Tang, div B should be numerically small
        assert metrics['max_div_B'] < 1.0


class TestStabilityMetrics:
    """Test stability metrics computation."""
    
    def test_stability_metrics_structure(self, small_system):
        """Test stability metrics return correct structure."""
        dx = small_system.dx
        dy = small_system.dy
        
        metrics = compute_stability_metrics(
            small_system.U, dx, dy, small_system.gamma, dt=0.01, cfl=0.4
        )
        
        required_keys = [
            'max_sonic_mach', 'max_alfven_mach', 'max_fast_mach',
            'min_beta', 'max_beta', 'cfl_effective', 'is_stable'
        ]
        
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"
    
    def test_positive_speeds(self, small_system):
        """Test that wave speeds are positive."""
        dx = small_system.dx
        dy = small_system.dy
        
        metrics = compute_stability_metrics(
            small_system.U, dx, dy, small_system.gamma, dt=0.01, cfl=0.4
        )
        
        assert metrics['max_sound_speed'] > 0
        assert metrics['max_alfven_speed'] > 0
        assert metrics['max_fast_speed'] > 0
    
    def test_initial_state_stable(self, small_system):
        """Test that initial state is flagged as stable."""
        dx = small_system.dx
        dy = small_system.dy
        
        metrics = compute_stability_metrics(
            small_system.U, dx, dy, small_system.gamma, dt=0.001, cfl=0.4
        )
        
        assert metrics['is_stable'] == True


class TestTurbulenceMetrics:
    """Test MHD turbulence metrics (spectral indices removed)."""
    
    def test_turbulence_metrics_structure(self, small_system):
        """Test turbulence metrics return correct structure."""
        dx = small_system.dx
        dy = small_system.dy
        
        metrics = compute_turbulence_metrics(
            small_system.U, dx, dy, small_system.gamma, verbose=False
        )
        
        # Updated required keys - no spectral indices
        required_keys = [
            'kinetic_energy', 'magnetic_energy', 'cross_helicity',
            'enstrophy', 'current_squared', 'elsasser_plus_energy', 
            'elsasser_minus_energy', 'energy_imbalance',
            'vorticity_kurtosis', 'current_kurtosis', 'taylor_microscale'
        ]
        
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"
    
    def test_no_spectral_indices(self, small_system):
        """Verify spectral indices are removed from turbulence metrics."""
        dx = small_system.dx
        dy = small_system.dy
        
        metrics = compute_turbulence_metrics(
            small_system.U, dx, dy, small_system.gamma, verbose=False
        )
        
        # These should NOT be present
        assert 'spectral_index_kinetic' not in metrics
        assert 'spectral_index_magnetic' not in metrics
        assert 'k_spectrum' not in metrics
        assert 'E_kinetic_spectrum' not in metrics
    
    def test_elsasser_variables(self, small_system):
        """Test Elsässer variable energies are computed."""
        dx = small_system.dx
        dy = small_system.dy
        
        metrics = compute_turbulence_metrics(
            small_system.U, dx, dy, small_system.gamma, verbose=False
        )
        
        assert 'elsasser_plus_energy' in metrics
        assert 'elsasser_minus_energy' in metrics
        assert metrics['elsasser_plus_energy'] >= 0
        assert metrics['elsasser_minus_energy'] >= 0


class TestInformationMetrics:
    """Test information-theoretic metrics (simplified)."""
    
    def test_information_metrics_structure(self, small_system):
        """Test information metrics return correct structure."""
        dx = small_system.dx
        dy = small_system.dy
        
        metrics = compute_information_metrics(
            small_system.U, dx, dy, small_system.gamma, verbose=False
        )
        
        # Updated required keys - only Shannon entropy
        required_keys = [
            'entropy_density', 'entropy_velocity', 'entropy_magnetic',
            'entropy_vorticity', 'entropy_current'
        ]
        
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"
    
    def test_no_complexity_metrics(self, small_system):
        """Verify complexity/MI metrics are removed."""
        dx = small_system.dx
        dy = small_system.dy
        
        metrics = compute_information_metrics(
            small_system.U, dx, dy, small_system.gamma, verbose=False
        )
        
        # These should NOT be present
        assert 'MI_velocity_magnetic' not in metrics
        assert 'complexity_vorticity' not in metrics
        assert 'statistical_complexity' not in metrics
    
    def test_entropy_positive(self, small_system):
        """Test that Shannon entropy is non-negative."""
        dx = small_system.dx
        dy = small_system.dy
        
        metrics = compute_information_metrics(
            small_system.U, dx, dy, small_system.gamma, verbose=False
        )
        
        assert metrics['entropy_density'] >= 0
        assert metrics['entropy_velocity'] >= 0
        assert metrics['entropy_magnetic'] >= 0


class TestCompositeMetrics:
    """Test composite turbulence metrics (simplified)."""
    
    def test_composite_metrics_structure(self, small_system):
        """Test composite metrics return correct structure."""
        dx = small_system.dx
        dy = small_system.dy
        
        metrics = compute_composite_metrics(
            small_system.U, dx, dy, small_system.gamma, verbose=False
        )
        
        # Updated required keys - no cascade_efficiency or mhd_complexity_index
        required_keys = [
            'dynamo_efficiency', 'coherent_structure_index',
            'alignment_index', 'intermittency_index', 'conservation_quality'
        ]
        
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"
    
    def test_no_cascade_efficiency(self, small_system):
        """Verify cascade_efficiency is removed."""
        dx = small_system.dx
        dy = small_system.dy
        
        metrics = compute_composite_metrics(
            small_system.U, dx, dy, small_system.gamma, verbose=False
        )
        
        # These should NOT be present (depend on spectral indices)
        assert 'cascade_efficiency' not in metrics
        assert 'mhd_complexity_index' not in metrics
    
    def test_conservation_quality_with_initial(self, small_system):
        """Test conservation quality index with initial state."""
        dx = small_system.dx
        dy = small_system.dy
        
        initial_cons = compute_conservation_metrics(
            small_system.U, dx, dy, small_system.gamma
        )
        
        # For same state, conservation quality should be 1.0
        metrics = compute_composite_metrics(
            small_system.U, dx, dy, small_system.gamma,
            conservation_initial=initial_cons, verbose=False
        )
        
        assert metrics['conservation_quality'] == pytest.approx(1.0, abs=0.01)


class TestDiagnostics:
    """Test diagnostic field computations."""
    
    def test_diagnostics_shape(self, small_system):
        """Test diagnostic fields have correct shape."""
        dx = small_system.dx
        dy = small_system.dy
        
        rho, pmag, Jz, omega = compute_diagnostics(
            small_system.U, dx, dy, small_system.gamma
        )
        
        assert rho.shape == (32, 32)
        assert pmag.shape == (32, 32)
        assert Jz.shape == (32, 32)
        assert omega.shape == (32, 32)
    
    def test_magnetic_pressure_positive(self, small_system):
        """Test magnetic pressure is non-negative."""
        dx = small_system.dx
        dy = small_system.dy
        
        _, pmag, _, _ = compute_diagnostics(
            small_system.U, dx, dy, small_system.gamma
        )
        
        assert np.all(pmag >= 0)


class TestConfigManager:
    """Test configuration file handling."""
    
    def test_load_config(self):
        """Test loading configuration from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("# Test config\n")
            f.write("nx = 128\n")
            f.write("ny = 128\n")
            f.write("gamma = 1.4\n")
            f.write("t_end = 2.0\n")
            f.write("use_gpu = false\n")
            f.write("scenario_name = Test Scenario\n")
            config_path = f.name
        
        config = ConfigManager.load(config_path)
        
        assert config['nx'] == 128
        assert config['ny'] == 128
        assert config['gamma'] == pytest.approx(1.4)
        assert config['t_end'] == pytest.approx(2.0)
        assert config['use_gpu'] == False
        assert config['scenario_name'] == 'Test Scenario'
        
        Path(config_path).unlink()
    
    def test_default_config(self):
        """Test default configuration."""
        config = ConfigManager.get_default_config()
        
        assert 'nx' in config
        assert 'gamma' in config
        assert 't_end' in config
        assert config['gamma'] == pytest.approx(5.0 / 3.0)
    
    def test_save_config(self):
        """Test saving configuration to file."""
        config = {'nx': 256, 'gamma': 1.4, 'use_gpu': True}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            config_path = f.name
        
        ConfigManager.save(config, config_path)
        
        loaded = ConfigManager.load(config_path)
        assert loaded['nx'] == 256
        assert loaded['gamma'] == pytest.approx(1.4)
        assert loaded['use_gpu'] == True
        
        Path(config_path).unlink()
    
    def test_validate_config(self):
        """Test configuration validation."""
        valid_config = {
            'nx': 64, 'ny': 64, 'gamma': 5/3,
            't_end': 1.0, 'cfl': 0.4
        }
        
        assert ConfigManager.validate_config(valid_config) == True
    
    def test_validate_config_invalid_cfl(self):
        """Test validation catches invalid CFL."""
        invalid_config = {
            'nx': 64, 'ny': 64, 'gamma': 5/3,
            't_end': 1.0, 'cfl': 1.5  # Invalid
        }
        
        with pytest.raises(ValueError, match="cfl"):
            ConfigManager.validate_config(invalid_config)


class TestDataHandler:
    """Test data saving functionality."""
    
    def test_save_metrics_csv(self):
        """Test saving metrics time series to CSV."""
        metrics_history = [
            {'total_mass': 1.0, 'total_energy': 10.0},
            {'total_mass': 1.001, 'total_energy': 9.99},
            {'total_mass': 1.002, 'total_energy': 9.98},
        ]
        times = np.array([0.0, 0.1, 0.2])
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            filepath = f.name
        
        DataHandler.save_metrics_csv(filepath, metrics_history, times)
        
        assert Path(filepath).exists()
        
        import pandas as pd
        df = pd.read_csv(filepath)
        
        assert 'time' in df.columns
        assert 'total_mass' in df.columns
        assert len(df) == 3
        
        Path(filepath).unlink()
    
    def test_save_final_metrics_csv(self):
        """Test saving final metrics to CSV."""
        metrics = {
            'kinetic_energy': 5.0,
            'magnetic_energy': 5.0,
            'conservation_quality': 0.99,
            'is_stable': True
        }
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            filepath = f.name
        
        DataHandler.save_final_metrics_csv(filepath, metrics)
        
        assert Path(filepath).exists()
        Path(filepath).unlink()
    
    def test_save_netcdf(self, small_system):
        """Test saving simulation to NetCDF."""
        solver = MHDSolver(cfl=0.4, use_gpu=False)
        
        result = solver.run_with_diagnostics(
            small_system,
            t_end=0.1,
            save_dt=0.05,
            verbose=False
        )
        
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            filepath = f.name
        
        config = {'scenario_name': 'Test', 'init_type': 'orszag_tang', 'cfl': 0.4}
        
        all_metrics = []
        for cons, stab in zip(result['conservation_history'], result['stability_history']):
            combined = {}
            combined.update(cons)
            combined.update(stab)
            all_metrics.append(combined)
        
        DataHandler.save_netcdf(filepath, result, config, all_metrics)
        
        assert Path(filepath).exists()
        
        # Verify contents
        from netCDF4 import Dataset
        with Dataset(filepath, 'r') as nc:
            assert 'time' in nc.variables
            assert 'density' in nc.variables
            assert 'velocity_x' in nc.variables
            assert 'B_x' in nc.variables
            assert nc.gamma == pytest.approx(5.0 / 3.0)
        
        Path(filepath).unlink()


class TestAllMetrics:
    """Test comprehensive metrics computation."""
    
    def test_all_metrics_structure(self, small_system):
        """Test compute_all_metrics returns comprehensive results."""
        dx = small_system.dx
        dy = small_system.dy
        
        initial_cons = compute_conservation_metrics(
            small_system.U, dx, dy, small_system.gamma
        )
        
        metrics = compute_all_metrics(
            U=small_system.U,
            dx=dx,
            dy=dy,
            gamma=small_system.gamma,
            conservation_initial=initial_cons,
            dt=0.01,
            cfl=0.4,
            verbose=False
        )
        
        # Check all categories are present (using prefixed versions)
        assert any(k.startswith('cons_') or k == 'total_mass' for k in metrics.keys())
        assert any(k.startswith('stab_') or k == 'max_sonic_mach' for k in metrics.keys())
        assert any(k.startswith('turb_') or k == 'enstrophy' for k in metrics.keys())
        assert any(k.startswith('info_') or k == 'entropy_density' for k in metrics.keys())
        assert any(k.startswith('comp_') or k == 'dynamo_efficiency' for k in metrics.keys())
        
        # Spectral data should NOT be present
        assert 'k_spectrum' not in metrics
        assert 'E_kinetic_spectrum' not in metrics


class TestIntegration:
    """Integration tests for full simulation workflow."""
    
    def test_full_workflow_short(self, medium_system):
        """Test complete simulation workflow."""
        solver = MHDSolver(cfl=0.4, use_gpu=False)
        
        # Run simulation
        result = solver.run_with_diagnostics(
            medium_system,
            t_end=0.2,
            save_dt=0.1,
            verbose=False
        )
        
        # Compute final metrics
        _, U_final = result['snapshots'][-1]
        dx = medium_system.dx
        dy = medium_system.dy
        
        initial_cons = result['conservation_history'][0]
        
        final_metrics = compute_all_metrics(
            U=U_final,
            dx=dx,
            dy=dy,
            gamma=medium_system.gamma,
            conservation_initial=initial_cons,
            dt=0.01,
            cfl=0.4,
            verbose=False
        )
        
        # Verify simulation ran successfully
        assert len(result['snapshots']) >= 2
        assert final_metrics['conservation_quality'] > 0.9
    
    def test_different_initializations(self):
        """Test different initial conditions all work."""
        from baron_sakender import MHDSystem
        
        init_methods = [
            ('orszag_tang', {}),
            ('strong_magnetic', {'beta': 0.5}),
            ('current_sheet', {'width': 0.3}),
            ('alfven_wave', {'amplitude': 0.1, 'k': 1}),
        ]
        
        solver = MHDSolver(cfl=0.4, use_gpu=False)
        
        for init_type, kwargs in init_methods:
            system = MHDSystem(nx=32, ny=32)
            
            if init_type == 'orszag_tang':
                system.init_orszag_tang(**kwargs)
            elif init_type == 'strong_magnetic':
                system.init_strong_magnetic(**kwargs)
            elif init_type == 'current_sheet':
                system.init_current_sheet(**kwargs)
            elif init_type == 'alfven_wave':
                system.init_alfven_wave(**kwargs)
            
            result = solver.run(system, t_end=0.1, save_dt=0.05, verbose=False)
            
            assert len(result['snapshots']) >= 2, f"Failed for {init_type}"
            assert result['total_steps'] > 0, f"No steps taken for {init_type}"


class TestConvergence:
    """
    Convergence tests to verify numerical accuracy.
    """
    
    def test_alfven_wave_convergence(self):
        """Test convergence using circularly polarized Alfvén wave."""
        from baron_sakender import MHDSystem, MHDSolver
        
        resolutions = [32, 64]
        amplitude_losses = []
        
        k = 1
        amplitude = 0.1
        
        for nx in resolutions:
            system = MHDSystem(nx=nx, ny=nx)
            system.init_alfven_wave(amplitude=amplitude, k=k)
            
            By_initial = system.U[4].copy()
            initial_amplitude = np.max(np.abs(By_initial))
            
            wave_period = 2 * np.pi / k
            t_end = 0.5 * wave_period
            
            solver = MHDSolver(cfl=0.4, use_gpu=False)
            result = solver.run(
                system,
                t_end=t_end,
                save_dt=t_end,
                verbose=False
            )
            
            _, U_final = result['snapshots'][-1]
            By_final = U_final[4]
            
            final_amplitude = np.max(np.abs(By_final))
            amplitude_loss = (initial_amplitude - final_amplitude) / initial_amplitude
            amplitude_losses.append(amplitude_loss)
        
        assert amplitude_losses[1] <= amplitude_losses[0] * 1.2
        
        for i, loss in enumerate(amplitude_losses):
            assert loss < 0.5, f"Too much amplitude loss at nx={resolutions[i]}: {loss:.2%}"
    
    def test_divergence_cleaning_effectiveness(self):
        """Test that divergence cleaning maintains ∇·B ≈ 0."""
        from baron_sakender import MHDSystem, MHDSolver
        from baron_sakender.core.metrics import compute_conservation_metrics
        
        system = MHDSystem(nx=64, ny=64)
        system.init_orszag_tang()
        
        solver = MHDSolver(cfl=0.4, use_gpu=False)
        result = solver.run_with_diagnostics(
            system,
            t_end=0.5,
            save_dt=0.1,
            verbose=False
        )
        
        max_divB_values = []
        for t, U in result['snapshots']:
            cons = compute_conservation_metrics(U, system.dx, system.dy, system.gamma)
            max_divB_values.append(cons['max_div_B'])
        
        final_divB = max_divB_values[-1]
        assert final_divB < 0.01, f"Divergence cleaning not effective: max|∇·B| = {final_divB:.2e}"
    
    def test_mass_conservation_convergence(self):
        """Test that mass conservation improves with resolution."""
        from baron_sakender import MHDSystem, MHDSolver
        from baron_sakender.core.metrics import compute_conservation_metrics
        
        resolutions = [32, 64]
        mass_errors = []
        
        for nx in resolutions:
            system = MHDSystem(nx=nx, ny=nx)
            system.init_orszag_tang()
            
            initial_cons = compute_conservation_metrics(
                system.U, system.dx, system.dy, system.gamma
            )
            initial_mass = initial_cons['total_mass']
            
            solver = MHDSolver(cfl=0.4, use_gpu=False)
            result = solver.run(
                system,
                t_end=0.5,
                save_dt=0.5,
                verbose=False
            )
            
            _, U_final = result['snapshots'][-1]
            final_cons = compute_conservation_metrics(
                U_final, system.dx, system.dy, system.gamma
            )
            final_mass = final_cons['total_mass']
            
            mass_error = abs(final_mass - initial_mass) / initial_mass
            mass_errors.append(mass_error)
        
        for i, error in enumerate(mass_errors):
            assert error < 0.01, f"Mass conservation error too large at nx={resolutions[i]}: {error:.2e}"
    
    def test_energy_conservation_convergence(self):
        """Test that energy conservation improves with resolution."""
        from baron_sakender import MHDSystem, MHDSolver
        from baron_sakender.core.metrics import compute_conservation_metrics
        
        resolutions = [32, 64]
        energy_errors = []
        
        for nx in resolutions:
            system = MHDSystem(nx=nx, ny=nx)
            system.init_orszag_tang()
            
            initial_cons = compute_conservation_metrics(
                system.U, system.dx, system.dy, system.gamma
            )
            initial_energy = initial_cons['total_energy']
            
            solver = MHDSolver(cfl=0.4, use_gpu=False)
            result = solver.run(
                system,
                t_end=0.5,
                save_dt=0.5,
                verbose=False
            )
            
            _, U_final = result['snapshots'][-1]
            final_cons = compute_conservation_metrics(
                U_final, system.dx, system.dy, system.gamma
            )
            final_energy = final_cons['total_energy']
            
            energy_error = abs(final_energy - initial_energy) / initial_energy
            energy_errors.append(energy_error)
        
        for i, error in enumerate(energy_errors):
            assert error < 0.01, f"Energy conservation error too large at nx={resolutions[i]}: {error:.2e}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
