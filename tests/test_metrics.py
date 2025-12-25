"""
Tests for simplified MHD metrics module.


Tests verify:
    - Conservation metrics computation
    - Stability metrics computation
    - Turbulence metrics computation 
    - Information metrics 
    - Composite metrics
    - Physical correctness of derived quantities
"""

import numpy as np
import pytest
from baron_sakender.core.metrics import (
    compute_diagnostics,
    compute_conservation_metrics,
    compute_stability_metrics,
    compute_turbulence_metrics,
    compute_information_metrics,
    compute_composite_metrics,
    compute_all_metrics,
)


class TestComputeDiagnostics:
    """Test diagnostic field computations."""
    
    def test_diagnostics_shape(self):
        """Test that diagnostics return correct shapes."""
        nx, ny = 64, 64
        U = np.random.rand(7, nx, ny) + 0.1
        dx = dy = 0.1
        gamma = 5/3
        
        rho, pmag, Jz, omega_z = compute_diagnostics(U, dx, dy, gamma)
        
        assert rho.shape == (nx, ny)
        assert pmag.shape == (nx, ny)
        assert Jz.shape == (nx, ny)
        assert omega_z.shape == (nx, ny)
    
    def test_uniform_field_zero_curl(self):
        """Test that uniform fields give zero current and vorticity."""
        nx, ny = 32, 32
        dx = dy = 0.1
        gamma = 5/3
        
        # Create uniform state
        U = np.ones((7, nx, ny))
        U[0] = 1.0   # rho
        U[1] = 0.0   # rho*vx
        U[2] = 0.0   # rho*vy
        U[3] = 1.0   # Bx (uniform)
        U[4] = 0.0   # By (uniform)
        U[5] = 1.0   # E
        U[6] = 0.0   # psi
        
        rho, pmag, Jz, omega_z = compute_diagnostics(U, dx, dy, gamma)
        
        # Uniform B should give zero current
        assert np.allclose(Jz, 0, atol=1e-10)
        # Zero velocity should give zero vorticity
        assert np.allclose(omega_z, 0, atol=1e-10)


class TestConservationMetrics:
    """Test conservation metrics computation."""
    
    def test_conservation_metrics_keys(self):
        """Test that all expected keys are present."""
        nx, ny = 32, 32
        U = np.random.rand(7, nx, ny) + 0.1
        dx = dy = 0.1
        gamma = 5/3
        
        metrics = compute_conservation_metrics(U, dx, dy, gamma)
        
        expected_keys = [
            'total_mass', 'total_momentum_x', 'total_momentum_y', 'total_momentum',
            'total_energy', 'kinetic_energy', 'magnetic_energy', 'thermal_energy',
            'cross_helicity', 'max_div_B', 'mean_div_B'
        ]
        
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"
    
    def test_positive_energies(self):
        """Test that energies are positive."""
        nx, ny = 32, 32
        dx = dy = 0.1
        gamma = 5/3
        
        # Create valid physical state
        U = np.ones((7, nx, ny))
        U[0] = 1.0    # rho
        U[1] = 0.1    # rho*vx
        U[2] = 0.1    # rho*vy
        U[3] = 0.5    # Bx
        U[4] = 0.5    # By
        U[5] = 2.0    # E (must be > kinetic + magnetic)
        U[6] = 0.0    # psi
        
        metrics = compute_conservation_metrics(U, dx, dy, gamma)
        
        assert metrics['kinetic_energy'] >= 0
        assert metrics['magnetic_energy'] >= 0
        assert metrics['thermal_energy'] >= 0
        assert metrics['total_mass'] > 0
    
    def test_mass_computation(self):
        """Test mass computation accuracy."""
        nx, ny = 32, 32
        dx = dy = 0.1
        gamma = 5/3
        
        rho_0 = 2.0
        U = np.ones((7, nx, ny))
        U[0] = rho_0
        U[5] = 1.0  # E
        
        metrics = compute_conservation_metrics(U, dx, dy, gamma)
        
        expected_mass = rho_0 * (nx * dx) * (ny * dy)
        assert np.isclose(metrics['total_mass'], expected_mass, rtol=1e-10)


class TestStabilityMetrics:
    """Test stability metrics computation."""
    
    def test_stability_metrics_keys(self):
        """Test that all expected keys are present."""
        nx, ny = 32, 32
        U = np.ones((7, nx, ny))
        U[0] = 1.0
        U[5] = 2.0
        dx = dy = 0.1
        gamma = 5/3
        dt = 0.001
        cfl = 0.4
        
        metrics = compute_stability_metrics(U, dx, dy, gamma, dt, cfl)
        
        expected_keys = [
            'max_sonic_mach', 'mean_sonic_mach', 'max_alfven_mach', 'mean_alfven_mach',
            'max_fast_mach', 'max_velocity', 'max_sound_speed', 'max_alfven_speed',
            'max_fast_speed', 'min_beta', 'max_beta', 'mean_beta',
            'cfl_x', 'cfl_y', 'cfl_effective', 'target_cfl', 'dt',
            'min_density', 'min_pressure', 'is_stable'
        ]
        
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"
    
    def test_stability_positive_speeds(self):
        """Test that characteristic speeds are positive."""
        nx, ny = 32, 32
        U = np.ones((7, nx, ny))
        U[0] = 1.0    # rho
        U[3] = 0.5    # Bx
        U[4] = 0.5    # By
        U[5] = 2.0    # E
        dx = dy = 0.1
        gamma = 5/3
        dt = 0.001
        cfl = 0.4
        
        metrics = compute_stability_metrics(U, dx, dy, gamma, dt, cfl)
        
        assert metrics['max_sound_speed'] > 0
        assert metrics['max_alfven_speed'] > 0
        assert metrics['max_fast_speed'] > 0
    
    def test_stable_state_detection(self):
        """Test detection of stable states."""
        nx, ny = 32, 32
        U = np.ones((7, nx, ny))
        U[0] = 1.0    # positive density
        U[5] = 2.0    # sufficient energy for positive pressure
        dx = dy = 0.1
        gamma = 5/3
        dt = 0.001
        cfl = 0.4
        
        metrics = compute_stability_metrics(U, dx, dy, gamma, dt, cfl)
        
        assert metrics['is_stable'] == 1.0
        assert metrics['min_density'] > 0
        assert metrics['min_pressure'] > 0


class TestTurbulenceMetrics:
    """Test turbulence metrics computation (spectral indices removed)."""
    
    def test_turbulence_metrics_keys(self):
        """Test that all expected keys are present."""
        nx, ny = 64, 64
        U = np.random.rand(7, nx, ny) + 0.1
        U[5] = 2.0  # Ensure positive pressure
        dx = dy = 0.1
        gamma = 5/3
        
        metrics = compute_turbulence_metrics(U, dx, dy, gamma)
        
        expected_keys = [
            'kinetic_energy', 'magnetic_energy', 'total_energy', 'energy_ratio',
            'enstrophy', 'mean_vorticity', 'max_vorticity', 'rms_vorticity',
            'current_squared', 'mean_current', 'max_current', 'rms_current',
            'cross_helicity', 'normalized_cross_helicity',
            'residual_energy', 'normalized_residual_energy',
            'elsasser_plus_energy', 'elsasser_minus_energy', 'energy_imbalance',
            'vorticity_kurtosis', 'current_kurtosis', 'taylor_microscale'
        ]
        
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"
    
    def test_no_spectral_indices(self):
        """Verify spectral indices are NOT present (removed due to HLL diffusion)."""
        nx, ny = 64, 64
        U = np.random.rand(7, nx, ny) + 0.1
        U[5] = 2.0
        dx = dy = 0.1
        gamma = 5/3
        
        metrics = compute_turbulence_metrics(U, dx, dy, gamma)
        
        # These should NOT be present
        assert 'spectral_index_kinetic' not in metrics
        assert 'spectral_index_magnetic' not in metrics
        assert 'k_spectrum' not in metrics
        assert 'E_kinetic_spectrum' not in metrics
        assert 'E_magnetic_spectrum' not in metrics
    
    def test_energy_consistency(self):
        """Test that total energy equals sum of kinetic and magnetic."""
        nx, ny = 32, 32
        U = np.random.rand(7, nx, ny) + 0.1
        U[5] = 2.0
        dx = dy = 0.1
        gamma = 5/3
        
        metrics = compute_turbulence_metrics(U, dx, dy, gamma)
        
        total = metrics['kinetic_energy'] + metrics['magnetic_energy']
        assert np.isclose(metrics['total_energy'], total, rtol=1e-10)
    
    def test_elsasser_energy_consistency(self):
        """Test Elsässer energy properties."""
        nx, ny = 32, 32
        U = np.random.rand(7, nx, ny) + 0.1
        U[5] = 2.0
        dx = dy = 0.1
        gamma = 5/3
        
        metrics = compute_turbulence_metrics(U, dx, dy, gamma)
        
        # Energy imbalance should be bounded between -1 and 1
        assert -1 <= metrics['energy_imbalance'] <= 1
        
        # Elsässer energies should be positive
        assert metrics['elsasser_plus_energy'] >= 0
        assert metrics['elsasser_minus_energy'] >= 0
    
    def test_kurtosis_values(self):
        """Test that kurtosis values are reasonable."""
        nx, ny = 64, 64
        U = np.random.rand(7, nx, ny) + 0.1
        U[5] = 2.0
        dx = dy = 0.1
        gamma = 5/3
        
        metrics = compute_turbulence_metrics(U, dx, dy, gamma)
        
        # Kurtosis should be positive (fourth moment / variance^2)
        assert metrics['vorticity_kurtosis'] > 0
        assert metrics['current_kurtosis'] > 0


class TestInformationMetrics:
    """Test information metrics (simplified - Shannon entropy only)."""
    
    def test_information_metrics_keys(self):
        """Test that expected keys are present (entropy only)."""
        nx, ny = 64, 64
        U = np.random.rand(7, nx, ny) + 0.1
        U[5] = 2.0
        dx = dy = 0.1
        gamma = 5/3
        
        metrics = compute_information_metrics(U, dx, dy, gamma)
        
        expected_keys = [
            'entropy_density', 'entropy_velocity', 'entropy_magnetic',
            'entropy_vorticity', 'entropy_current'
        ]
        
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"
    
    def test_no_complexity_metrics(self):
        """Verify complexity/MI metrics are NOT present (removed)."""
        nx, ny = 64, 64
        U = np.random.rand(7, nx, ny) + 0.1
        U[5] = 2.0
        dx = dy = 0.1
        gamma = 5/3
        
        metrics = compute_information_metrics(U, dx, dy, gamma)
        
        # These should NOT be present
        assert 'MI_density_velocity' not in metrics
        assert 'MI_velocity_magnetic' not in metrics
        assert 'MI_vorticity_current' not in metrics
        assert 'complexity_vorticity' not in metrics
        assert 'statistical_complexity' not in metrics
    
    def test_entropy_positive(self):
        """Test that Shannon entropy is non-negative."""
        nx, ny = 32, 32
        U = np.random.rand(7, nx, ny) + 0.1
        U[5] = 2.0
        dx = dy = 0.1
        gamma = 5/3
        
        metrics = compute_information_metrics(U, dx, dy, gamma)
        
        assert metrics['entropy_density'] >= 0
        assert metrics['entropy_velocity'] >= 0
        assert metrics['entropy_magnetic'] >= 0
        assert metrics['entropy_vorticity'] >= 0
        assert metrics['entropy_current'] >= 0
    
    def test_uniform_field_low_entropy(self):
        """Test that uniform fields have lower entropy than random."""
        nx, ny = 64, 64
        dx = dy = 0.1
        gamma = 5/3
        
        # Uniform state
        U_uniform = np.ones((7, nx, ny))
        U_uniform[0] = 1.0
        U_uniform[5] = 2.0
        
        # Random state
        U_random = np.random.rand(7, nx, ny) + 0.1
        U_random[5] = 2.0
        
        metrics_uniform = compute_information_metrics(U_uniform, dx, dy, gamma)
        metrics_random = compute_information_metrics(U_random, dx, dy, gamma)
        
        # Uniform density should have zero or very low entropy
        # Random density should have higher entropy
        assert metrics_uniform['entropy_density'] < metrics_random['entropy_density']


class TestCompositeMetrics:
    """Test composite metrics (simplified - no cascade efficiency)."""
    
    def test_composite_metrics_keys(self):
        """Test that expected keys are present."""
        nx, ny = 32, 32
        U = np.ones((7, nx, ny))
        U[0] = 1.0
        U[5] = 2.0
        dx = dy = 0.1
        gamma = 5/3
        
        metrics = compute_composite_metrics(U, dx, dy, gamma)
        
        expected_keys = [
            'dynamo_efficiency', 'alignment_index', 'coherent_structure_index',
            'intermittency_index', 'conservation_quality',
            'mass_conservation_error', 'energy_conservation_error'
        ]
        
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"
    
    def test_no_cascade_efficiency(self):
        """Verify cascade_efficiency is NOT present (depends on spectral index)."""
        nx, ny = 32, 32
        U = np.ones((7, nx, ny))
        U[0] = 1.0
        U[5] = 2.0
        dx = dy = 0.1
        gamma = 5/3
        
        metrics = compute_composite_metrics(U, dx, dy, gamma)
        
        # These should NOT be present (depend on spectral indices)
        assert 'cascade_efficiency' not in metrics
        assert 'mhd_complexity_index' not in metrics
    
    def test_dynamo_efficiency_bounded(self):
        """Test that dynamo efficiency is between 0 and 1."""
        nx, ny = 32, 32
        U = np.random.rand(7, nx, ny) + 0.1
        U[5] = 2.0
        dx = dy = 0.1
        gamma = 5/3
        
        metrics = compute_composite_metrics(U, dx, dy, gamma)
        
        assert 0 <= metrics['dynamo_efficiency'] <= 1
    
    def test_conservation_quality_identical_states(self):
        """Test conservation quality for identical states."""
        nx, ny = 32, 32
        U = np.ones((7, nx, ny))
        U[0] = 1.0
        U[5] = 2.0
        dx = dy = 0.1
        gamma = 5/3
        
        # Compute initial conservation
        initial = compute_conservation_metrics(U, dx, dy, gamma)
        
        # Compute composite with initial
        metrics = compute_composite_metrics(U, dx, dy, gamma, conservation_initial=initial)
        
        # For identical states, conservation quality should be 1.0
        assert np.isclose(metrics['conservation_quality'], 1.0, atol=0.01)
        assert np.isclose(metrics['mass_conservation_error'], 0.0, atol=1e-10)
        assert np.isclose(metrics['energy_conservation_error'], 0.0, atol=1e-10)


class TestComputeAllMetrics:
    """Test the master metrics function."""
    
    def test_all_metrics_without_initial(self):
        """Test compute_all_metrics without initial metrics."""
        nx, ny = 32, 32
        U = np.ones((7, nx, ny))
        U[0] = 1.0
        U[5] = 2.0
        dx = dy = 0.1
        gamma = 5/3
        
        metrics = compute_all_metrics(U, dx, dy, gamma)
        
        # Check that key categories are present
        assert 'total_mass' in metrics
        assert 'max_sonic_mach' in metrics
        assert 'kinetic_energy' in metrics
        assert 'entropy_density' in metrics
        assert 'dynamo_efficiency' in metrics
    
    def test_all_metrics_with_initial(self):
        """Test compute_all_metrics with initial metrics."""
        nx, ny = 32, 32
        U = np.ones((7, nx, ny))
        U[0] = 1.0
        U[5] = 2.0
        dx = dy = 0.1
        gamma = 5/3
        
        # First compute initial metrics
        initial = compute_conservation_metrics(U, dx, dy, gamma)
        
        # Then compute all metrics with initial
        metrics = compute_all_metrics(U, dx, dy, gamma, conservation_initial=initial)
        
        # Conservation errors should be zero for identical states
        assert np.isclose(metrics['mass_conservation_error'], 0, atol=1e-10)
        assert np.isclose(metrics['energy_conservation_error'], 0, atol=1e-10)
    
    def test_no_nan_values(self):
        """Test that no NaN values are produced for valid input."""
        nx, ny = 64, 64
        U = np.random.rand(7, nx, ny) + 0.1
        U[5] = 3.0  # Ensure positive pressure
        dx = dy = 0.1
        gamma = 5/3
        
        metrics = compute_all_metrics(U, dx, dy, gamma)
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                assert not np.isnan(value), f"NaN value for {key}"
    
    def test_no_spectral_data(self):
        """Verify no spectral data is included."""
        nx, ny = 32, 32
        U = np.ones((7, nx, ny))
        U[0] = 1.0
        U[5] = 2.0
        dx = dy = 0.1
        gamma = 5/3
        
        metrics = compute_all_metrics(U, dx, dy, gamma)
        
        # These should NOT be present
        assert 'k_spectrum' not in metrics
        assert 'E_kinetic_spectrum' not in metrics
        assert 'E_magnetic_spectrum' not in metrics
        assert 'spectral_index_kinetic' not in metrics


class TestPhysicalCorrectness:
    """Test physical correctness of metrics."""
    
    def test_alfven_wave_properties(self):
        """Test metrics for Alfvén wave-like state."""
        nx, ny = 64, 64
        dx = dy = 2 * np.pi / nx
        gamma = 5/3
        
        x = np.linspace(0, 2*np.pi, nx, endpoint=False)
        y = np.linspace(0, 2*np.pi, ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Create Alfvén wave: δvy = δBy, background Bx = 1
        amplitude = 0.1
        U = np.zeros((7, nx, ny))
        U[0] = 1.0                              # rho
        U[1] = 0.0                              # rho*vx
        U[2] = amplitude * np.sin(X)           # rho*vy
        U[3] = 1.0                              # Bx (background)
        U[4] = amplitude * np.sin(X)           # By (perturbation)
        U[5] = 1.0 + 0.5 * (U[2]**2 + U[3]**2 + U[4]**2)  # E
        U[6] = 0.0                              # psi
        
        metrics = compute_turbulence_metrics(U, dx, dy, gamma)
        
        # For Alfvén wave: |v| ~ |B| perturbation
        # Cross helicity should be non-zero
        assert abs(metrics['cross_helicity']) > 0
    
    def test_high_beta_vs_low_beta(self):
        """Test plasma beta behavior."""
        nx, ny = 32, 32
        dx = dy = 0.1
        gamma = 5/3
        dt = 0.001
        cfl = 0.4
        
        # High beta (pressure dominated)
        U_high = np.ones((7, nx, ny))
        U_high[0] = 1.0
        U_high[3] = 0.01  # weak B
        U_high[4] = 0.01
        U_high[5] = 10.0  # high energy -> high pressure
        
        # Low beta (magnetically dominated)
        U_low = np.ones((7, nx, ny))
        U_low[0] = 1.0
        U_low[3] = 5.0    # strong B
        U_low[4] = 5.0
        U_low[5] = 30.0   # lower thermal pressure relative to magnetic
        
        metrics_high = compute_stability_metrics(U_high, dx, dy, gamma, dt, cfl)
        metrics_low = compute_stability_metrics(U_low, dx, dy, gamma, dt, cfl)
        
        assert metrics_high['mean_beta'] > metrics_low['mean_beta']
    
    def test_equipartition_state(self):
        """Test metrics for equipartition state (E_k ~ E_m)."""
        nx, ny = 32, 32
        dx = dy = 0.1
        gamma = 5/3
        
        # Create state with equal kinetic and magnetic energy
        U = np.ones((7, nx, ny))
        U[0] = 1.0      # rho
        U[1] = 0.5      # rho*vx -> v^2/2 = 0.125
        U[2] = 0.5      # rho*vy
        U[3] = 0.5      # Bx -> B^2/2 = 0.25
        U[4] = 0.5      # By
        U[5] = 2.0      # E
        
        metrics = compute_turbulence_metrics(U, dx, dy, gamma)
        
        # Energy ratio should be close to 1 for equipartition
        # (allowing some tolerance due to numerical factors)
        assert 0.1 < metrics['energy_ratio'] < 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
