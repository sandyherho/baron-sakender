"""Data handler for saving MHD simulation results to CSV and NetCDF."""

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple


class DataHandler:
    """Handle saving MHD simulation data to various formats."""
    
    @staticmethod
    def save_metrics_csv(filepath: str, metrics_history: List[Dict[str, Any]], times: np.ndarray):
        """
        Save time series of metrics to CSV.
        
        Args:
            filepath: Output file path
            metrics_history: List of metrics dictionaries at each time
            times: Time array
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Build DataFrame from metrics history
        rows = []
        for i, (t, metrics) in enumerate(zip(times, metrics_history)):
            row = {'time': t}
            for key, value in metrics.items():
                if isinstance(value, (int, float, bool)):
                    row[key] = value
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, float_format='%.8e')
    
    @staticmethod
    def save_final_metrics_csv(filepath: str, metrics: Dict[str, Any]):
        """
        Save final state metrics to CSV.
        
        Args:
            filepath: Output file path
            metrics: Metrics dictionary
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for key, value in sorted(metrics.items()):
            if isinstance(value, (int, float, bool, str)):
                rows.append({
                    'Metric': key,
                    'Value': value,
                    'Type': type(value).__name__
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
    
    @staticmethod
    def save_netcdf(
        filepath: str,
        result: Dict[str, Any],
        config: Dict[str, Any],
        all_metrics: Optional[List[Dict[str, Any]]] = None,
        final_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Save complete simulation data to CF-compliant NetCDF format.
        
        Creates a NetCDF file with:
            - Snapshots of all fields at saved times
            - Time series of conservation/stability metrics
            - Full diagnostic metrics
            - Physical units and metadata
        
        Args:
            filepath: Output file path
            result: Simulation result dictionary
            config: Configuration dictionary
            all_metrics: List of metrics at each snapshot
            final_metrics: Metrics for final state
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        snapshots = result['snapshots']
        params = result['params']
        times = np.array([s[0] for s in snapshots])
        n_time = len(times)
        
        # Get dimensions from first snapshot
        U0 = snapshots[0][1]
        n_var, nx, ny = U0.shape
        
        with Dataset(filepath, 'w', format='NETCDF4') as nc:
            # ============ Dimensions ============
            nc.createDimension('time', n_time)
            nc.createDimension('x', nx)
            nc.createDimension('y', ny)
            nc.createDimension('variable', n_var)
            
            # ============ Coordinate Variables ============
            # Time
            nc_time = nc.createVariable('time', 'f8', ('time',), zlib=True)
            nc_time[:] = times
            nc_time.units = f"tau_A (Alfven_time = {params.tau_A:.6e} s)"
            nc_time.long_name = "simulation_time"
            nc_time.standard_name = "time"
            nc_time.axis = "T"
            
            # X coordinate
            x_vals = np.linspace(0, params.Lx, nx, endpoint=False)
            nc_x = nc.createVariable('x', 'f8', ('x',), zlib=True)
            nc_x[:] = x_vals
            nc_x.units = f"L_0 (reference_length = {params.L_0:.6e} m)"
            nc_x.long_name = "x_coordinate"
            nc_x.standard_name = "projection_x_coordinate"
            nc_x.axis = "X"
            
            # Y coordinate
            y_vals = np.linspace(0, params.Ly, ny, endpoint=False)
            nc_y = nc.createVariable('y', 'f8', ('y',), zlib=True)
            nc_y[:] = y_vals
            nc_y.units = f"L_0 (reference_length = {params.L_0:.6e} m)"
            nc_y.long_name = "y_coordinate"
            nc_y.standard_name = "projection_y_coordinate"
            nc_y.axis = "Y"
            
            # ============ Field Variables ============
            # Stack all snapshots
            U_all = np.zeros((n_time, n_var, nx, ny), dtype=np.float64)
            for i, (t, U) in enumerate(snapshots):
                U_all[i] = U
            
            # Full conservative variables array
            nc_U = nc.createVariable('U', 'f4', ('time', 'variable', 'x', 'y'), zlib=True)
            nc_U[:] = U_all.astype(np.float32)
            nc_U.long_name = "conservative_variables"
            nc_U.description = "rho, rho*vx, rho*vy, Bx, By, E"
            
            # Individual fields for convenience
            # Density
            nc_rho = nc.createVariable('density', 'f4', ('time', 'x', 'y'), zlib=True)
            nc_rho[:] = U_all[:, 0].astype(np.float32)
            nc_rho.units = f"rho_0 (reference_density = {params.rho_0:.6e} kg/m^3)"
            nc_rho.long_name = "mass_density"
            nc_rho.standard_name = "mass_density"
            
            # Velocity components
            rho = U_all[:, 0]
            vx = U_all[:, 1] / rho
            vy = U_all[:, 2] / rho
            
            nc_vx = nc.createVariable('velocity_x', 'f4', ('time', 'x', 'y'), zlib=True)
            nc_vx[:] = vx.astype(np.float32)
            nc_vx.units = f"v_A (Alfven_velocity = {params.v_A:.6e} m/s)"
            nc_vx.long_name = "x_velocity"
            nc_vx.standard_name = "eastward_velocity"
            
            nc_vy = nc.createVariable('velocity_y', 'f4', ('time', 'x', 'y'), zlib=True)
            nc_vy[:] = vy.astype(np.float32)
            nc_vy.units = f"v_A (Alfven_velocity = {params.v_A:.6e} m/s)"
            nc_vy.long_name = "y_velocity"
            nc_vy.standard_name = "northward_velocity"
            
            # Magnetic field
            nc_Bx = nc.createVariable('B_x', 'f4', ('time', 'x', 'y'), zlib=True)
            nc_Bx[:] = U_all[:, 3].astype(np.float32)
            nc_Bx.units = f"B_0 (reference_field = {params.B_0:.6e} T)"
            nc_Bx.long_name = "x_magnetic_field"
            
            nc_By = nc.createVariable('B_y', 'f4', ('time', 'x', 'y'), zlib=True)
            nc_By[:] = U_all[:, 4].astype(np.float32)
            nc_By.units = f"B_0 (reference_field = {params.B_0:.6e} T)"
            nc_By.long_name = "y_magnetic_field"
            
            # Total energy
            nc_E = nc.createVariable('total_energy', 'f4', ('time', 'x', 'y'), zlib=True)
            nc_E[:] = U_all[:, 5].astype(np.float32)
            nc_E.units = f"P_0 (magnetic_pressure = {params.P_0:.6e} Pa)"
            nc_E.long_name = "total_energy_density"
            
            # Derived fields (compute pressure and magnetic pressure)
            gamma = params.gamma
            kinetic = 0.5 * rho * (vx**2 + vy**2)
            magnetic_pressure = 0.5 * (U_all[:, 3]**2 + U_all[:, 4]**2)
            pressure = (gamma - 1) * (U_all[:, 5] - kinetic - magnetic_pressure)
            pressure = np.maximum(pressure, 1e-10)
            
            nc_p = nc.createVariable('pressure', 'f4', ('time', 'x', 'y'), zlib=True)
            nc_p[:] = pressure.astype(np.float32)
            nc_p.units = f"P_0 (magnetic_pressure = {params.P_0:.6e} Pa)"
            nc_p.long_name = "thermal_pressure"
            
            nc_pmag = nc.createVariable('magnetic_pressure', 'f4', ('time', 'x', 'y'), zlib=True)
            nc_pmag[:] = magnetic_pressure.astype(np.float32)
            nc_pmag.units = f"P_0 (magnetic_pressure = {params.P_0:.6e} Pa)"
            nc_pmag.long_name = "magnetic_pressure"
            
            # ============ Metrics Variables ============
            if all_metrics is not None and len(all_metrics) > 0:
                # Conservation metrics time series
                metrics_keys = [k for k in all_metrics[0].keys() 
                               if isinstance(all_metrics[0][k], (int, float))]
                
                for key in metrics_keys:
                    values = np.array([m.get(key, np.nan) for m in all_metrics])
                    nc_var = nc.createVariable(f'metric_{key}', 'f8', ('time',), zlib=True)
                    nc_var[:] = values
                    nc_var.long_name = key.replace('_', ' ')
            
            # ============ Final Metrics ============
            if final_metrics is not None:
                for key, value in final_metrics.items():
                    if isinstance(value, (int, float)):
                        nc.setncattr(f'final_{key}', float(value))
                    elif isinstance(value, bool):
                        nc.setncattr(f'final_{key}', int(value))
            
            # ============ Global Attributes ============
            # Physical parameters
            nc.gamma = float(params.gamma)
            nc.nx = int(params.nx)
            nc.ny = int(params.ny)
            nc.Lx = float(params.Lx)
            nc.Ly = float(params.Ly)
            
            # Reference units (SI)
            nc.rho_0_SI = float(params.rho_0)
            nc.rho_0_units = "kg/m^3"
            nc.B_0_SI = float(params.B_0)
            nc.B_0_units = "T"
            nc.L_0_SI = float(params.L_0)
            nc.L_0_units = "m"
            nc.v_A_SI = float(params.v_A)
            nc.v_A_units = "m/s"
            nc.tau_A_SI = float(params.tau_A)
            nc.tau_A_units = "s"
            nc.P_0_SI = float(params.P_0)
            nc.P_0_units = "Pa"
            
            # Simulation parameters
            nc.t_end = float(result.get('t_end', times[-1]))
            nc.cfl = float(config.get('cfl', 0.4))
            nc.total_steps = int(result.get('total_steps', 0))
            nc.n_snapshots = int(n_time)
            
            # Configuration
            nc.scenario_name = config.get('scenario_name', 'MHD Simulation')
            nc.init_type = config.get('init_type', 'unknown')
            nc.backend = result.get('backend', 'cpu')
            
            # CF Metadata
            nc.title = "2D Ideal MHD Simulation - baron-sakender"
            nc.institution = "baron-sakender v0.0.1"
            nc.source = "JAX-accelerated finite volume solver"
            nc.history = f"Created {datetime.now().isoformat()}"
            nc.Conventions = "CF-1.8"
            
            # Author info
            nc.author = "Sandy H. S. Herho, Nurjanna J. Trilaksono"
            nc.contact = "sandy.herho@email.ucr.edu"
            nc.license = "MIT"
    
    @staticmethod
    def load_netcdf(filepath: str) -> Dict[str, Any]:
        """
        Load simulation data from NetCDF file.
        
        Args:
            filepath: Path to NetCDF file
        
        Returns:
            Dictionary with simulation data
        """
        with Dataset(filepath, 'r') as nc:
            result = {
                'times': np.array(nc.variables['time'][:]),
                'x': np.array(nc.variables['x'][:]),
                'y': np.array(nc.variables['y'][:]),
                'U': np.array(nc.variables['U'][:]),
                'density': np.array(nc.variables['density'][:]),
                'velocity_x': np.array(nc.variables['velocity_x'][:]),
                'velocity_y': np.array(nc.variables['velocity_y'][:]),
                'B_x': np.array(nc.variables['B_x'][:]),
                'B_y': np.array(nc.variables['B_y'][:]),
                'total_energy': np.array(nc.variables['total_energy'][:]),
                'gamma': nc.gamma,
                'nx': nc.nx,
                'ny': nc.ny,
            }
            
            # Load attributes
            for attr in nc.ncattrs():
                result[attr] = nc.getncattr(attr)
        
        return result
