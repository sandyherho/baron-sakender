"""Configuration file parser for MHD simulations."""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


class ConfigManager:
    """Parse and manage configuration files for MHD simulations."""
    
    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        File format:
            # Comments
            key = value
        
        Args:
            config_path: Path to configuration file
        
        Returns:
            Dictionary of configuration parameters
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        config = {}
        
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty/comment lines
                if not line or line.startswith('#'):
                    continue
                
                if '=' not in line:
                    continue
                
                # Parse key = value
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove inline comments
                if '#' in value:
                    value = value.split('#')[0].strip()
                
                # Parse type
                config[key] = ConfigManager._parse_value(value)
        
        return config
    
    @staticmethod
    def _parse_value(value: str) -> Any:
        """Parse string to appropriate Python type."""
        # Boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Numeric
        try:
            if '.' in value or 'e' in value.lower():
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value
    
    @staticmethod
    def save(config: Dict[str, Any], config_path: str):
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            config_path: Output path
        """
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            f.write("# Baron-Sakender MHD Configuration\n")
            f.write("# Generated automatically\n\n")
            
            for key, value in sorted(config.items()):
                if isinstance(value, bool):
                    value_str = 'true' if value else 'false'
                elif isinstance(value, float):
                    if abs(value) < 1e-4 or abs(value) > 1e4:
                        value_str = f"{value:.6e}"
                    else:
                        value_str = f"{value}"
                else:
                    value_str = str(value)
                
                f.write(f"{key} = {value_str}\n")
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'scenario_name': 'Standard Orszag-Tang',
            'nx': 256,
            'ny': 256,
            'gamma': 5.0 / 3.0,
            'Lx': 2 * np.pi,
            'Ly': 2 * np.pi,
            't_end': 3.0,
            'save_dt': 0.05,
            'cfl': 0.4,
            'init_type': 'orszag_tang',
            'amplitude': 1.0,
            'use_gpu': False,
            'compute_metrics': True,
            'save_csv': True,
            'save_netcdf': True,
            'save_png': True,
            'save_gif': True,
            'output_dir': 'outputs',
            'animation_fps': 20,
            'animation_dpi': 100,
            'png_dpi': 150,
            # Physical units (SI)
            'rho_0': 1.0e-12,  # kg/m³
            'B_0': 1.0e-9,     # T
            'L_0': 1.0e6,      # m
        }
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            True if valid
        
        Raises:
            ValueError: If configuration is invalid
        """
        required = ['nx', 'ny', 'gamma', 't_end', 'cfl']
        
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required parameter: {key}")
        
        if config.get('nx', 0) < 16:
            raise ValueError("nx must be >= 16")
        
        if config.get('ny', 0) < 16:
            raise ValueError("ny must be >= 16")
        
        if config.get('cfl', 0) <= 0 or config.get('cfl', 0) > 1:
            raise ValueError("cfl must be in (0, 1]")
        
        if config.get('t_end', 0) <= 0:
            raise ValueError("t_end must be > 0")
        
        return True
