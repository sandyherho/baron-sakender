"""Comprehensive simulation logger for MHD simulations."""

import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


class SimulationLogger:
    """Logger for MHD simulations with detailed diagnostics."""
    
    def __init__(
        self,
        scenario_name: str,
        log_dir: str = "logs",
        verbose: bool = True
    ):
        """
        Initialize simulation logger.
        
        Args:
            scenario_name: Scenario name (for log filename)
            log_dir: Directory for log files
            verbose: Print messages to console
        """
        self.scenario_name = scenario_name
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{scenario_name}.log"
        
        self.logger = self._setup_logger()
        self.warnings: List[str] = []
        self.errors: List[str] = []
    
    def _setup_logger(self) -> logging.Logger:
        """Configure Python logging."""
        logger = logging.getLogger(f"baron_sakender_{self.scenario_name}")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        
        handler = logging.FileHandler(self.log_file, mode='w')
        handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        return logger
    
    def info(self, msg: str):
        """Log informational message."""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)
        self.warnings.append(msg)
        
        if self.verbose:
            print(f"  WARNING: {msg}")
    
    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)
        self.errors.append(msg)
        
        if self.verbose:
            print(f"  ERROR: {msg}")
    
    def log_parameters(self, config: Dict[str, Any]):
        """Log all simulation parameters."""
        self.info("=" * 70)
        self.info("2D IDEAL MHD SIMULATION - BARON-SAKENDER")
        self.info(f"Scenario: {config.get('scenario_name', 'Unknown')}")
        self.info("=" * 70)
        self.info("")
        
        self.info("GRID PARAMETERS:")
        self.info(f"  nx = {config.get('nx', 256)}")
        self.info(f"  ny = {config.get('ny', 256)}")
        self.info(f"  Lx = {config.get('Lx', 2*np.pi):.4f}")
        self.info(f"  Ly = {config.get('Ly', 2*np.pi):.4f}")
        
        self.info("")
        self.info("PHYSICAL PARAMETERS:")
        self.info(f"  gamma = {config.get('gamma', 5/3):.4f}")
        self.info(f"  rho_0 = {config.get('rho_0', 1e-12):.2e} kg/m^3")
        self.info(f"  B_0 = {config.get('B_0', 1e-9):.2e} T")
        self.info(f"  L_0 = {config.get('L_0', 1e6):.2e} m")
        
        self.info("")
        self.info("SIMULATION PARAMETERS:")
        self.info(f"  t_end = {config.get('t_end', 3.0)}")
        self.info(f"  save_dt = {config.get('save_dt', 0.05)}")
        self.info(f"  CFL = {config.get('cfl', 0.4)}")
        self.info(f"  use_gpu = {config.get('use_gpu', False)}")
        self.info(f"  init_type = {config.get('init_type', 'orszag_tang')}")
        
        self.info("=" * 70)
        self.info("")
    
    def log_conservation(self, metrics: Dict[str, float], t: float):
        """Log conservation metrics at a time."""
        self.info(f"Conservation at t={t:.4f}:")
        self.info(f"  Mass: {metrics.get('total_mass', 0):.8e}")
        self.info(f"  Energy: {metrics.get('total_energy', 0):.8e}")
        self.info(f"  Momentum: {metrics.get('total_momentum', 0):.8e}")
        self.info(f"  Cross helicity: {metrics.get('cross_helicity', 0):.8e}")
        self.info(f"  Max |∇·B|: {metrics.get('max_div_B', 0):.8e}")
    
    def log_stability(self, metrics: Dict[str, float], t: float):
        """Log stability metrics."""
        self.info(f"Stability at t={t:.4f}:")
        self.info(f"  Max Mach: {metrics.get('max_sonic_mach', 0):.4f}")
        self.info(f"  Max Alfven Mach: {metrics.get('max_alfven_mach', 0):.4f}")
        self.info(f"  Min beta: {metrics.get('min_beta', 0):.4f}")
        self.info(f"  CFL effective: {metrics.get('cfl_effective', 0):.4f}")
        self.info(f"  Is stable: {metrics.get('is_stable', False)}")
    
    def log_turbulence(self, metrics: Dict[str, float], t: float):
        """Log turbulence metrics."""
        self.info(f"Turbulence at t={t:.4f}:")
        self.info(f"  Kinetic energy: {metrics.get('kinetic_energy', 0):.6e}")
        self.info(f"  Magnetic energy: {metrics.get('magnetic_energy', 0):.6e}")
        self.info(f"  Cross helicity: {metrics.get('cross_helicity', 0):.6e}")
        self.info(f"  Spectral index (kinetic): {metrics.get('spectral_index_kinetic', np.nan):.2f}")
    
    def log_timing(self, timing: Dict[str, float]):
        """Log timing breakdown."""
        self.info("=" * 70)
        self.info("TIMING BREAKDOWN:")
        self.info("=" * 70)
        
        for key, value in sorted(timing.items()):
            if key != 'total':
                self.info(f"  {key}: {value:.3f} s")
        
        self.info(f"  {'-' * 40}")
        total_time = timing.get('total', sum(timing.values()))
        self.info(f"  TOTAL: {total_time:.3f} s")
        
        self.info("=" * 70)
        self.info("")
    
    def log_final_metrics(self, metrics: Dict[str, Any]):
        """Log final simulation metrics."""
        self.info("=" * 70)
        self.info("FINAL METRICS:")
        self.info("=" * 70)
        
        # Conservation
        self.info("\nCONSERVATION:")
        for key in sorted(metrics.keys()):
            if key.startswith('cons_'):
                value = metrics[key]
                if isinstance(value, (int, float)):
                    self.info(f"  {key[5:]}: {value:.8e}")
        
        # Stability
        self.info("\nSTABILITY:")
        for key in sorted(metrics.keys()):
            if key.startswith('stab_'):
                value = metrics[key]
                if isinstance(value, (int, float)):
                    self.info(f"  {key[5:]}: {value:.6f}")
        
        # Turbulence
        self.info("\nTURBULENCE:")
        for key in sorted(metrics.keys()):
            if key.startswith('turb_'):
                value = metrics[key]
                if isinstance(value, (int, float)):
                    self.info(f"  {key[5:]}: {value:.6e}")
        
        # Information
        self.info("\nINFORMATION METRICS:")
        for key in sorted(metrics.keys()):
            if key.startswith('info_'):
                value = metrics[key]
                if isinstance(value, (int, float)):
                    self.info(f"  {key[5:]}: {value:.6f}")
        
        # Composite
        self.info("\nCOMPOSITE METRICS:")
        for key in sorted(metrics.keys()):
            if key.startswith('comp_'):
                value = metrics[key]
                if isinstance(value, (int, float)):
                    self.info(f"  {key[5:]}: {value:.6f}")
        
        self.info("=" * 70)
    
    def finalize(self):
        """Write final summary."""
        self.info("=" * 70)
        self.info("SIMULATION SUMMARY:")
        self.info("=" * 70)
        self.info("")
        
        if self.errors:
            self.info(f"ERRORS: {len(self.errors)}")
            for i, err in enumerate(self.errors, 1):
                self.info(f"  {i}. {err}")
        else:
            self.info("ERRORS: None")
        
        self.info("")
        
        if self.warnings:
            self.info(f"WARNINGS: {len(self.warnings)}")
            for i, warn in enumerate(self.warnings, 1):
                self.info(f"  {i}. {warn}")
        else:
            self.info("WARNINGS: None")
        
        self.info("")
        self.info(f"Log file: {self.log_file}")
        self.info("=" * 70)
        self.info(f"Simulation completed: {self.scenario_name}")
        self.info(f"Timestamp: {datetime.now().isoformat()}")
        self.info("=" * 70)
