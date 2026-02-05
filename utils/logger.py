"""
Logger Module
=============
Logging utilities for the multimodal misinformation detection system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class Logger:
    """
    Logger wrapper for consistent logging across the project.
    Logs to both console and file.
    """
    
    def __init__(
        self,
        name: str = "MMD",
        log_dir: str = "logs",
        log_level: int = logging.INFO,
    ):
        """
        Initialize Logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            log_level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"log_{timestamp}.txt"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def section(self, title: str):
        """Log a section header."""
        separator = "=" * 80
        self.info(f"\n{separator}")
        self.info(f"{title.upper()}")
        self.info(f"{separator}\n")
    
    def subsection(self, title: str):
        """Log a subsection header."""
        separator = "-" * 80
        self.info(f"\n{separator}")
        self.info(f"{title}")
        self.info(f"{separator}\n")


class ExperimentLogger:
    """
    Logs experiment details and results.
    """
    
    def __init__(self, experiment_name: str, log_dir: str = "experiments"):
        """
        Initialize ExperimentLogger.
        
        Args:
            experiment_name: Name of experiment
            log_dir: Directory for experiment logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = self.log_dir / f"{experiment_name}_{self.timestamp}"
        self.exp_dir.mkdir(exist_ok=True)
        
        self.logger = Logger(
            name=f"Exp_{experiment_name}",
            log_dir=str(self.exp_dir),
        )
        
        self.results = {}
    
    def log_config(self, config_dict: dict):
        """
        Log configuration.
        
        Args:
            config_dict: Configuration dictionary
        """
        import json
        config_path = self.exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        self.logger.info(f"Configuration saved to {config_path}")
    
    def log_result(self, key: str, value):
        """
        Log result.
        
        Args:
            key: Result key
            value: Result value
        """
        self.results[key] = value
        self.logger.info(f"{key}: {value}")
    
    def save_results(self):
        """Save all results to JSON file."""
        import json
        results_path = self.exp_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.logger.info(f"Results saved to {results_path}")
    
    def get_experiment_dir(self) -> str:
        """Get experiment directory path."""
        return str(self.exp_dir)


def setup_logging(
    experiment_name: str = "default",
    log_level: int = logging.INFO,
) -> tuple:
    """
    Setup logging for the entire project.
    
    Args:
        experiment_name: Name of experiment
        log_level: Logging level
        
    Returns:
        Tuple of (logger, experiment_logger)
    """
    logger = Logger(log_level=log_level)
    exp_logger = ExperimentLogger(experiment_name)
    
    logger.section("LOGGING INITIALIZED")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Log directory: {exp_logger.exp_dir}")
    
    return logger, exp_logger
