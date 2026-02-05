"""
Helper Functions Module
=======================
Utility functions for data handling, device management, and common operations.
"""

import torch
import torch.nn as nn
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from datetime import datetime


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Additional PyTorch settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> str:
    """
    Get appropriate device (GPU if available, else CPU).
    
    Args:
        prefer_cuda: Prefer CUDA if available
        
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if prefer_cuda and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else -1,
    }
    
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    
    return info


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable and total parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def get_model_info(model: nn.Module) -> Dict:
    """
    Get comprehensive model information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model info
    """
    total, trainable = count_parameters(model)
    
    return {
        'total_parameters': total,
        'trainable_parameters': trainable,
        'frozen_parameters': total - trainable,
        'model_size_mb': sum(p.numel() * 4 for p in model.parameters()) / 1e6,  # Assuming float32
    }


def save_json(data: Dict, path: str):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Dict:
    """
    Load dictionary from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)


def create_directories(paths: List[str]):
    """
    Create multiple directories.
    
    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get the latest checkpoint from a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        return None
    
    checkpoints = list(checkpoint_path.glob('checkpoint_*.pt'))
    
    if not checkpoints:
        return None
    
    # Return the most recently modified checkpoint
    return str(max(checkpoints, key=lambda p: p.stat().st_mtime))


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_memory_usage(device: str = 'cuda') -> Dict[str, float]:
    """
    Get memory usage on device.
    
    Args:
        device: Device to check ('cuda' or 'cpu')
        
    Returns:
        Dictionary with memory info (in MB)
    """
    if device == 'cuda' and torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1e6,
            'reserved': torch.cuda.memory_reserved() / 1e6,
            'max_allocated': torch.cuda.max_memory_allocated() / 1e6,
        }
    else:
        return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}


def freeze_model(model: nn.Module):
    """
    Freeze all parameters in a model.
    
    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: nn.Module):
    """
    Unfreeze all parameters in a model.
    
    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = True


def freeze_backbone(model: nn.Module, module_name: str):
    """
    Freeze specific module in a model.
    
    Args:
        model: PyTorch model
        module_name: Name of module to freeze
    """
    for name, module in model.named_modules():
        if module_name in name:
            for param in module.parameters():
                param.requires_grad = False


def get_model_device(model: nn.Module) -> str:
    """
    Get device of model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Device string
    """
    return next(model.parameters()).device


class EarlyStopping:
    """
    Early stopping callback to prevent overfitting.
    """
    
    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = 'min',
    ):
        """
        Initialize EarlyStopping.
        
        Args:
            patience: Number of checks with no improvement after which training stops
            min_delta: Minimum change in monitored value to qualify as improvement
            mode: 'min' or 'max' - whether lower or higher metric is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.best_score = float('inf')
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.best_score = float('-inf')
            self.is_better = lambda current, best: current > best + min_delta
    
    def __call__(self, current_score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_score: Current metric score
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class ProgressTracker:
    """
    Track and display training progress.
    """
    
    def __init__(self, total_steps: int):
        """
        Initialize ProgressTracker.
        
        Args:
            total_steps: Total number of steps
        """
        self.total_steps = total_steps
        self.current_step = 0
    
    def update(self, n: int = 1):
        """
        Update progress.
        
        Args:
            n: Number of steps to add
        """
        self.current_step += n
    
    def get_progress(self) -> float:
        """
        Get progress as percentage.
        
        Returns:
            Progress percentage (0-100)
        """
        return 100 * self.current_step / self.total_steps if self.total_steps > 0 else 0
    
    def __str__(self) -> str:
        """String representation with progress bar."""
        progress = self.get_progress()
        bar_length = 30
        filled = int(bar_length * self.current_step / self.total_steps)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        return f"[{bar}] {progress:.1f}% ({self.current_step}/{self.total_steps})"
