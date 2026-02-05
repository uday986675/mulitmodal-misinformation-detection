"""
Configuration Module
====================
Centralized configuration for the multimodal misinformation detection system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Dataset paths
    data_dir: str = "datasets"
    output_dir: str = "outputs"
    
    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Text preprocessing
    text_max_length: int = 128
    text_model: str = "distilbert-base-uncased"
    
    # Image preprocessing
    image_size: int = 224
    image_augmentation: bool = True
    
    # Data loading
    batch_size: int = 32
    num_workers: int = 0


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # Text encoder
    text_encoder_type: str = "distilbert"  # or "lstm", "bert"
    text_hidden_dim: int = 768
    text_dropout: float = 0.1
    text_freeze_backbone: bool = False
    
    # Image encoder
    image_encoder_type: str = "efficientnet_b0"  # or "resnet50"
    image_hidden_dim: int = 768
    image_dropout: float = 0.1
    
    # Fusion
    fusion_type: str = "concat"  # or "bilinear", "gating", "cross_attention"
    fusion_hidden_dim: int = 512
    
    # Classifier
    classifier_hidden_dim: int = 256
    classifier_dropout: float = 0.2
    
    # General
    output_dim: int = 2  # Binary classification


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Optimization
    optimizer: str = "adamw"
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    
    # Loss
    loss_function: str = "focal"  # or "ce", "combined"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    
    # Training
    num_epochs: int = 10
    early_stopping_patience: int = 3
    metric_to_monitor: str = "f1"
    
    # Mixed precision
    mixed_precision: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_frequency: int = 1


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    
    # Model loading
    model_checkpoint: Optional[str] = None
    
    # Prediction
    confidence_threshold: float = 0.5
    return_embeddings: bool = False
    
    # Uncertainty
    use_mc_dropout: bool = False
    mc_samples: int = 10


@dataclass
class Config:
    """Main configuration class."""
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # General
    device: str = "cuda"
    seed: int = 42
    verbose: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Config':
        """
        Create config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Config instance
        """
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            inference=InferenceConfig(**config_dict.get('inference', {})),
            device=config_dict.get('device', 'cuda'),
            seed=config_dict.get('seed', 42),
            verbose=config_dict.get('verbose', True),
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> 'Config':
        """
        Load config from JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            Config instance
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict:
        """
        Convert config to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'inference': self.inference.__dict__,
            'device': self.device,
            'seed': self.seed,
            'verbose': self.verbose,
        }
    
    def save(self, path: str):
        """
        Save configuration to JSON file.
        
        Args:
            path: Path to save configuration
        """
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def __str__(self) -> str:
        """String representation."""
        return json.dumps(self.to_dict(), indent=2)


# Default configuration instances
DEFAULT_CONFIG = Config()

# Lightweight config for quick testing
QUICK_TEST_CONFIG = Config(
    data=DataConfig(batch_size=16),
    model=ModelConfig(
        text_hidden_dim=512,
        image_hidden_dim=512,
        fusion_hidden_dim=256,
    ),
    training=TrainingConfig(
        num_epochs=2,
        learning_rate=5e-5,
    ),
)

# Production config
PRODUCTION_CONFIG = Config(
    data=DataConfig(
        batch_size=64,
        num_workers=4,
    ),
    model=ModelConfig(
        text_freeze_backbone=True,
        fusion_type="cross_attention",
    ),
    training=TrainingConfig(
        num_epochs=15,
        early_stopping_patience=5,
    ),
    inference=InferenceConfig(
        use_mc_dropout=True,
        mc_samples=20,
    ),
)
