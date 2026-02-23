"""Configuration management for the fingerprint robustness framework."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Configuration for generative models."""
    name: str
    dataset: str
    image_size: int
    checkpoint_path: Optional[str] = None
    device: str = None  # Auto-detect optimal device
    batch_size: int = 32


@dataclass
class FingerprintConfig:
    """Configuration for fingerprinting methods."""
    method: str
    is_differentiable: bool
    has_analytic_approx: bool = False
    feature_dim: Optional[int] = None
    params: Dict[str, Any] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class AttackConfig:
    """Configuration for attack strategies."""
    strategy: str  # w1, w2, w3, b1, b2
    attack_type: str  # removal, forgery
    epsilon: float = 0.1
    num_steps: int = 100
    step_size: float = 0.01
    targeted: bool = False
    confidence_penalty: float = 0.0


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    name: str
    model: ModelConfig
    fingerprint: FingerprintConfig
    attack: Optional[AttackConfig] = None
    num_images: int = 1000
    seed: int = 42
    output_dir: str = "experiments"


class Config:
    """Main configuration class."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.experiments_dir = self.project_root / "experiments"
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.experiments_dir.mkdir(exist_ok=True)
        
        if config_path:
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Parse configuration sections
        if 'model' in config_dict:
            self.model = ModelConfig(**config_dict['model'])
        if 'fingerprint' in config_dict:
            self.fingerprint = FingerprintConfig(**config_dict['fingerprint'])
        if 'attack' in config_dict:
            self.attack = AttackConfig(**config_dict['attack'])
        if 'experiment' in config_dict:
            self.experiment = ExperimentConfig(**config_dict['experiment'])
    
    def save_to_file(self, config_path: str):
        """Save configuration to YAML file."""
        config_dict = {}
        if hasattr(self, 'model'):
            config_dict['model'] = asdict(self.model)
        if hasattr(self, 'fingerprint'):
            config_dict['fingerprint'] = asdict(self.fingerprint)
        if hasattr(self, 'attack'):
            config_dict['attack'] = asdict(self.attack)
        if hasattr(self, 'experiment'):
            config_dict['experiment'] = asdict(self.experiment)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


def load_config(config_path: str) -> Config:
    """Load configuration from file."""
    return Config(config_path)