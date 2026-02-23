"""
Main model loading orchestrator.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from .attribution_model_loader import AttributionModelLoader


class ModelLoader:
    """Main class for loading all required models."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.attribution_loader = AttributionModelLoader(device)
    
    def load_all_models(
        self,
        h_model_path: str,
        h_s_model_path: str,
        phi_s_model_path: str,
        fingerprint_method: str,
        num_models: int,
        feature_dim: int,
        real_data_path: Optional[str] = None,
        cache_dir: str = "cache",
        data_dir: Optional[str] = None
    ) -> Tuple[torch.nn.Module, torch.nn.Module, Optional[torch.nn.Module]]:
        """
        Load all three pre-trained models.
        
        Args:
            h_model_path: Path to true attribution model
            h_s_model_path: Path to surrogate attribution model
            phi_s_model_path: Path to surrogate extractor model
            fingerprint_method: Fingerprint method name
            num_models: Number of models/classes
            feature_dim: Feature dimension
            real_data_path: Path to real data (for some methods)
            cache_dir: Cache directory
            data_dir: Data directory
            
        Returns:
            Tuple of (true_attribution_model, surrogate_attribution_model, surrogate_extractor)
        """
        
        print("Loading pre-trained models...")
        
        # Check if this is an implicit fingerprint method
        try:
            from src.data_loader.fingerprint_extractor_factory import FingerprintExtractorFactory
            is_implicit = FingerprintExtractorFactory.is_implicit_method(
                fingerprint_method,
                self.device,
                real_data_path=real_data_path,
                cache_dir=cache_dir,
                data_dir=data_dir
            )
        except ImportError:
            # Fallback: assume it's not implicit if we can't check
            is_implicit = False
        
        # Load True Attribution Model (h) or Implicit Fingerprint Model
        true_attribution_model = self.attribution_loader.load_true_attribution_model(
            h_model_path, fingerprint_method, num_models, feature_dim, is_implicit,
            real_data_path, cache_dir, data_dir
        )
        
        # Load Surrogate Attribution Model (h_s)
        surrogate_attribution_model = self.attribution_loader.load_surrogate_attribution_model(
            h_s_model_path, fingerprint_method, num_models
        )
        
        # Load Surrogate Extractor (φ_s) - only needed for explicit fingerprint methods
        surrogate_extractor = self.attribution_loader.load_surrogate_extractor(
            phi_s_model_path, fingerprint_method, is_implicit
        )
        
        print("✅ All models loaded successfully!")
        
        return true_attribution_model, surrogate_attribution_model, surrogate_extractor
