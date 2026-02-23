"""Base class for fingerprint extraction methods."""

import torch
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Dict, Any, Optional
import logging

from ..utils import setup_logger


class FingerprintExtractor(ABC):
    """
    Base class for fingerprint extraction methods.
    
    This class provides a common interface for different fingerprinting techniques
    to extract features from images for model attribution.
    """
    
    # Registry of available fingerprint methods
    _registry = {}
    
    def __init__(self, method_name: str, is_differentiable: bool = False, 
                 has_analytic_approx: bool = False, feature_dim: Optional[int] = None,
                 is_implicit_fingerprint: bool = False):
        """
        Initialize the fingerprint extractor.
        
        Args:
            method_name: Name of the fingerprinting method
            is_differentiable: Whether the extractor is differentiable
            has_analytic_approx: Whether an analytic approximation exists
            feature_dim: Dimension of extracted features
            is_implicit_fingerprint: Whether this is an implicit fingerprint method
                                    (the network itself is the attribution model)
        """
        self.method_name = method_name
        self.is_differentiable = is_differentiable
        self.has_analytic_approx = has_analytic_approx
        self.feature_dim = feature_dim
        self.is_implicit_fingerprint = is_implicit_fingerprint
        
        self.logger = setup_logger(f"FingerprintExtractor_{method_name}")
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register fingerprint methods."""
        def decorator(extractor_class):
            cls._registry[name] = extractor_class
            return extractor_class
        return decorator
    
    @classmethod
    def create(cls, method_name: str, **kwargs) -> 'FingerprintExtractor':
        """
        Factory method to create appropriate fingerprint extractor.
        
        Args:
            method_name: Name of the fingerprinting method
            **kwargs: Additional parameters for the method
            
        Returns:
            Appropriate FingerprintExtractor instance
        """
        if method_name not in cls._registry:
            raise ValueError(f"Unknown fingerprint method: {method_name}. "
                           f"Available methods: {list(cls._registry.keys())}")
        
        extractor_class = cls._registry[method_name]
        return extractor_class(**kwargs)
    
    @abstractmethod
    def extract_fingerprint(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract fingerprints from images.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Fingerprint features tensor of shape (N, feature_dim)
        """
        pass
    
    def predict_attribution(self, images: torch.Tensor) -> torch.Tensor:
        """
        Predict model attribution directly (for implicit fingerprint methods).
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Attribution logits tensor of shape (N, num_classes)
        """
        if not self.is_implicit_fingerprint:
            raise NotImplementedError(f"Method {self.method_name} is not an implicit fingerprint method")
        
        # Default implementation for implicit methods
        return self.extract_fingerprint(images)
    
    def get_attribution_model(self) -> Optional[torch.nn.Module]:
        """
        Get the attribution model (for implicit fingerprint methods).
        
        Returns:
            The attribution model if this is an implicit fingerprint method, None otherwise
        """
        if not self.is_implicit_fingerprint:
            return None
        
        # For implicit methods, the extractor itself is the model
        return self if hasattr(self, 'forward') else None
    
    def extract_and_save(self, images: torch.Tensor, model_name: str, 
                        dataset: str, image_size: int) -> torch.Tensor:
        """
        Extract fingerprints and save to file.
        
        Args:
            images: Input images tensor
            model_name: Name of the generative model
            dataset: Dataset name
            image_size: Image size
            
        Returns:
            Extracted fingerprint features
        """
        self.logger.info(f"Extracting {self.method_name} fingerprints from {len(images)} images")
        
        features = self.extract_fingerprint(images)
        
        # Note: This method is deprecated. The new entry scripts (train_models.py, run_attacks.py)
        # handle feature saving directly with the new directory structure.
        # This method is kept for backward compatibility but is not used by the main workflow.
        
        self.logger.warning("extract_and_save() is deprecated. Use the new entry scripts instead.")
        
        return features
    
    def load_features(self, model_name: str, dataset: str, image_size: int) -> torch.Tensor:
        """
        Load previously extracted features.
        
        Args:
            model_name: Name of the generative model
            dataset: Dataset name
            image_size: Image size
            
        Returns:
            Loaded fingerprint features
        """
        # Note: This method is deprecated. The new entry scripts (train_models.py, run_attacks.py)
        # handle feature loading directly with the new directory structure.
        # This method is kept for backward compatibility but is not used by the main workflow.
        
        self.logger.warning("load_features() is deprecated. Use the new entry scripts instead.")
        raise NotImplementedError("Use the new entry scripts (train_models.py, run_attacks.py) for feature loading")
    
    def get_method_info(self) -> Dict[str, Any]:
        """Get information about the fingerprinting method."""
        return {
            'method_name': self.method_name,
            'is_differentiable': self.is_differentiable,
            'has_analytic_approx': self.has_analytic_approx,
            'feature_dim': self.feature_dim,
            'class_name': self.__class__.__name__
        }
    
    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images before fingerprint extraction.
        
        Args:
            images: Input images tensor
            
        Returns:
            Preprocessed images tensor
        """
        # Default preprocessing: ensure images are in [0, 1] range
        if images.min() < 0:
            # Convert from [-1, 1] to [0, 1]
            images = (images + 1) / 2
        
        # Ensure values are in [0, 1]
        images = torch.clamp(images, 0, 1)
        
        return images
    
    def batch_extract(self, images: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        """
        Extract fingerprints in batches to handle large datasets.
        
        Args:
            images: Input images tensor
            batch_size: Batch size for processing
            
        Returns:
            Extracted fingerprint features
        """
        num_images = len(images)
        all_features = []
        
        for i in range(0, num_images, batch_size):
            end_idx = min(i + batch_size, num_images)
            batch_images = images[i:end_idx]
            
            batch_features = self.extract_fingerprint(batch_images)
            all_features.append(batch_features)
            
            if (i // batch_size + 1) % 10 == 0:
                self.logger.info(f"Processed {end_idx}/{num_images} images")
        
        return torch.cat(all_features, dim=0)


class AnalyticApproximation(ABC):
    """
    Base class for analytic approximations of non-differentiable fingerprint methods.
    
    This is used for W3 attacks where we need differentiable approximations
    of originally non-differentiable fingerprint extractors.
    """
    
    def __init__(self, original_extractor: FingerprintExtractor):
        """
        Initialize with reference to original extractor.
        
        Args:
            original_extractor: The original non-differentiable extractor
        """
        self.original_extractor = original_extractor
        self.method_name = f"{original_extractor.method_name}_approx"
        self.is_differentiable = True
        self.logger = setup_logger(f"AnalyticApprox_{original_extractor.method_name}")
    
    @abstractmethod
    def extract_fingerprint_approx(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract fingerprints using differentiable approximation.
        
        Args:
            images: Input images tensor
            
        Returns:
            Approximate fingerprint features tensor
        """
        pass
    
    def validate_approximation(self, images: torch.Tensor, tolerance: float = 0.1) -> Dict[str, float]:
        """
        Validate the approximation against the original method.
        
        Args:
            images: Test images
            tolerance: Acceptable difference tolerance
            
        Returns:
            Validation metrics
        """
        with torch.no_grad():
            original_features = self.original_extractor.extract_fingerprint(images)
            approx_features = self.extract_fingerprint_approx(images)
            
            # Calculate differences
            mse = torch.mean((original_features - approx_features) ** 2).item()
            mae = torch.mean(torch.abs(original_features - approx_features)).item()
            max_error = torch.max(torch.abs(original_features - approx_features)).item()
            
            # Check if approximation is within tolerance
            within_tolerance = max_error <= tolerance
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'max_error': max_error,
                'within_tolerance': within_tolerance,
                'tolerance': tolerance
            }
            
            self.logger.info(f"Approximation validation: MSE={mse:.6f}, MAE={mae:.6f}, "
                           f"Max Error={max_error:.6f}, Within Tolerance={within_tolerance}")
            
            return metrics