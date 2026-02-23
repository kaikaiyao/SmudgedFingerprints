"""Base class for model samplers."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import sys
import os
import gc
import importlib
import weakref

from ..utils import save_images, setup_logger


class ModelSampler(ABC):
    """
    Base class for model samplers.
    
    Provides a unified interface for loading and sampling from different generative models.
    """
    
    _registry = {}
    
    def __init__(self, model_name: str, dataset: str, image_size: int, 
                 device: str = "cuda", batch_size: int = 32):
        self.model_name = model_name
        self.dataset = dataset
        self.image_size = image_size
        self.device = device
        self.batch_size = batch_size
        self._model = None
        
        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Hugging Face may try to route downloads through the optional Xet backend,
        # which requires an extra dependency. Disable it globally so standard HTTP
        # downloads are used unless explicitly overridden.
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        
        # Model isolation state
        self._original_sys_path = None
        self._original_modules = None
        self._loaded_modules = set()
        self._is_isolated = False
    
    @classmethod
    def register(cls, name: str):
        """Register a model sampler class."""
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator
    
    @classmethod
    def create(cls, model_name: str, dataset: str, image_size: int, 
               device: str = "cuda", batch_size: int = 32):
        """Create a model sampler instance."""
        if model_name not in cls._registry:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls._registry.keys())}")
        
        return cls._registry[model_name](model_name, dataset, image_size, device, batch_size)
    
    def _setup_model_isolation(self):
        """Setup isolation environment for model loading."""
        if self._is_isolated:
            return
        
        # Store original state
        self._original_sys_path = sys.path.copy()
        self._original_modules = set(sys.modules.keys())
        self._loaded_modules = set()
        self._is_isolated = True
        
        self.logger.debug(f"Setup model isolation for {self.model_name}")
    
    def _cleanup_model_isolation(self):
        """Clean up isolation environment after model loading."""
        if not self._is_isolated:
            return
        
        try:
            # Get current modules that were added during loading
            current_modules = set(sys.modules.keys())
            new_modules = current_modules - self._original_modules
            
            # Remove all newly loaded modules, except for PyO3-dependent ones
            for module_name in list(new_modules):
                # Skip diffusers and related modules to avoid PyO3 re-initialization issues
                if any(prefix in module_name for prefix in ['diffusers', 'transformers', 'accelerate', 'tensor_transforms']):
                    self.logger.debug(f"Skipping cleanup of module that needs to persist: {module_name}")
                    continue
                    
                try:
                    if module_name in sys.modules:
                        del sys.modules[module_name]
                        self._loaded_modules.add(module_name)
                except Exception as e:
                    self.logger.debug(f"Could not remove module {module_name}: {e}")
            
            # Restore original sys.path
            sys.path = self._original_sys_path.copy()
            
            # Force garbage collection
            gc.collect()
            
            # Clear any cached imports (except modules that need to persist)
            for module_name in self._loaded_modules:
                if any(prefix in module_name for prefix in ['diffusers', 'transformers', 'accelerate', 'tensor_transforms']):
                    continue
                try:
                    if module_name in sys.modules:
                        del sys.modules[module_name]
                except:
                    pass
            
            self.logger.debug(f"Cleaned up isolation for {self.model_name}")
            
        except Exception as e:
            self.logger.warning(f"Error during model isolation cleanup: {e}")
        finally:
            # Reset isolation state
            self._original_sys_path = None
            self._original_modules = None
            self._loaded_modules = set()
            self._is_isolated = False
    
    def _isolated_import(self, module_name: str, package_path: str = None):
        """Import a module in isolation, ensuring it doesn't conflict with other models."""
        try:
            # Add package path to sys.path temporarily if provided
            original_path = None
            if package_path:
                original_path = sys.path.copy()
                if package_path not in sys.path:
                    sys.path.insert(0, package_path)
            
            # Import the module
            module = importlib.import_module(module_name)
            
            # Restore original path
            if original_path:
                sys.path = original_path
            
            return module
            
        except Exception as e:
            self.logger.error(f"Failed to import {module_name}: {e}")
            raise
    
    def _safe_model_loading(self, loading_function):
        """Execute model loading function in a safe, isolated environment."""
        self._setup_model_isolation()
        
        try:
            # Execute the actual loading function
            result = loading_function()
            return result
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            raise
        finally:
            # Always cleanup, even if loading failed
            self._cleanup_model_isolation()
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load the model. Override this method in subclasses."""
        pass
    
    def load_model(self) -> None:
        """Load the model with proper isolation."""
        if self._model is not None:
            self.logger.info(f"Model {self.model_name} already loaded")
            return
        
        self.logger.info(f"Loading model {self.model_name}...")
        
        # Use safe loading wrapper
        self._safe_model_loading(self._load_model)
        
        if self._model is None:
            raise RuntimeError(f"Failed to load model {self.model_name}")
        
        self.logger.info(f"Successfully loaded model {self.model_name}")
    
    @abstractmethod
    def _generate_batch(self, batch_size: int, **kwargs) -> torch.Tensor:
        """Generate a batch of images. Override this method in subclasses."""
        pass
    
    def sample(self, num_images: int, save_to_file: bool = True, **kwargs) -> torch.Tensor:
        """Sample images from the model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Generate images in batches
        all_images = []
        remaining = num_images
        
        while remaining > 0:
            batch_size = min(self.batch_size, remaining)
            batch_images = self._generate_batch(batch_size, **kwargs)
            all_images.append(batch_images)
            remaining -= batch_size
        
        # Concatenate all batches
        images = torch.cat(all_images, dim=0)
        
        # Save to file if requested
        if save_to_file:
            self._save_images(images)
        
        return images
    
    def _save_images(self, images: torch.Tensor) -> None:
        """Save generated images to file."""
        # Implementation depends on specific requirements
        pass
    
    @property
    def model(self) -> Optional[nn.Module]:
        """Get the underlying model."""
        return self._model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model_name": self.model_name,
            "dataset": self.dataset,
            "image_size": self.image_size,
            "device": self.device,
            "batch_size": self.batch_size
        }
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        try:
            if self._is_isolated:
                self._cleanup_model_isolation()
        except:
            pass
