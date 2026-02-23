"""
Model isolation utilities for preventing conflicts between different generative models.

This module provides utilities to ensure that each model loads in a clean environment
and properly cleans up after itself to prevent interference with subsequent models.
"""

import sys
import os
import gc
import importlib
import logging
from typing import Optional, Callable, Any, Dict, Set
from contextlib import contextmanager
from pathlib import Path
import weakref

logger = logging.getLogger(__name__)


class ModelIsolationManager:
    """
    Manages isolation between different generative models to prevent conflicts.
    
    This class ensures that each model loads in a clean environment and properly
    cleans up after itself to prevent interference with subsequent models.
    """
    
    def __init__(self):
        self._original_sys_path = None
        self._original_modules = None
        self._loaded_modules = set()
        self._is_active = False
        self._model_paths = set()
        
    def setup_isolation(self, model_name: str) -> None:
        """Setup isolation environment for a specific model."""
        if self._is_active:
            logger.warning(f"Isolation already active, cleaning up before setting up for {model_name}")
            self.cleanup_isolation()
        
        # Store original state
        self._original_sys_path = sys.path.copy()
        self._original_modules = set(sys.modules.keys())
        self._loaded_modules = set()
        self._model_paths = set()
        self._is_active = True
        
        logger.debug(f"Setup model isolation for {model_name}")
    
    def cleanup_isolation(self) -> None:
        """Clean up isolation environment."""
        if not self._is_active:
            return
        
        try:
            # Get current modules that were added during loading
            current_modules = set(sys.modules.keys())
            new_modules = current_modules - self._original_modules
            
            # Remove all newly loaded modules
            for module_name in list(new_modules):
                try:
                    if module_name in sys.modules:
                        del sys.modules[module_name]
                        self._loaded_modules.add(module_name)
                        logger.debug(f"Removed module: {module_name}")
                except Exception as e:
                    logger.debug(f"Could not remove module {module_name}: {e}")
            
            # Restore original sys.path
            sys.path = self._original_sys_path.copy()
            
            # Force garbage collection
            gc.collect()
            
            # Clear any cached imports
            for module_name in self._loaded_modules:
                try:
                    if module_name in sys.modules:
                        del sys.modules[module_name]
                except:
                    pass
            
            logger.debug("Cleaned up model isolation")
            
        except Exception as e:
            logger.warning(f"Error during model isolation cleanup: {e}")
        finally:
            # Reset isolation state
            self._original_sys_path = None
            self._original_modules = None
            self._loaded_modules = set()
            self._model_paths = set()
            self._is_active = False
    
    def add_model_path(self, path: str) -> None:
        """Add a model-specific path to sys.path during isolation."""
        if not self._is_active:
            logger.warning("Cannot add path: isolation not active")
            return
        
        if path not in sys.path:
            sys.path.insert(0, path)
            self._model_paths.add(path)
            logger.debug(f"Added model path: {path}")
    
    def remove_model_path(self, path: str) -> None:
        """Remove a model-specific path from sys.path."""
        if path in sys.path:
            sys.path.remove(path)
            self._model_paths.discard(path)
            logger.debug(f"Removed model path: {path}")
    
    def isolated_import(self, module_name: str, package_path: Optional[str] = None) -> Any:
        """Import a module in isolation."""
        if not self._is_active:
            logger.warning("Cannot import in isolation: isolation not active")
            return importlib.import_module(module_name)
        
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
            logger.error(f"Failed to import {module_name}: {e}")
            raise
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_isolation()


@contextmanager
def model_isolation(model_name: str):
    """
    Context manager for model isolation.
    
    Usage:
        with model_isolation("ganformer-ffhq-256"):
            # Load and use model here
            pass
        # Isolation is automatically cleaned up
    """
    manager = ModelIsolationManager()
    try:
        manager.setup_isolation(model_name)
        yield manager
    finally:
        manager.cleanup_isolation()


def safe_model_loading(loading_function: Callable, model_name: str) -> Any:
    """
    Execute a model loading function in a safe, isolated environment.
    
    Args:
        loading_function: Function that loads the model
        model_name: Name of the model for logging
        
    Returns:
        Result of the loading function
        
    Raises:
        Exception: If loading fails
    """
    with model_isolation(model_name) as isolation:
        try:
            logger.info(f"Loading model {model_name} in isolated environment...")
            result = loading_function()
            logger.info(f"Successfully loaded model {model_name}")
            return result
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise


def cleanup_all_models():
    """Force cleanup of all model-related resources."""
    try:
        # Clear all modules that might be from models
        # Note: Excluding 'diffusers', 'ncsnpp', and 'tensor_transforms' to avoid module re-initialization issues
        model_module_prefixes = [
            'dnnlib', 'legacy', 'training', 'models', 'networks', 'utils', 'ops',
            'FP16Stages', 'torch_utils', 'generator', 'loader', 'stylegan', 'r3gan',
            'ganformer', 'vdvae', 'styleswin', 'cips',
            'vae', 'hps', 'model', 'vitvqganvae', 'vector_quantize_pytorch', 'nvae'
        ]
        
        modules_to_remove = []
        for module_name in list(sys.modules.keys()):
            if any(module_name.startswith(prefix) for prefix in model_module_prefixes):
                modules_to_remove.append(module_name)
        
        for module_name in modules_to_remove:
            try:
                del sys.modules[module_name]
                logger.debug(f"Cleaned up module: {module_name}")
            except Exception as e:
                logger.debug(f"Could not clean up module {module_name}: {e}")
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        # Clear MPS cache if available
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except:
            pass
        
        logger.info("Cleaned up all model resources")
        
    except Exception as e:
        logger.warning(f"Error during global cleanup: {e}")


def get_model_load_order() -> Dict[str, int]:
    """
    Get recommended loading order for models to minimize conflicts.
    
    Returns:
        Dictionary mapping model names to priority (lower = load first)
    """
    return {
        # Load simpler models first
        'r3gan-ffhq-256': 1,        # Modern GAN
        'ganformer-ffhq-256': 2,    # GAN with attention
        'adm-ffhq-256': 3,          # ADM variant
        'ncsnpp-ffhq-256': 4,       # Score-based diffusion
        'ldm-ffhq-256': 5,          # HF latent diffusion
        'vdvae-ffhq-256': 6,        # Simple VAE
        'nvae-ffhq-256': 7,         # Hierarchical VAE
        'vqvae-ffhq-256': 8,        # VQ-VAE variant
        'styleswin-ffhq-256': 9,    # Transformer-based
        'cips-ffhq-256': 10,        # Coordinate-based
        'stylegan2-ffhq-256': 11,   # StyleGAN2
        'stylegan3-ffhq-256': 12,   # Most complex, load last
    }


def sort_models_by_load_order(model_names: list) -> list:
    """
    Sort model names by recommended loading order.
    
    Args:
        model_names: List of model names to sort
        
    Returns:
        Sorted list of model names
    """
    load_order = get_model_load_order()
    
    def get_priority(model_name):
        return load_order.get(model_name, 999)  # Unknown models get lowest priority
    
    return sorted(model_names, key=get_priority)
