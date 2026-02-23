"""
Device utility functions for cross-platform compatibility.
Supports CUDA, Apple MPS (M1/M2/M3/M4), and CPU.
"""

import torch
import logging

def get_optimal_device() -> str:
    """
    Get the optimal device for the current system.
    
    Priority: CUDA > MPS (Apple Silicon) > CPU
    
    Returns:
        str: Device string ("cuda", "mps", or "cpu")
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def setup_device(preferred_device: str = None) -> str:
    """
    Setup and validate device for PyTorch operations.
    
    Args:
        preferred_device: Preferred device ("cuda", "mps", "cpu", or None for auto)
        
    Returns:
        str: Validated device string
    """
    logger = logging.getLogger(__name__)
    
    # If no preference, use optimal device
    if preferred_device is None:
        device = get_optimal_device()
        logger.info(f"Auto-selected device: {device}")
        return device
    
    # Validate preferred device
    if preferred_device == "cuda":
        if torch.cuda.is_available():
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            return "cuda"
        else:
            logger.warning("CUDA requested but not available, falling back to optimal device")
            return get_optimal_device()
    
    elif preferred_device == "mps":
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Using Apple MPS device")
            return "mps"
        else:
            logger.warning("MPS requested but not available, falling back to optimal device")
            return get_optimal_device()
    
    elif preferred_device == "cpu":
        logger.info("Using CPU device")
        return "cpu"
    
    else:
        logger.warning(f"Unknown device '{preferred_device}', using optimal device")
        return get_optimal_device()

def create_generator(device: str, seed: int = None) -> torch.Generator:
    """
    Create a PyTorch generator for the specified device.
    
    Args:
        device: Device string ("cuda", "mps", or "cpu")
        seed: Random seed (optional)
        
    Returns:
        torch.Generator: Generator object for the device
    """
    if device == "cuda":
        generator = torch.Generator(device="cuda")
    elif device == "mps":
        # MPS generators need to be created as CPU generators
        generator = torch.Generator(device="cpu")
    else:
        generator = torch.Generator(device="cpu")
    
    if seed is not None:
        generator.manual_seed(seed)
    
    return generator

def clear_cache(device: str):
    """
    Clear device memory cache.
    
    Args:
        device: Device string ("cuda", "mps", or "cpu")
    """
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't have explicit cache clearing, but we can trigger garbage collection
        import gc
        gc.collect()

def get_device_info(device: str) -> dict:
    """
    Get information about the specified device.
    
    Args:
        device: Device string
        
    Returns:
        dict: Device information
    """
    info = {"device": device, "available": True}
    
    if device == "cuda":
        if torch.cuda.is_available():
            info.update({
                "name": torch.cuda.get_device_name(0),
                "memory_total": torch.cuda.get_device_properties(0).total_memory,
                "device_count": torch.cuda.device_count()
            })
        else:
            info["available"] = False
    
    elif device == "mps":
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info.update({
                "name": "Apple MPS",
                "memory_total": "Unknown",  # MPS doesn't expose memory info
                "device_count": 1
            })
        else:
            info["available"] = False
    
    else:  # CPU
        import psutil
        info.update({
            "name": "CPU",
            "memory_total": psutil.virtual_memory().total,
            "device_count": psutil.cpu_count()
        })
    
    return info

def set_seed_all_devices(seed: int):
    """
    Set random seed for all available devices.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # MPS uses the same seed as CPU
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)