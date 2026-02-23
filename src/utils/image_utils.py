"""Image processing utilities."""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Union, List, Optional
import torchvision.transforms as transforms


def save_images(images: torch.Tensor, filepath: Union[str, Path], 
                metadata: Optional[dict] = None) -> None:
    """
    Save images tensor to file with optional metadata.
    
    Args:
        images: Tensor of shape (N, C, H, W)
        filepath: Path to save the images
        metadata: Optional metadata dictionary
    """
    data = {
        'images': images,
        'metadata': metadata or {}
    }
    torch.save(data, filepath)


def load_images(filepath: Union[str, Path]) -> tuple[torch.Tensor, dict]:
    """
    Load images from file.
    
    Args:
        filepath: Path to the images file
        
    Returns:
        Tuple of (images tensor, metadata dict)
    """
    data = torch.load(filepath, map_location='cpu')
    if isinstance(data, dict):
        return data['images'], data.get('metadata', {})
    else:
        # Legacy format - just images
        return data, {}


def normalize_images(images: torch.Tensor, 
                    source_range: tuple = (-1, 1),
                    target_range: tuple = (0, 1)) -> torch.Tensor:
    """
    Normalize images from source range to target range.
    
    Args:
        images: Input images tensor
        source_range: Current range of pixel values
        target_range: Desired range of pixel values
        
    Returns:
        Normalized images tensor
    """
    source_min, source_max = source_range
    target_min, target_max = target_range
    
    # Normalize to [0, 1]
    normalized = (images - source_min) / (source_max - source_min)
    
    # Scale to target range
    scaled = normalized * (target_max - target_min) + target_min
    
    return scaled


def tensor_to_pil(tensor: torch.Tensor) -> List[Image.Image]:
    """
    Convert tensor to PIL Images.
    
    Args:
        tensor: Images tensor of shape (N, C, H, W) or (C, H, W)
        
    Returns:
        List of PIL Images
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    # Normalize to [0, 1] if needed
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = normalize_images(tensor, source_range=(-1, 1), target_range=(0, 1))
    
    # Convert to PIL
    to_pil = transforms.ToPILImage()
    images = []
    for i in range(tensor.size(0)):
        img = to_pil(tensor[i])
        images.append(img)
    
    return images


def pil_to_tensor(images: List[Image.Image]) -> torch.Tensor:
    """
    Convert PIL Images to tensor.
    
    Args:
        images: List of PIL Images
        
    Returns:
        Images tensor of shape (N, C, H, W)
    """
    to_tensor = transforms.ToTensor()
    tensors = [to_tensor(img) for img in images]
    return torch.stack(tensors)


def add_noise(images: torch.Tensor, noise_type: str = "gaussian", 
              strength: float = 0.1) -> torch.Tensor:
    """
    Add noise to images.
    
    Args:
        images: Input images tensor
        noise_type: Type of noise ("gaussian", "uniform")
        strength: Noise strength
        
    Returns:
        Noisy images tensor
    """
    if noise_type == "gaussian":
        noise = torch.randn_like(images) * strength
    elif noise_type == "uniform":
        noise = (torch.rand_like(images) - 0.5) * 2 * strength
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return images + noise


def clip_images(images: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
    """
    Clip image values to valid range.
    
    Args:
        images: Input images tensor
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clipped images tensor
    """
    return torch.clamp(images, min_val, max_val)


def rgb_to_grayscale(images: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB images to grayscale using standard coefficients.
    
    Args:
        images: Input RGB images tensor of shape (N, 3, H, W)
        
    Returns:
        Grayscale images tensor of shape (N, 1, H, W)
    """
    if images.shape[1] != 3:
        raise ValueError(f"Expected RGB images with 3 channels, got {images.shape[1]} channels")
    
    # Standard RGB to grayscale coefficients (ITU-R BT.601)
    rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=images.device)
    
    # Compute weighted sum along channel dimension
    grayscale = torch.sum(images * rgb_weights.view(1, 3, 1, 1), dim=1, keepdim=True)
    
    return grayscale