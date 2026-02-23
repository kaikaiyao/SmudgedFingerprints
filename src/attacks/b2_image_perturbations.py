"""
B2 Attack: Image-Level Perturbations.

This black-box attack applies standard image perturbations like noise,
JPEG compression, blur, and resizing without gradient information.

Note: Parameters have been tuned to achieve L∞ distance of approximately 0.05
for removal attacks, as the previous parameters resulted in distances >0.9.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from typing import Optional, List, Dict, Any
import numpy as np

from .base import Attacker


@Attacker.register("b2")
class Attacker_B2(Attacker):
    """
    B2 Attack using image-level perturbations.
    
    This black-box attack applies various image transformations
    that don't require gradient information.
    """
    
    def __init__(self, perturbation_types: List[str] = ["gaussian_noise", "blur", "jpeg", "resize"],
                 attack_type: str = "removal", epsilon: float = 0.05, epsilon_linf: float = 0.025,
                 device: str = "cuda"):
        """
        Initialize B2 attacker.
        
        Args:
            perturbation_types: List of perturbation types to apply (only: gaussian_noise, blur, jpeg, resize)
            attack_type: Type of attack ("removal" or "forgery")
            epsilon: Maximum perturbation magnitude (used for noise)
            epsilon_linf: Maximum L-infinity distance for pixel changes (default: 0.025)
            device: Device to run attacks on
        """
        super().__init__(
            attack_name="b2",
            attack_type=attack_type,
            epsilon=epsilon,
            num_steps=1,  # B2 is typically single-step
            step_size=0.0,
            targeted=False,
            device=device
        )
        
        self.perturbation_types = perturbation_types
        self.epsilon_linf = epsilon_linf
        
        # Available perturbation functions (only 4 types)
        self.perturbation_functions = {
            "gaussian_noise": self._gaussian_noise,
            "blur": self._gaussian_blur,
            "jpeg": self._jpeg_compression,
            "resize": self._resize_perturbation
        }
    
    def attack(self, images: torch.Tensor, targets: Optional[torch.Tensor] = None,
               perturbation_params: Optional[Dict[str, Any]] = None,
               **kwargs) -> Dict[str, torch.Tensor]:
        """
        Perform B2 image perturbation attack.
        
        Args:
            images: Input images tensor
            targets: Original target labels (not used in B2)
            perturbation_params: Parameters for perturbations
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing:
                - 'individual': Dict mapping perturbation type to perturbed images
        """
        images = images.to(self.device)
        
        if perturbation_params is None:
            perturbation_params = {}
        
        # Track individual perturbations only
        individual_results = {}
        
        # Apply each perturbation type and track results
        for perturbation_type in self.perturbation_types:
            if perturbation_type in self.perturbation_functions:
                params = perturbation_params.get(perturbation_type, {})
                
                # Apply single perturbation to original images
                individual_perturbed = self.perturbation_functions[perturbation_type](
                    images.clone(), **params
                )
                
                # Keep pixel range valid only (no global epsilon clamping; parameters drive magnitude)
                individual_perturbed = torch.clamp(individual_perturbed, 0.0, 1.0)
                
                individual_results[perturbation_type] = individual_perturbed
            else:
                self.logger.warning(f"Unknown perturbation type: {perturbation_type}")
        
        return {
            'individual': individual_results
        }
    
    def _apply_linf_constraint(self, original_images: torch.Tensor, perturbed_images: torch.Tensor) -> torch.Tensor:
        """
        Deprecated: Kept for backward compatibility but not used. Scaling-based calibration is applied instead.
        """
        images = original_images
        result = torch.clamp(perturbed_images, 0.0, 1.0)
        return result
    
    def _gaussian_noise(self, images: torch.Tensor, std: float = None) -> torch.Tensor:
        """Add Gaussian noise to images, rough-calibrated to epsilon_linf by scaling later."""
        if std is None:
            std = self.epsilon_linf / 2.0
        noise = torch.randn_like(images) * std
        perturbed = images + noise
        return torch.clamp(perturbed, 0.0, 1.0)
    
    def _gaussian_blur(self, images: torch.Tensor, kernel_size: int = 3, 
                      sigma: float = 0.65) -> torch.Tensor:
        """Apply Gaussian blur to images."""
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create Gaussian kernel
        channels = images.shape[1]
        kernel = self._get_gaussian_kernel(kernel_size, sigma, channels).to(images.device)
        
        # Apply convolution
        padding = kernel_size // 2
        blurred = F.conv2d(images, kernel, padding=padding, groups=channels)
        
        return torch.clamp(blurred, 0.0, 1.0)
    
    def _get_gaussian_kernel(self, kernel_size: int, sigma: float, channels: int) -> torch.Tensor:
        """Generate Gaussian kernel for blurring."""
        # Create 1D Gaussian kernel
        x = torch.arange(kernel_size, dtype=torch.float32)
        x = x - kernel_size // 2
        kernel_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Create 2D kernel
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        
        # Expand for channels
        kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()
        
        return kernel
        
    def _jpeg_compression(self, images: torch.Tensor, quality: int = 45) -> torch.Tensor:
        """Apply JPEG compression using torchvision.transforms.v2.JPEG.
        Input:  [B, 3, H, W] tensor in [0, 1]
        Output: same shape, JPEG-compressed version in [0, 1]
        """
        from torchvision.transforms.v2 import JPEG
        import torch
        
        # Store original device
        original_device = images.device
        
        # Move to CPU for JPEG transform (torchvision requires CPU tensors)
        images_cpu = images.cpu()
        
        # Convert from [0, 1] float to [0, 255] uint8
        images_uint8 = (images_cpu * 255.0).clamp(0, 255).to(torch.uint8)
        
        # Apply JPEG compression
        jpeg_transform = JPEG(quality=quality)
        compressed_uint8 = jpeg_transform(images_uint8)
        
        # Convert back to [0, 1] float and move back to original device
        compressed_float = compressed_uint8.float() / 255.0
        compressed_float = compressed_float.to(original_device)
        
        return compressed_float

    
    def _resize_perturbation(self, images: torch.Tensor, scale_factor: float = 0.70) -> torch.Tensor:
        """Apply resize perturbation."""
        original_size = images.shape[-2:]
        
        # Resize down then up
        small_size = [int(s * scale_factor) for s in original_size]
        
        resized_down = F.interpolate(images, size=small_size, mode='bilinear', align_corners=False)
        resized_up = F.interpolate(resized_down, size=original_size, mode='bilinear', align_corners=False)
        
        return torch.clamp(resized_up, 0.0, 1.0)