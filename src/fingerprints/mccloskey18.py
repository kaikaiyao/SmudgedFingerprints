"""
McCloskey18 fingerprint extraction method implementation.

Based on "Detecting GAN-generated Imagery using Color Cues" by Scott McCloskey and Michael Albright (Honeywell ACST).

This method detects GAN-generated images by exploiting the observation that GANs tend to lack 
over-exposed (saturated) pixels due to normalization layers in the generator architecture.

The fingerprint measures the proportion of saturated pixels for different thresholds [240, 245, 250, 255]
in grayscale intensity, producing a 4-dimensional feature vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

from .base import FingerprintExtractor, AnalyticApproximation


@FingerprintExtractor.register("mccloskey18")
class McCloskey18(FingerprintExtractor):
    """
    McCloskey18 fingerprint extractor using saturation-based detection.
    
    Computes the proportion of saturated pixels for different thresholds in grayscale intensity,
    producing a 4-dimensional feature vector as described in the original paper.
    """
    
    def __init__(self, thresholds: Optional[list] = None):
        """
        Initialize McCloskey18 extractor.
        
        Args:
            thresholds: List of saturation thresholds (default: [240, 245, 250, 255])
        """
        if thresholds is None:
            thresholds = [240, 245, 250, 255]
        
        # Feature dimension is the number of thresholds
        feature_dim = len(thresholds)
        
        super().__init__(
            method_name="mccloskey18",
            is_differentiable=False,
            has_analytic_approx=True,
            feature_dim=feature_dim
        )
        
        self.thresholds = thresholds
    
    def extract_fingerprint(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract saturation-based fingerprints from images.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Saturation fingerprints tensor of shape (N, len(thresholds))
        """
        images = self.preprocess_images(images)
        batch_size = images.shape[0]
        
        all_fingerprints = []
        
        for i in range(batch_size):
            image = images[i]  # Shape: (C, H, W)
            
            # Convert to grayscale intensity
            # Using standard RGB to grayscale conversion: 0.299*R + 0.587*G + 0.114*B
            grayscale = (0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]).cpu().numpy()
            
            # Scale to [0, 255] range
            grayscale_255 = (grayscale * 255).astype(np.uint8)
            
            # Calculate proportion of saturated pixels for each threshold
            image_fingerprint = []
            for threshold in self.thresholds:
                # Count pixels >= threshold
                saturated_pixels = np.sum(grayscale_255 >= threshold)
                # Calculate proportion
                total_pixels = grayscale_255.size
                proportion = saturated_pixels / total_pixels
                image_fingerprint.append(proportion)
            
            all_fingerprints.append(image_fingerprint)
        
        # Stack all images: (N, len(thresholds))
        return torch.tensor(np.stack(all_fingerprints), dtype=torch.float32)


class McCloskey18_Approx(AnalyticApproximation):
    """
    Differentiable approximation of McCloskey18 saturation-based fingerprint computation.
    
    This class provides a soft, differentiable approximation of the saturation detection
    for use in W2 (analytic approximation) attacks.
    """
    
    def __init__(self, original_extractor: McCloskey18, temperature: float = 0.1):
        """
        Initialize the approximation.
        
        Args:
            original_extractor: Original McCloskey18 extractor
            temperature: Temperature for soft approximations
        """
        super().__init__(original_extractor)
        self.thresholds = original_extractor.thresholds
        self.temperature = temperature
    
    def extract_fingerprint_approx(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract approximate saturation-based fingerprints using differentiable operations.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Approximate saturation fingerprints tensor of shape (N, len(thresholds))
        """
        images = self.original_extractor.preprocess_images(images)
        batch_size = images.shape[0]
        
        all_fingerprints = []
        
        for i in range(batch_size):
            image = images[i]  # Shape: (C, H, W)
            
            # Convert to grayscale intensity using differentiable operations
            # Using standard RGB to grayscale conversion: 0.299*R + 0.587*G + 0.114*B
            grayscale = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]  # Shape: (H, W)
            
            # Scale to [0, 255] range
            grayscale_255 = grayscale * 255
            
            # Calculate soft proportion of saturated pixels for each threshold
            image_fingerprint = []
            for threshold in self.thresholds:
                # Use soft thresholding with sigmoid
                # This creates a smooth transition around the threshold
                soft_saturated = torch.sigmoid((grayscale_255 - threshold) / self.temperature)
                
                # Calculate proportion (mean across all pixels)
                proportion = torch.mean(soft_saturated)
                image_fingerprint.append(proportion)
            
            # Stack thresholds: (len(thresholds),)
            image_fingerprint = torch.stack(image_fingerprint)
            all_fingerprints.append(image_fingerprint)
        
        # Stack all images: (N, len(thresholds))
        return torch.stack(all_fingerprints)
