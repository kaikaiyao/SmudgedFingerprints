"""
Nataraj19 fingerprint extraction method implementation.

Based on "Detecting GAN generated Fake Images using Co-occurrence Matrices"
by Nataraj et al. (2019).

This method computes co-occurrence matrices directly on image pixels for each RGB channel,
producing 3x256x256 tensors that serve as fingerprints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

from .base import FingerprintExtractor, AnalyticApproximation


@FingerprintExtractor.register("nataraj19")
class Nataraj19(FingerprintExtractor):
    """
    Nataraj19 fingerprint extractor using co-occurrence matrices.
    
    Computes co-occurrence matrices directly on image pixels for each RGB channel,
    producing 3x256x256 tensors as described in the original paper.
    """
    
    def __init__(self, levels: int = 256, symmetric: bool = True, normed: bool = True, device: str = "cpu"):
        """
        Initialize Nataraj19 extractor.
        
        Args:
            levels: Number of gray levels (default 256 for 8-bit images)
            symmetric: Whether to make co-occurrence matrices symmetric
            normed: Whether to normalize co-occurrence matrices
            device: Device to use for computations (default "cpu")
        """
        # Feature dimension is 3 * levels * levels (3 RGB channels, each with levels x levels matrix)
        feature_dim = 3 * levels * levels
        
        super().__init__(
            method_name="nataraj19",
            is_differentiable=False,
            has_analytic_approx=True,
            feature_dim=feature_dim
        )
        
        self.levels = levels
        self.symmetric = symmetric
        self.normed = normed
        self.device = device
    
    def extract_fingerprint(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract co-occurrence matrices from images.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Co-occurrence matrices tensor of shape (N, 3, levels, levels)
        """
        images = self.preprocess_images(images)
        batch_size = images.shape[0]
        
        all_fingerprints = []
        
        for i in range(batch_size):
            image = images[i]  # Shape: (C, H, W)
            image_fingerprint = []
            
            # Process each RGB channel separately
            for c in range(3):  # RGB channels
                channel = image[c].cpu().numpy()  # Shape: (H, W)
                
                # Convert to appropriate integer range for co-occurrence matrix
                channel_int = (channel * (self.levels - 1)).astype(np.uint8)
                
                # Compute co-occurrence matrix for this channel
                # Using horizontal pixel pairs (distance=1, angle=0)
                cooccurrence_matrix = self._compute_cooccurrence_matrix(channel_int)
                
                image_fingerprint.append(cooccurrence_matrix)
            
            # Stack the 3 channels: (3, levels, levels)
            image_fingerprint = np.stack(image_fingerprint, axis=0)
            all_fingerprints.append(image_fingerprint)
        
        # Stack all images: (N, 3, levels, levels)
        return torch.tensor(np.stack(all_fingerprints), dtype=torch.float32)
    
    def _compute_cooccurrence_matrix(self, channel: np.ndarray) -> np.ndarray:
        """
        Compute co-occurrence matrix for a single channel.
        
        Args:
            channel: Single channel image as numpy array of shape (H, W)
            
        Returns:
            Co-occurrence matrix of shape (levels, levels)
        """
        height, width = channel.shape
        cooccurrence = np.zeros((self.levels, self.levels), dtype=np.float32)
        
        # Compute horizontal pixel pairs (distance=1, angle=0)
        # For each pixel, look at its right neighbor
        for y in range(height):
            for x in range(width - 1):
                i = channel[y, x]      # Current pixel value
                j = channel[y, x + 1]  # Right neighbor pixel value
                
                cooccurrence[i, j] += 1
                
                # If symmetric, also count the reverse pair
                if self.symmetric:
                    cooccurrence[j, i] += 1
        
        # Normalize if requested
        if self.normed:
            total = np.sum(cooccurrence)
            if total > 0:
                cooccurrence = cooccurrence / total
        
        return cooccurrence


class Nataraj19_Approx(AnalyticApproximation):
    """
    Differentiable approximation of Nataraj19 co-occurrence matrix computation.
    
    This class provides a soft, differentiable approximation of the co-occurrence matrix
    computation for use in W2 (analytic approximation) attacks.
    """
    
    def __init__(self, original_extractor: Nataraj19, temperature: float = 0.1):
        """
        Initialize the approximation.
        
        Args:
            original_extractor: Original Nataraj19 extractor
            temperature: Temperature for soft approximations
        """
        super().__init__(original_extractor)
        self.levels = original_extractor.levels
        self.symmetric = original_extractor.symmetric
        self.normed = original_extractor.normed
        self.temperature = temperature
    
    def extract_fingerprint_approx(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract approximate co-occurrence matrices using differentiable operations.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Approximate co-occurrence matrices tensor of shape (N, 3, levels, levels)
        """
        images = self.original_extractor.preprocess_images(images)
        batch_size, channels, height, width = images.shape
        
        all_fingerprints = []
        
        for i in range(batch_size):
            image = images[i]  # Shape: (C, H, W)
            image_fingerprint = []
            
            # Process each RGB channel
            for c in range(3):  # RGB channels
                channel = image[c]  # Shape: (H, W)
                
                # Compute approximate co-occurrence matrix for this channel
                cooccurrence_matrix = self._compute_soft_cooccurrence_matrix(channel)
                image_fingerprint.append(cooccurrence_matrix)
            
            # Stack the 3 channels: (3, levels, levels)
            image_fingerprint = torch.stack(image_fingerprint, dim=0)
            all_fingerprints.append(image_fingerprint)
        
        # Stack all images: (N, 3, levels, levels)
        return torch.stack(all_fingerprints)
    
    def _compute_soft_cooccurrence_matrix(self, channel: torch.Tensor) -> torch.Tensor:
        """
        Compute soft approximation of co-occurrence matrix for a single channel.
        
        Args:
            channel: Single channel image tensor of shape (H, W)
            
        Returns:
            Soft co-occurrence matrix of shape (levels, levels)
        """
        # Scale channel values to [0, levels-1]
        channel_scaled = channel * (self.levels - 1)
        
        # Create soft quantization matrix
        # Shape: (H*W, levels)
        pixel_values = channel_scaled.view(-1, 1)  # Shape: (H*W, 1)
        level_values = torch.arange(self.levels, device=channel.device).view(1, -1)  # Shape: (1, levels)
        soft_assignments = torch.exp(-((pixel_values - level_values) ** 2) / (2 * self.temperature ** 2))
        soft_assignments = soft_assignments / soft_assignments.sum(dim=1, keepdim=True)  # Normalize
        
        # Get pairs of adjacent pixels
        pixels_left = channel_scaled[:, :-1].contiguous().view(-1, 1)   # Shape: (H*(W-1), 1)
        pixels_right = channel_scaled[:, 1:].contiguous().view(-1, 1)   # Shape: (H*(W-1), 1)
        
        # Compute soft assignments for both pixels
        soft_assign_left = torch.exp(-((pixels_left - level_values) ** 2) / (2 * self.temperature ** 2))
        soft_assign_left = soft_assign_left / soft_assign_left.sum(dim=1, keepdim=True)
        
        soft_assign_right = torch.exp(-((pixels_right - level_values) ** 2) / (2 * self.temperature ** 2))
        soft_assign_right = soft_assign_right / soft_assign_right.sum(dim=1, keepdim=True)
        
        # Compute co-occurrence matrix using batch matrix multiplication
        cooccurrence = torch.matmul(soft_assign_left.t(), soft_assign_right)
        
        # Add symmetric counterpart if needed
        if self.symmetric:
            cooccurrence = (cooccurrence + cooccurrence.t()) / 2
        
        # Normalize if requested
        if self.normed:
            total = torch.sum(cooccurrence)
            if total > 0:
                cooccurrence = cooccurrence / total
        
        return cooccurrence
    
    def _soft_quantize(self, channel: torch.Tensor, levels: int) -> torch.Tensor:
        """
        Soft quantization of channel values.
        
        Args:
            channel: Input channel tensor
            levels: Number of quantization levels
            
        Returns:
            Soft-quantized channel tensor
        """
        # Scale to [0, levels-1]
        scaled = channel * (levels - 1)
        
        # Soft quantization using temperature
        floor_vals = torch.floor(scaled)
        ceil_vals = torch.ceil(scaled)
        
        # Soft assignment weights
        weights = torch.sigmoid((scaled - floor_vals - 0.5) / self.temperature)
        
        return floor_vals * (1 - weights) + ceil_vals * weights
    
    def _soft_assign_to_matrix(self, matrix: torch.Tensor, i: torch.Tensor, j: torch.Tensor):
        """
        Soft assignment to co-occurrence matrix.
        
        Args:
            matrix: Co-occurrence matrix to update
            i: First pixel value (row index)
            j: Second pixel value (column index)
        """
        # Convert to integer indices for matrix access
        i_idx = torch.clamp(torch.round(i), 0, self.levels - 1).long()
        j_idx = torch.clamp(torch.round(j), 0, self.levels - 1).long()
        
        # Add contribution to the matrix
        matrix[i_idx, j_idx] += 1.0