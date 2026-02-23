"""
Nowroozi22 fingerprint extraction method implementation.

Based on "Detecting High-Quality GAN-Generated Face Images using Neural Networks"
by Nowroozi et al. (2022).

This method extends Nataraj19 by computing both intra-band (spatial) and cross-band 
(spectral) co-occurrence matrices. It computes 6 matrices total:
- 3 intra-band: Red, Green, Blue channels (like Nataraj19)
- 3 cross-band: Red-Green, Red-Blue, Green-Blue pairs (new)

The fingerprint dimension is 6 * 256 * 256 = 393,216 features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

from .base import FingerprintExtractor, AnalyticApproximation


@FingerprintExtractor.register("nowroozi22")
class Nowroozi22(FingerprintExtractor):
    """
    Nowroozi22 fingerprint extractor using cross-band co-occurrence matrices.
    
    Computes both intra-band (spatial) and cross-band (spectral) co-occurrence matrices
    to detect GAN-generated faces by leveraging color channel correlations.
    """
    
    def __init__(self, levels: int = 256, symmetric: bool = True, normed: bool = True,
                 spatial_offset: Tuple[int, int] = (0, 1), cross_offset: Tuple[int, int] = (0, 0)):
        """
        Initialize Nowroozi22 extractor.
        
        Args:
            levels: Number of gray levels (default 256 for 8-bit images)
            symmetric: Whether to make co-occurrence matrices symmetric
            normed: Whether to normalize co-occurrence matrices
            spatial_offset: Offset for spatial co-occurrence (τ in paper)
            cross_offset: Offset for cross-band co-occurrence (τ' in paper)
        """
        # Feature dimension is 6 * levels * levels (3 intra-band + 3 cross-band matrices)
        feature_dim = 6 * levels * levels
        
        super().__init__(
            method_name="nowroozi22",
            is_differentiable=False,
            has_analytic_approx=True,
            feature_dim=feature_dim
        )
        
        self.levels = levels
        self.symmetric = symmetric
        self.normed = normed
        self.spatial_offset = spatial_offset
        self.cross_offset = cross_offset
    
    def extract_fingerprint(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract cross-band co-occurrence matrices from images.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Co-occurrence matrices tensor of shape (N, 6, levels, levels)
            Order: [Red, Green, Blue, Red-Green, Red-Blue, Green-Blue]
        """
        images = self.preprocess_images(images)
        batch_size = images.shape[0]
        
        all_fingerprints = []
        
        for i in range(batch_size):
            image = images[i]  # Shape: (C, H, W)
            image_fingerprint = []
            
            # Convert to numpy for processing
            image_np = image.cpu().numpy()  # Shape: (C, H, W)
            
            # 1. Compute intra-band co-occurrence matrices (like Nataraj19)
            for c in range(3):  # RGB channels
                channel = image_np[c]  # Shape: (H, W)
                channel_int = (channel * (self.levels - 1)).astype(np.uint8)
                cooccurrence_matrix = self._compute_spatial_cooccurrence_matrix(channel_int)
                image_fingerprint.append(cooccurrence_matrix)
            
            # 2. Compute cross-band co-occurrence matrices (new)
            # Red-Green cross-band
            rg_matrix = self._compute_cross_band_cooccurrence_matrix(
                image_np[0], image_np[1])  # Red, Green
            image_fingerprint.append(rg_matrix)
            
            # Red-Blue cross-band
            rb_matrix = self._compute_cross_band_cooccurrence_matrix(
                image_np[0], image_np[2])  # Red, Blue
            image_fingerprint.append(rb_matrix)
            
            # Green-Blue cross-band
            gb_matrix = self._compute_cross_band_cooccurrence_matrix(
                image_np[1], image_np[2])  # Green, Blue
            image_fingerprint.append(gb_matrix)
            
            # Stack all 6 matrices: (6, levels, levels)
            image_fingerprint = np.stack(image_fingerprint, axis=0)
            all_fingerprints.append(image_fingerprint)
        
        # Stack all images: (N, 6, levels, levels)
        return torch.tensor(np.stack(all_fingerprints), dtype=torch.float32)
    
    def _compute_spatial_cooccurrence_matrix(self, channel: np.ndarray) -> np.ndarray:
        """
        Compute spatial co-occurrence matrix for a single channel (intra-band).
        
        Args:
            channel: Single channel image as numpy array of shape (H, W)
            
        Returns:
            Spatial co-occurrence matrix of shape (levels, levels)
        """
        height, width = channel.shape
        cooccurrence = np.zeros((self.levels, self.levels), dtype=np.float32)
        
        # Apply spatial offset
        offset_y, offset_x = self.spatial_offset
        
        # Compute co-occurrence with spatial offset
        for y in range(height):
            for x in range(width):
                # Current pixel position
                curr_y, curr_x = y, x
                # Offset pixel position
                offset_y_pos = curr_y + offset_y
                offset_x_pos = curr_x + offset_x
                
                # Check bounds
                if (0 <= offset_y_pos < height and 0 <= offset_x_pos < width):
                    i = channel[curr_y, curr_x]      # Current pixel value
                    j = channel[offset_y_pos, offset_x_pos]  # Offset pixel value
                    
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
    
    def _compute_cross_band_cooccurrence_matrix(self, channel1: np.ndarray, channel2: np.ndarray) -> np.ndarray:
        """
        Compute cross-band co-occurrence matrix between two channels.
        
        Args:
            channel1: First channel image as numpy array of shape (H, W)
            channel2: Second channel image as numpy array of shape (H, W)
            
        Returns:
            Cross-band co-occurrence matrix of shape (levels, levels)
        """
        height, width = channel1.shape
        cooccurrence = np.zeros((self.levels, self.levels), dtype=np.float32)
        
        # Convert to integer range
        channel1_int = (channel1 * (self.levels - 1)).astype(np.uint8)
        channel2_int = (channel2 * (self.levels - 1)).astype(np.uint8)
        
        # Apply cross-band offset
        offset_y, offset_x = self.cross_offset
        
        # Compute cross-band co-occurrence
        for y in range(height):
            for x in range(width):
                # Current pixel position
                curr_y, curr_x = y, x
                # Offset pixel position
                offset_y_pos = curr_y + offset_y
                offset_x_pos = curr_x + offset_x
                
                # Check bounds
                if (0 <= offset_y_pos < height and 0 <= offset_x_pos < width):
                    i = channel1_int[curr_y, curr_x]      # Channel1 pixel value
                    j = channel2_int[offset_y_pos, offset_x_pos]  # Channel2 pixel value
                    
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


class Nowroozi22_Approx(AnalyticApproximation):
    """
    Differentiable approximation of Nowroozi22 cross-band co-occurrence matrix computation.
    
    This class provides a soft, differentiable approximation of both spatial and cross-band
    co-occurrence matrix computation for use in W2 (analytic approximation) attacks.
    """
    
    def __init__(self, original_extractor: Nowroozi22, temperature: float = 0.1, 
                 reduced_dim: bool = True, feature_reduction_factor: int = 4):
        """
        Initialize the approximation.
        
        Args:
            original_extractor: Original Nowroozi22 extractor
            temperature: Temperature for soft approximations
            reduced_dim: Whether to use reduced feature dimension for memory efficiency
            feature_reduction_factor: Factor to reduce feature dimension (e.g., 4 means 64x64 instead of 256x256)
        """
        super().__init__(original_extractor)
        self.levels = original_extractor.levels
        self.symmetric = original_extractor.symmetric
        self.normed = original_extractor.normed
        self.temperature = temperature
        self.spatial_offset = original_extractor.spatial_offset
        self.cross_offset = original_extractor.cross_offset
        self.reduced_dim = reduced_dim
        self.feature_reduction_factor = feature_reduction_factor
        
        # Calculate effective levels for reduced dimension
        if self.reduced_dim:
            self.effective_levels = self.levels // self.feature_reduction_factor
            self.effective_feature_dim = 6 * self.effective_levels * self.effective_levels
        else:
            self.effective_levels = self.levels
            self.effective_feature_dim = 6 * self.levels * self.levels
    
    def extract_fingerprint_approx(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract approximate cross-band co-occurrence matrices using differentiable operations.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Approximate co-occurrence matrices tensor of shape (N, 6, effective_levels, effective_levels)
        """
        images = self.original_extractor.preprocess_images(images)
        batch_size, channels, height, width = images.shape
        
        all_fingerprints = []
        
        for i in range(batch_size):
            image = images[i]  # Shape: (C, H, W)
            image_fingerprint = []
            
            # 1. Compute approximate intra-band co-occurrence matrices
            for c in range(3):  # RGB channels
                channel = image[c]  # Shape: (H, W)
                cooccurrence_matrix = self._compute_soft_spatial_cooccurrence_matrix(channel)
                image_fingerprint.append(cooccurrence_matrix)
            
            # 2. Compute approximate cross-band co-occurrence matrices
            # Red-Green cross-band
            rg_matrix = self._compute_soft_cross_band_cooccurrence_matrix(
                image[0], image[1])  # Red, Green
            image_fingerprint.append(rg_matrix)
            
            # Red-Blue cross-band
            rb_matrix = self._compute_soft_cross_band_cooccurrence_matrix(
                image[0], image[2])  # Red, Blue
            image_fingerprint.append(rb_matrix)
            
            # Green-Blue cross-band
            gb_matrix = self._compute_soft_cross_band_cooccurrence_matrix(
                image[1], image[2])  # Green, Blue
            image_fingerprint.append(gb_matrix)
            
            # Stack all 6 matrices: (6, effective_levels, effective_levels)
            image_fingerprint = torch.stack(image_fingerprint, dim=0)
            all_fingerprints.append(image_fingerprint)
        
        # Stack all images: (N, 6, effective_levels, effective_levels)
        return torch.stack(all_fingerprints)
    
    def _compute_soft_spatial_cooccurrence_matrix(self, channel: torch.Tensor) -> torch.Tensor:
        """
        Compute soft approximation of spatial co-occurrence matrix for a single channel.
        
        Args:
            channel: Single channel image tensor of shape (H, W)
            
        Returns:
            Soft spatial co-occurrence matrix of shape (effective_levels, effective_levels)
        """
        # Scale channel values to [0, effective_levels-1]
        channel_scaled = channel * (self.effective_levels - 1)
        
        # Create soft quantization matrix
        pixel_values = channel_scaled.reshape(-1, 1)  # Shape: (H*W, 1)
        level_values = torch.arange(self.effective_levels, device=channel.device).view(1, -1)  # Shape: (1, effective_levels)
        soft_assignments = torch.exp(-((pixel_values - level_values) ** 2) / (2 * self.temperature ** 2))
        soft_assignments = soft_assignments / soft_assignments.sum(dim=1, keepdim=True)  # Normalize
        
        # Apply spatial offset to get offset pixel values
        offset_y, offset_x = self.spatial_offset
        if offset_y != 0 or offset_x != 0:
            # Pad the image to handle offset
            pad_y = abs(offset_y)
            pad_x = abs(offset_x)
            # Use 'replicate' mode instead of 'reflect' for better compatibility
            channel_padded = F.pad(channel_scaled.unsqueeze(0).unsqueeze(0), 
                                 (pad_x, pad_x, pad_y, pad_y), mode='replicate')
            channel_padded = channel_padded.squeeze(0).squeeze(0)
            
            # Extract offset pixels
            if offset_y >= 0:
                start_y = offset_y
                end_y = start_y + channel_scaled.shape[0]
            else:
                start_y = 0
                end_y = channel_scaled.shape[0]
            
            if offset_x >= 0:
                start_x = offset_x
                end_x = start_x + channel_scaled.shape[1]
            else:
                start_x = 0
                end_x = channel_scaled.shape[1]
            
            offset_pixels = channel_padded[start_y:end_y, start_x:end_x]
        else:
            offset_pixels = channel_scaled
        
        # Create soft assignments for offset pixels
        offset_pixel_values = offset_pixels.reshape(-1, 1)  # Shape: (H*W, 1)
        offset_soft_assignments = torch.exp(-((offset_pixel_values - level_values) ** 2) / (2 * self.temperature ** 2))
        offset_soft_assignments = offset_soft_assignments / offset_soft_assignments.sum(dim=1, keepdim=True)  # Normalize
        
        # Compute soft co-occurrence matrix more efficiently
        # Instead of creating (H*W, levels, levels) tensor, compute directly using matrix multiplication
        # cooccurrence_matrix = sum over all pixels of outer product of soft assignments
        cooccurrence_matrix = torch.zeros(self.effective_levels, self.effective_levels, 
                                        device=channel.device, dtype=channel.dtype)
        
        # Process in chunks to avoid memory issues
        # Use smaller chunks for larger images to prevent memory issues
        if soft_assignments.shape[0] > 50000:  # For very large images (e.g., 256x256)
            if self.effective_levels >= 256:  # Full dimension
                chunk_size = 100  # Very small chunks for full dimension
            else:  # Reduced dimension
                chunk_size = 500  # Smaller chunks for large images
        else:
            chunk_size = min(1000, soft_assignments.shape[0])  # Process 1000 pixels at a time
        
        for i in range(0, soft_assignments.shape[0], chunk_size):
            end_idx = min(i + chunk_size, soft_assignments.shape[0])
            chunk_assignments = soft_assignments[i:end_idx]  # Shape: (chunk_size, levels)
            chunk_offset_assignments = offset_soft_assignments[i:end_idx]  # Shape: (chunk_size, levels)
            
            # Compute outer product for this chunk: (chunk_size, levels, levels)
            chunk_cooccurrence = torch.einsum('ni,nj->nij', chunk_assignments, chunk_offset_assignments)
            
            # Sum over the chunk dimension
            cooccurrence_matrix += chunk_cooccurrence.sum(dim=0)  # Shape: (levels, levels)
            
            # Clean up chunk memory
            del chunk_assignments, chunk_offset_assignments, chunk_cooccurrence
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        # Make symmetric if requested
        if self.symmetric:
            cooccurrence_matrix = cooccurrence_matrix + cooccurrence_matrix.t()
        
        # Normalize if requested
        if self.normed:
            total = cooccurrence_matrix.sum()
            if total > 0:
                cooccurrence_matrix = cooccurrence_matrix / total
        
        return cooccurrence_matrix
    
    def _compute_soft_cross_band_cooccurrence_matrix(self, channel1: torch.Tensor, channel2: torch.Tensor) -> torch.Tensor:
        """
        Compute soft approximation of cross-band co-occurrence matrix between two channels.
        
        Args:
            channel1: First channel image tensor of shape (H, W)
            channel2: Second channel image tensor of shape (H, W)
            
        Returns:
            Soft cross-band co-occurrence matrix of shape (effective_levels, effective_levels)
        """
        # Scale channel values to [0, effective_levels-1]
        channel1_scaled = channel1 * (self.effective_levels - 1)
        channel2_scaled = channel2 * (self.effective_levels - 1)
        
        # Create soft quantization matrices for both channels
        pixel_values_1 = channel1_scaled.reshape(-1, 1)  # Shape: (H*W, 1)
        pixel_values_2 = channel2_scaled.reshape(-1, 1)  # Shape: (H*W, 1)
        level_values = torch.arange(self.effective_levels, device=channel1.device).view(1, -1)  # Shape: (1, effective_levels)
        
        # Soft assignments for both channels
        soft_assignments_1 = torch.exp(-((pixel_values_1 - level_values) ** 2) / (2 * self.temperature ** 2))
        soft_assignments_1 = soft_assignments_1 / soft_assignments_1.sum(dim=1, keepdim=True)
        
        soft_assignments_2 = torch.exp(-((pixel_values_2 - level_values) ** 2) / (2 * self.temperature ** 2))
        soft_assignments_2 = soft_assignments_2 / soft_assignments_2.sum(dim=1, keepdim=True)
        
        # Apply cross-band offset to channel2
        offset_y, offset_x = self.cross_offset
        if offset_y != 0 or offset_x != 0:
            # Pad the image to handle offset
            pad_y = abs(offset_y)
            pad_x = abs(offset_x)
            # Use 'replicate' mode instead of 'reflect' for better compatibility
            channel2_padded = F.pad(channel2_scaled.unsqueeze(0).unsqueeze(0), 
                                  (pad_x, pad_x, pad_y, pad_y), mode='replicate')
            channel2_padded = channel2_padded.squeeze(0).squeeze(0)
            
            # Extract offset pixels
            if offset_y >= 0:
                start_y = offset_y
                end_y = start_y + channel2_scaled.shape[0]
            else:
                start_y = 0
                end_y = channel2_scaled.shape[0]
            
            if offset_x >= 0:
                start_x = offset_x
                end_x = start_x + channel2_scaled.shape[1]
            else:
                start_x = 0
                end_x = channel2_scaled.shape[1]
            
            offset_pixels_2 = channel2_padded[start_y:end_y, start_x:end_x]
        else:
            offset_pixels_2 = channel2_scaled
        
        # Create soft assignments for offset pixels in channel2
        offset_pixel_values_2 = offset_pixels_2.reshape(-1, 1)  # Shape: (H*W, 1)
        offset_soft_assignments_2 = torch.exp(-((offset_pixel_values_2 - level_values) ** 2) / (2 * self.temperature ** 2))
        offset_soft_assignments_2 = offset_soft_assignments_2 / offset_soft_assignments_2.sum(dim=1, keepdim=True)  # Normalize
        
        # Compute soft cross-band co-occurrence matrix more efficiently
        # Instead of creating (H*W, levels, levels) tensor, compute directly using matrix multiplication
        # cooccurrence_matrix = sum over all pixels of outer product of soft assignments
        cooccurrence_matrix = torch.zeros(self.effective_levels, self.effective_levels, 
                                        device=channel1.device, dtype=channel1.dtype)
        
        # Process in chunks to avoid memory issues
        # Use smaller chunks for larger images to prevent memory issues
        if soft_assignments_1.shape[0] > 50000:  # For very large images (e.g., 256x256)
            if self.effective_levels >= 256:  # Full dimension
                chunk_size = 100  # Very small chunks for full dimension
            else:  # Reduced dimension
                chunk_size = 500  # Smaller chunks for large images
        else:
            chunk_size = min(1000, soft_assignments_1.shape[0])  # Process 1000 pixels at a time
        
        for i in range(0, soft_assignments_1.shape[0], chunk_size):
            end_idx = min(i + chunk_size, soft_assignments_1.shape[0])
            chunk_assignments_1 = soft_assignments_1[i:end_idx]  # Shape: (chunk_size, levels)
            chunk_assignments_2 = offset_soft_assignments_2[i:end_idx]  # Shape: (chunk_size, levels)
            
            # Compute outer product for this chunk: (chunk_size, levels, levels)
            chunk_cooccurrence = torch.einsum('ni,nj->nij', chunk_assignments_1, chunk_assignments_2)
            
            # Sum over the chunk dimension
            cooccurrence_matrix += chunk_cooccurrence.sum(dim=0)  # Shape: (levels, levels)
            
            # Clean up chunk memory
            del chunk_assignments_1, chunk_assignments_2, chunk_cooccurrence
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        # Make symmetric if requested
        if self.symmetric:
            cooccurrence_matrix = cooccurrence_matrix + cooccurrence_matrix.t()
        
        # Normalize if requested
        if self.normed:
            total = cooccurrence_matrix.sum()
            if total > 0:
                cooccurrence_matrix = cooccurrence_matrix / total
        
        return cooccurrence_matrix
