"""
Giudice21 fingerprint extraction method implementation.

Based on "Fighting Deepfakes by Detecting GAN DCT Anomalies" by Giudice et al. (2021).

This method targets GAN-specific frequency artifacts as fingerprints in the DCT domain.
It observes that GAN-generated images exhibit anomalous patterns in their DCT coefficients,
especially in the high-frequency AC components, that deviate from real images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .base import FingerprintExtractor, AnalyticApproximation


@FingerprintExtractor.register("giudice21")
class Giudice21(FingerprintExtractor, nn.Module):
    """
    Giudice21 fingerprint extraction using DCT frequency domain analysis.
    
    This method extracts fingerprints by analyzing DCT coefficients in 8x8 blocks.
    It computes β_c = mean(|DCT_c|) for each AC coefficient (c = 1..63) to create
    a 63-dimensional fingerprint vector that captures GAN-specific frequency artifacts.
    
    This is an explicit fingerprint method that is fully differentiable.
    """
    
    def __init__(self, img_size: int = 256, device: str = None):
        """
        Initialize Giudice21 fingerprint extractor.
        
        Args:
            img_size: Input image size (assumed square)
            device: Device to run on ("cuda", "cpu", "mps", or None for auto)
        """
        # Initialize nn.Module first
        nn.Module.__init__(self)
        
        # Giudice21 extracts 63 AC DCT coefficients as features
        feature_dim = 63
        
        FingerprintExtractor.__init__(
            self,
            method_name="giudice21",
            is_differentiable=True,  # Fully differentiable DCT operations
            has_analytic_approx=False,  # No approximation needed as it's already differentiable
            feature_dim=feature_dim,
            is_implicit_fingerprint=False  # This is an explicit fingerprint method
        )
        
        self.img_size = img_size
        
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
        # Precomputed zigzag lookup table: (u,v) -> zigzag index c ∈ [0,63]
        self.register_buffer('zigzag_idx', torch.tensor([
            [0, 1, 5, 6,14,15,27,28],
            [2, 4, 7,13,16,26,29,42],
            [3, 8,12,17,25,30,41,43],
            [9,11,18,24,31,40,44,53],
            [10,19,23,32,39,45,52,54],
            [20,22,33,38,46,51,55,60],
            [21,34,37,47,50,56,59,61],
            [35,36,48,49,57,58,62,63],
        ], dtype=torch.long))
        
        # Precompute 8x8 DCT matrix
        self.register_buffer('dct_matrix', self._create_dct_matrix(8))
        
        # Move the model to the specified device
        self.to(self.device)
    
    def _create_dct_matrix(self, N: int) -> torch.Tensor:
        """
        Create DCT transform matrix of shape (N,N).
        
        Args:
            N: Size of the DCT matrix
            
        Returns:
            DCT matrix of shape (N, N)
        """
        k = torch.arange(N, dtype=torch.float32).unsqueeze(1)  # (N,1)
        n = torch.arange(N, dtype=torch.float32).unsqueeze(0)  # (1,N)
        alpha = torch.ones(N, dtype=torch.float32)
        alpha[0] = 1 / math.sqrt(2)
        D = alpha[:, None] * torch.cos(math.pi / N * (n + 0.5) * k)
        return D * math.sqrt(2 / N)
    
    def extract_fingerprint(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract Giudice21 fingerprints from images.
        
        Args:
            images: Input images tensor of shape (N, C, H, W) with values in [0,1]
            
        Returns:
            Fingerprint features tensor of shape (N, 63)
        """
        batch_size = images.shape[0]
        
        # Ensure images are in correct range [0, 1]
        images = self.preprocess_images(images)
        
        # Move images to the correct device
        images = images.to(self.device)
        
        # Convert to luminance (Y) using standard JPEG weights
        r, g, b = images[:, 0], images[:, 1], images[:, 2]
        Y = 0.299 * r + 0.587 * g + 0.114 * b  # shape [N, H, W]
        
        # Ensure image size is compatible with 8x8 blocks
        if Y.shape[1] != self.img_size or Y.shape[2] != self.img_size:
            Y = F.interpolate(Y.unsqueeze(1), size=(self.img_size, self.img_size), 
                            mode='bilinear', align_corners=False).squeeze(1)
        
        # Divide into non-overlapping 8x8 blocks
        blocks = Y.unfold(1, 8, 8).unfold(2, 8, 8)  # shape [N, H//8, W//8, 8, 8]
        blocks = blocks.contiguous().view(batch_size, -1, 8, 8)  # shape [N, num_blocks, 8, 8]
        
        # Apply 2D DCT to each block: D * block * D^T
        D = self.dct_matrix
        D_t = D.transpose(0, 1)
        
        # Reshape for batch matrix multiplication
        blocks_reshaped = blocks.view(-1, 8, 8)  # shape [N*num_blocks, 8, 8]
        dct_blocks = D @ blocks_reshaped @ D_t  # shape [N*num_blocks, 8, 8]
        dct_blocks = dct_blocks.view(batch_size, -1, 8, 8)  # shape [N, num_blocks, 8, 8]
        
        # Compute β_c = mean(|coeff_c|) for c = 1..63 (AC only)
        beta = torch.zeros(batch_size, 63, device=self.device, dtype=images.dtype)
        
        for u in range(8):
            for v in range(8):
                c = self.zigzag_idx[u, v].item()
                if c == 0:
                    continue  # skip DC coefficient
                coeffs = dct_blocks[:, :, u, v]  # shape [N, num_blocks]
                beta[:, c - 1] = coeffs.abs().mean(dim=1)  # mean across blocks
        
        return beta  # shape [N, 63]
    
    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images before fingerprint extraction.
        
        Args:
            images: Input images tensor
            
        Returns:
            Preprocessed images tensor in [0, 1] range
        """
        # Ensure images are in [0, 1] range
        if images.min() < 0:
            # Convert from [-1, 1] to [0, 1]
            images = (images + 1) / 2
        
        # Clamp to [0, 1]
        images = torch.clamp(images, 0, 1)
        
        return images


class Giudice21_Approx(AnalyticApproximation):
    """
    Analytic approximation for Giudice21 fingerprint extraction.
    
    Since Giudice21 is already differentiable, this approximation is not needed
    for W1 attacks (which can be done directly). This class is kept for compatibility
    with the framework.
    """
    
    def __init__(self, original_extractor: Giudice21):
        """
        Initialize Giudice21 approximation.
        
        Args:
            original_extractor: Original Giudice21 extractor
        """
        super().__init__(original_extractor)
    
    def extract_fingerprint_approx(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract approximate Giudice21 fingerprints.
        
        Since Giudice21 is already differentiable, this just calls the original method.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Fingerprint features tensor of shape (N, 63)
        """
        # Since Giudice21 is already differentiable, return the original result
        return self.original_extractor.extract_fingerprint(images)
