"""
Marra19a fingerprint extraction method implementation.

Based on "Do GANs Leave Artificial Fingerprints?" by Marra et al. (2019).

This method extracts fingerprints by computing residuals between original images
and their denoised versions using either BM3D (original) or Gaussian blur (differentiable).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import cv2

from .base import FingerprintExtractor, AnalyticApproximation


def bm3d_denoise(image: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """
    Apply BM3D-style denoising to an image.
    
    Note: This is an approximation of BM3D using OpenCV's non-local means denoising.
    For the exact BM3D algorithm, you would need to install the bm3d package.
    
    Args:
        image: Input image as numpy array with values in [0, 1]
        sigma: Noise standard deviation parameter
        
    Returns:
        Denoised image as numpy array with values in [0, 1]
    """
    # Ensure image is in uint8 format for OpenCV
    if image.dtype != np.uint8:
        image_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    else:
        image_uint8 = image
    
    # Apply non-local means denoising (approximation of BM3D)
    # Parameters: h=sigma (filter strength), templateWindowSize=7, searchWindowSize=21
    denoised = cv2.fastNlMeansDenoisingColored(image_uint8, None, sigma, sigma, 7, 21)
    
    return denoised.astype(np.float32) / 255.0


class GaussianBlur2D(nn.Module):
    """
    2D Gaussian blur layer for differentiable approximation.
    """
    
    def __init__(self, kernel_size: int = 15, sigma: float = 2.0, channels: int = 3):
        """
        Initialize Gaussian blur layer.
        
        Args:
            kernel_size: Size of the Gaussian kernel
            sigma: Standard deviation of the Gaussian
            channels: Number of input channels
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels
        
        # Create Gaussian kernel
        self.kernel = self._create_gaussian_kernel()
        
    def _create_gaussian_kernel(self) -> torch.Tensor:
        """Create 2D Gaussian kernel."""
        # Create coordinate grid
        x = torch.arange(-self.kernel_size // 2, self.kernel_size // 2 + 1, dtype=torch.float32)
        y = torch.arange(-self.kernel_size // 2, self.kernel_size // 2 + 1, dtype=torch.float32)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Create Gaussian kernel
        kernel = torch.exp(-(X**2 + Y**2) / (2 * self.sigma**2))
        kernel = kernel / kernel.sum()  # Normalize
        
        # Reshape for convolution: (out_channels, in_channels, height, width)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        kernel = kernel.repeat(self.channels, 1, 1, 1)  # (C, 1, H, W)
        
        return kernel
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian blur to input tensor.
        
        Args:
            x: Input tensor of shape (N, C, H, W)
            
        Returns:
            Blurred tensor of same shape
        """
        # Move kernel to same device as input
        kernel = self.kernel.to(x.device)
        
        # Apply convolution with padding to maintain spatial dimensions
        # For odd kernel sizes, padding should be kernel_size // 2
        # For even kernel sizes, we need to handle differently
        if self.kernel_size % 2 == 1:
            # Odd kernel size
            padding = self.kernel_size // 2
        else:
            # Even kernel size - use asymmetric padding
            padding = (self.kernel_size // 2 - 1, self.kernel_size // 2,
                      self.kernel_size // 2 - 1, self.kernel_size // 2)
        
        blurred = F.conv2d(x, kernel, padding=padding, groups=self.channels)
        
        # Ensure output has exactly the same spatial dimensions as input
        if blurred.shape[2:] != x.shape[2:]:
            # If there's still a mismatch, use interpolation to resize
            blurred = F.interpolate(blurred, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return blurred


@FingerprintExtractor.register("marra19a")
class Marra19a(FingerprintExtractor):
    """
    Marra19a fingerprint extractor using residual-based fingerprinting.
    
    Extracts fingerprints by computing residuals between original images
    and their denoised versions. Uses non-local means denoising as an
    approximation of BM3D for the original implementation.
    """
    
    def __init__(self, sigma: float = 25.0, image_size: int = 256):
        """
        Initialize Marra19a extractor.
        
        Args:
            sigma: Noise standard deviation for BM3D denoising
            image_size: Size of input images (assumed square)
        """
        super().__init__(
            method_name="marra19a",
            is_differentiable=False,
            has_analytic_approx=True,
            feature_dim=3 * image_size * image_size  # Full image dimensions as fingerprint
        )
        
        self.sigma = sigma
        self.image_size = image_size
    
    def extract_fingerprint(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract fingerprints by computing residuals between original and denoised images.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Residual fingerprints tensor of shape (N, C*H*W)
        """
        images = self.preprocess_images(images)
        batch_size = images.shape[0]
        
        all_fingerprints = []
        
        for i in range(batch_size):
            image = images[i]  # Shape: (C, H, W)
            
            # Convert to numpy for BM3D processing
            image_np = image.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            
            # Apply BM3D denoising
            denoised_np = bm3d_denoise(image_np, self.sigma)
            
            # Convert back to torch tensor
            denoised = torch.from_numpy(denoised_np).permute(2, 0, 1)  # (C, H, W)
            denoised = denoised.to(image.device)
            
            # Compute residual (fingerprint)
            residual = image - denoised  # Shape: (C, H, W)
            
            # Flatten to 1D feature vector
            fingerprint = residual.flatten()  # Shape: (C*H*W,)
            
            all_fingerprints.append(fingerprint)
        
        return torch.stack(all_fingerprints)


class Marra19a_Approx(AnalyticApproximation):
    """
    Differentiable approximation of Marra19a using Gaussian blur instead of BM3D.
    """
    
    def __init__(self, original_extractor: Marra19a, kernel_size: int = 15, sigma: float = 2.0):
        """
        Initialize the approximation.
        
        Args:
            original_extractor: Original Marra19a extractor
            kernel_size: Size of Gaussian blur kernel
            sigma: Standard deviation for Gaussian blur
        """
        super().__init__(original_extractor)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_blur = GaussianBlur2D(kernel_size=kernel_size, sigma=sigma, channels=3)
        self.image_size = original_extractor.image_size
    
    def extract_fingerprint_approx(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract approximate fingerprints using Gaussian blur denoising.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Approximate residual fingerprints tensor of shape (N, C*H*W)
        """
        images = self.original_extractor.preprocess_images(images)
        
        # Apply Gaussian blur denoising
        denoised = self.gaussian_blur(images)
        

        
        # Compute residual (fingerprint)
        residuals = images - denoised  # Shape: (N, C, H, W)
        
        # Flatten to 2D feature matrix
        fingerprints = residuals.view(images.shape[0], -1)  # Shape: (N, C*H*W)
        
        return fingerprints


class MultiScaleGaussianBlur2D(nn.Module):
    """
    Multi-scale Gaussian blur layer for enhanced denoising.
    """
    
    def __init__(self, kernel_sizes: list = [7, 11, 15, 19], sigmas: list = [1.0, 2.0, 3.0, 4.0], channels: int = 3):
        """
        Initialize multi-scale Gaussian blur layer.
        
        Args:
            kernel_sizes: List of kernel sizes for different scales
            sigmas: List of standard deviations for different scales
            channels: Number of input channels
        """
        super().__init__()
        self.num_scales = len(kernel_sizes)
        assert len(kernel_sizes) == len(sigmas), "Number of kernel sizes must match number of sigmas"
        
        self.blur_layers = nn.ModuleList([
            GaussianBlur2D(kernel_size=ks, sigma=s, channels=channels)
            for ks, s in zip(kernel_sizes, sigmas)
        ])
        
        # Learnable weights for combining different scales
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale Gaussian blur to input tensor.
        
        Args:
            x: Input tensor of shape (N, C, H, W)
            
        Returns:
            Multi-scale blurred tensor of same shape
        """
        # Apply softmax to ensure weights sum to 1
        weights = F.softmax(self.scale_weights, dim=0)
        
        # Apply blur at different scales
        blurred_outputs = []
        for blur_layer in self.blur_layers:
            blurred_outputs.append(blur_layer(x))
        
        # Weighted combination of different scales
        result = torch.zeros_like(x)
        for i, blurred in enumerate(blurred_outputs):
            result += weights[i] * blurred
            
        return result


class FrequencyEnhancer(nn.Module):
    """
    Frequency domain enhancement module inspired by Giudice21.
    """
    
    def __init__(self, img_size: int = 256):
        super().__init__()
        self.img_size = img_size
        
        # Create DCT matrix for 8x8 blocks
        self.register_buffer('dct_matrix', self._create_dct_matrix(8))
        
        # Frequency band weights (learnable)
        self.freq_weights = nn.Parameter(torch.ones(64))  # 8x8 = 64 coefficients
    
    def _create_dct_matrix(self, N: int) -> torch.Tensor:
        """Create DCT transform matrix."""
        import math
        k = torch.arange(N, dtype=torch.float32).unsqueeze(1)
        n = torch.arange(N, dtype=torch.float32).unsqueeze(0)
        alpha = torch.ones(N, dtype=torch.float32)
        alpha[0] = 1 / math.sqrt(2)
        D = alpha[:, None] * torch.cos(math.pi / N * (n + 0.5) * k)
        return D * math.sqrt(2 / N)
    
    def forward(self, residuals: torch.Tensor) -> torch.Tensor:
        """
        Enhance residuals using frequency domain analysis.
        
        Args:
            residuals: Input residuals tensor of shape (N, C, H, W)
            
        Returns:
            Enhanced residuals tensor of same shape
        """
        batch_size, channels, height, width = residuals.shape
        
        # Convert to grayscale for frequency analysis
        if channels == 3:
            gray = 0.299 * residuals[:, 0] + 0.587 * residuals[:, 1] + 0.114 * residuals[:, 2]
        else:
            gray = residuals.mean(dim=1)
        
        # Ensure divisible by 8
        if height % 8 != 0 or width % 8 != 0:
            new_h = (height // 8) * 8
            new_w = (width // 8) * 8
            gray = F.interpolate(gray.unsqueeze(1), size=(new_h, new_w), 
                               mode='bilinear', align_corners=False).squeeze(1)
        
        # Divide into 8x8 blocks
        blocks = gray.unfold(1, 8, 8).unfold(2, 8, 8)  # [N, H//8, W//8, 8, 8]
        blocks = blocks.contiguous().view(batch_size, -1, 8, 8)  # [N, num_blocks, 8, 8]
        
        # Apply DCT
        D = self.dct_matrix
        D_t = D.transpose(0, 1)
        blocks_reshaped = blocks.view(-1, 8, 8)
        dct_blocks = D @ blocks_reshaped @ D_t
        dct_blocks = dct_blocks.view(batch_size, -1, 8, 8)
        
        # Apply frequency weights
        freq_weights_2d = self.freq_weights.view(8, 8)
        enhanced_dct = dct_blocks * freq_weights_2d.unsqueeze(0).unsqueeze(0)
        
        # Inverse DCT
        enhanced_blocks = D_t @ enhanced_dct.view(-1, 8, 8) @ D
        enhanced_blocks = enhanced_blocks.view(batch_size, -1, 8, 8)
        
        # Reconstruct image
        num_blocks_h = gray.shape[1] // 8
        num_blocks_w = gray.shape[2] // 8
        enhanced_gray = enhanced_blocks.view(batch_size, num_blocks_h, num_blocks_w, 8, 8)
        enhanced_gray = enhanced_gray.permute(0, 1, 3, 2, 4).contiguous()
        enhanced_gray = enhanced_gray.view(batch_size, num_blocks_h * 8, num_blocks_w * 8)
        
        # Resize back to original size if needed
        if enhanced_gray.shape[1] != height or enhanced_gray.shape[2] != width:
            enhanced_gray = F.interpolate(enhanced_gray.unsqueeze(1), size=(height, width),
                                        mode='bilinear', align_corners=False).squeeze(1)
        
        # Apply enhancement to all channels
        enhancement_factor = enhanced_gray.unsqueeze(1) / (gray.unsqueeze(1) + 1e-8)
        enhanced_residuals = residuals * enhancement_factor
        
        return enhanced_residuals


class SpatialAttention(nn.Module):
    """
    Spatial attention module to focus on important regions.
    """
    
    def __init__(self, channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // 4, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(channels // 4, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention to input tensor.
        
        Args:
            x: Input tensor of shape (N, C, H, W)
            
        Returns:
            Attention-weighted tensor of same shape
        """
        # Generate attention map
        att = F.relu(self.conv1(x))
        att = self.sigmoid(self.conv2(att))  # Shape: (N, 1, H, W)
        
        # Apply attention
        return x * att



