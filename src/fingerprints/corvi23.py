"""
Corvi23 fingerprint extraction methods implementation.

Based on "Intriguing Properties of Synthetic Images: from Generative Adversarial Networks 
to Diffusion Models" by Corvi et al. (2023).

This module implements two fingerprinting methods:
1. corvi23r: Residual-based fingerprint using autocorrelation of noise residuals
2. corvi23s: Spectral statistics fingerprint using radial and angular spectra

Both methods are implemented as fully differentiable explicit fingerprint methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np
from typing import Optional, Tuple
import math

from .base import FingerprintExtractor, AnalyticApproximation


class GaussianBlur2D(nn.Module):
    """
    2D Gaussian blur layer for differentiable denoising approximation.
    """
    
    def __init__(self, kernel_size: int = 15, sigma: float = 2.0, channels: int = 1):
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


@FingerprintExtractor.register("corvi23r")
class Corvi23R(FingerprintExtractor, nn.Module):
    """
    Corvi23-R: Residual-based fingerprint using autocorrelation of noise residuals.
    
    This is a fully differentiable implementation that uses Gaussian blur as a differentiable
    approximation of DnCNN denoising.
    
    The fingerprint is the autocorrelation of the noise residual of the image:
    1. Convert RGB to grayscale
    2. Denoise using differentiable Gaussian blur (approximation of DnCNN)
    3. Compute residual: residual = grayscale - denoised
    4. Compute 2D FFT: F = FFT2(residual)
    5. Compute power spectrum: S = abs(F) ** 2
    6. Compute autocorrelation: R = IFFT2(S).real, normalized by R[0,0]
    7. Crop 65×65 center patch around lag (0,0)
    """
    
    def __init__(self, kernel_size: int = 15, sigma: float = 2.0, device: str = None):
        """
        Initialize Corvi23-R extractor.
        
        Args:
            kernel_size: Size of Gaussian blur kernel for denoising
            sigma: Standard deviation for Gaussian blur
            device: Device to run on ("cuda", "cpu", "mps", or None for auto)
        """
        # Initialize nn.Module first
        nn.Module.__init__(self)
        
        # Initialize FingerprintExtractor
        FingerprintExtractor.__init__(
            self,
            method_name="corvi23r",
            is_differentiable=True,  # Fully differentiable implementation
            has_analytic_approx=False,  # No approximation needed as it's already differentiable
            feature_dim=4225,  # 65 × 65 = 4225
            is_implicit_fingerprint=False  # This is an explicit fingerprint method
        )
        
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
        # Differentiable denoiser (Gaussian blur as approximation of DnCNN)
        self.denoiser = GaussianBlur2D(kernel_size=kernel_size, sigma=sigma, channels=1)
        
        # RGB to grayscale weights
        self.register_buffer('rgb_weights', torch.tensor([0.2989, 0.5870, 0.1140]))
        
        # Move the model to the specified device
        self.to(self.device)
    
    def extract_fingerprint(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract residual-based fingerprints from images.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Fingerprint features tensor of shape (N, 4225)
        """
        batch_size = images.shape[0]
        
        # Ensure images are in correct range [0, 1]
        images = self.preprocess_images(images)
        
        # Move images to the correct device
        images = images.to(self.device)
        
        # Convert RGB to grayscale
        grayscale = torch.sum(images * self.rgb_weights.view(1, 3, 1, 1), dim=1, keepdim=True)  # (N, 1, H, W)
        
        # Denoise using differentiable Gaussian blur
        denoised = self.denoiser(grayscale)  # (N, 1, H, W)
        
        # Compute residual
        residual = grayscale - denoised  # (N, 1, H, W)
        
        all_features = []
        
        for i in range(batch_size):
            # Compute 2D FFT
            fft_residual = torch.fft.fft2(residual[i, 0])  # (H, W)
            
            # Compute power spectrum
            power_spectrum = torch.abs(fft_residual) ** 2  # (H, W)
            
            # Compute autocorrelation (IFFT of power spectrum)
            autocorr = torch.fft.ifft2(power_spectrum).real  # (H, W)
            
            # Normalize by autocorr[0, 0]
            autocorr = autocorr / (autocorr[0, 0] + 1e-8)
            
            # Crop 65×65 center patch
            h, w = autocorr.shape
            center_h, center_w = h // 2, w // 2
            start_h = center_h - 32
            end_h = center_h + 33
            start_w = center_w - 32
            end_w = center_w + 33
            
            # Handle edge cases for different image sizes
            if h < 65 or w < 65:
                # Pad if image is too small
                pad_h = max(0, 65 - h)
                pad_w = max(0, 65 - w)
                autocorr = F.pad(autocorr, (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2))
                h, w = autocorr.shape
                center_h, center_w = h // 2, w // 2
                start_h = center_h - 32
                end_h = center_h + 33
                start_w = center_w - 32
                end_w = center_w + 33
            
            fingerprint = autocorr[start_h:end_h, start_w:end_w]  # (65, 65)
            
            # Flatten to vector
            fingerprint_flat = fingerprint.flatten()  # (4225,)
            
            all_features.append(fingerprint_flat)
        
        return torch.stack(all_features)
    
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


@FingerprintExtractor.register("corvi23s")
class Corvi23S(FingerprintExtractor, nn.Module):
    """
    Corvi23-S: Spectral statistics fingerprint using radial and angular spectra.
    
    This is a fully differentiable implementation that uses soft binning for radial
    and angular averaging.
    
    The fingerprint is composed of two 1D summaries of the image's frequency content:
    1. Convert RGB to grayscale
    2. Compute 2D FFT: F = FFT2(grayscale)
    3. Compute power spectrum: S = abs(F) ** 2, normalized by its mean
    4. Compute radial spectrum: 128-bin annular average over spatial frequencies [0, 0.5]
    5. Compute angular spectrum: 16-bin directional average over angles in [0, π), 
       using high-pass filtered spectrum (cutoff freq = 0.1)
    """
    
    def __init__(self, radial_bins: int = 128, angular_bins: int = 16, 
                 cutoff_freq: float = 0.1, temperature: float = 0.1, device: str = None):
        """
        Initialize Corvi23-S extractor.
        
        Args:
            radial_bins: Number of bins for radial spectrum
            angular_bins: Number of bins for angular spectrum
            cutoff_freq: High-pass filter cutoff frequency
            temperature: Temperature for soft binning
            device: Device to run on ("cuda", "cpu", "mps", or None for auto)
        """
        # Initialize nn.Module first
        nn.Module.__init__(self)
        
        # Initialize FingerprintExtractor
        FingerprintExtractor.__init__(
            self,
            method_name="corvi23s",
            is_differentiable=True,  # Fully differentiable implementation
            has_analytic_approx=False,  # No approximation needed as it's already differentiable
            feature_dim=radial_bins + angular_bins,  # 128 + 16 = 144
            is_implicit_fingerprint=False  # This is an explicit fingerprint method
        )
        
        self.radial_bins = radial_bins
        self.angular_bins = angular_bins
        self.cutoff_freq = cutoff_freq
        self.temperature = temperature
        
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
        # RGB to grayscale weights
        self.register_buffer('rgb_weights', torch.tensor([0.2989, 0.5870, 0.1140]))
        
        # Move the model to the specified device
        self.to(self.device)
    
    def extract_fingerprint(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract spectral statistics fingerprints from images.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Fingerprint features tensor of shape (N, 144)
        """
        batch_size = images.shape[0]
        
        # Ensure images are in correct range [0, 1]
        images = self.preprocess_images(images)
        
        # Move images to the correct device
        images = images.to(self.device)
        
        # Convert RGB to grayscale
        grayscale = torch.sum(images * self.rgb_weights.view(1, 3, 1, 1), dim=1)  # (N, H, W)
        
        all_features = []
        
        for i in range(batch_size):
            # Compute 2D FFT
            fft_grayscale = torch.fft.fft2(grayscale[i])  # (H, W)
            
            # Compute power spectrum
            power_spectrum = torch.abs(fft_grayscale) ** 2  # (H, W)
            
            # Normalize by mean
            power_spectrum = power_spectrum / (torch.mean(power_spectrum) + 1e-8)
            
            # Compute radial spectrum using soft binning
            radial_spectrum = self._compute_radial_spectrum_soft(power_spectrum)
            
            # Compute angular spectrum using soft binning
            angular_spectrum = self._compute_angular_spectrum_soft(power_spectrum)
            
            # Concatenate features
            fingerprint = torch.cat([radial_spectrum, angular_spectrum])  # (144,)
            
            all_features.append(fingerprint)
        
        return torch.stack(all_features)
    
    def _compute_radial_spectrum_soft(self, power_spectrum: torch.Tensor) -> torch.Tensor:
        """
        Compute radial spectrum using soft binning for differentiability.
        
        Args:
            power_spectrum: Power spectrum tensor of shape (H, W)
            
        Returns:
            Radial spectrum tensor of shape (radial_bins,)
        """
        h, w = power_spectrum.shape
        
        # Create frequency grid
        freq_y = torch.fft.fftfreq(h, device=power_spectrum.device)
        freq_x = torch.fft.fftfreq(w, device=power_spectrum.device)
        
        # Create 2D frequency grid
        freq_y_grid, freq_x_grid = torch.meshgrid(freq_y, freq_x, indexing='ij')
        
        # Compute radial frequencies (distance from center)
        radial_freq = torch.sqrt(freq_x_grid**2 + freq_y_grid**2)
        
        # Create radial bins
        max_freq = 0.5  # Maximum frequency to consider
        bin_centers = torch.linspace(0, max_freq, self.radial_bins, device=power_spectrum.device)
        
        # Compute soft assignments to bins
        radial_spectrum = torch.zeros(self.radial_bins, device=power_spectrum.device)
        
        for i in range(self.radial_bins):
            # Soft assignment using Gaussian kernel
            distances = torch.abs(radial_freq - bin_centers[i])
            weights = torch.exp(-distances**2 / (2 * self.temperature**2))
            weights = weights / (weights.sum() + 1e-8)  # Normalize
            
            # Weighted average
            radial_spectrum[i] = torch.sum(power_spectrum * weights)
        
        return radial_spectrum
    
    def _compute_angular_spectrum_soft(self, power_spectrum: torch.Tensor) -> torch.Tensor:
        """
        Compute angular spectrum using soft binning for differentiability.
        
        Args:
            power_spectrum: Power spectrum tensor of shape (H, W)
            
        Returns:
            Angular spectrum tensor of shape (angular_bins,)
        """
        h, w = power_spectrum.shape
        
        # Create frequency grid
        freq_y = torch.fft.fftfreq(h, device=power_spectrum.device)
        freq_x = torch.fft.fftfreq(w, device=power_spectrum.device)
        
        # Create 2D frequency grid
        freq_y_grid, freq_x_grid = torch.meshgrid(freq_y, freq_x, indexing='ij')
        
        # Compute radial frequencies
        radial_freq = torch.sqrt(freq_x_grid**2 + freq_y_grid**2)
        
        # Apply high-pass filter (soft threshold)
        high_pass_mask = torch.sigmoid((radial_freq - self.cutoff_freq) / self.temperature)
        filtered_spectrum = power_spectrum * high_pass_mask
        
        # Compute angles (atan2 returns values in [-π, π], we want [0, π))
        angles = torch.atan2(freq_y_grid, freq_x_grid)
        angles = torch.where(angles < 0, angles + math.pi, angles)  # Map to [0, π)
        
        # Create angular bins
        bin_centers = torch.linspace(0, math.pi, self.angular_bins, device=power_spectrum.device)
        
        # Compute soft assignments to bins
        angular_spectrum = torch.zeros(self.angular_bins, device=power_spectrum.device)
        
        for i in range(self.angular_bins):
            # Soft assignment using Gaussian kernel
            distances = torch.abs(angles - bin_centers[i])
            weights = torch.exp(-distances**2 / (2 * self.temperature**2))
            weights = weights * high_pass_mask  # Apply high-pass filter
            weights = weights / (weights.sum() + 1e-8)  # Normalize
            
            # Weighted average
            angular_spectrum[i] = torch.sum(filtered_spectrum * weights)
        
        return angular_spectrum
    
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


# Placeholder analytic approximations for compatibility
class Corvi23R_Approx(AnalyticApproximation):
    """
    Placeholder analytic approximation for Corvi23-R.
    
    Since Corvi23-R is already differentiable, this approximation is not needed
    for W1 attacks (which can be done directly). This class is kept for compatibility
    with the framework.
    """
    
    def __init__(self, original_extractor: Corvi23R):
        """
        Initialize Corvi23-R approximation.
        
        Args:
            original_extractor: Original Corvi23-R extractor
        """
        super().__init__(original_extractor)
    
    def extract_fingerprint_approx(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract approximate Corvi23-R fingerprints.
        
        Since Corvi23-R is already differentiable, this just calls the original method.
        
        Args:
            images: Input images tensor
            
        Returns:
            Fingerprint features tensor
        """
        return self.original_extractor.extract_fingerprint(images)


class Corvi23S_Approx(AnalyticApproximation):
    """
    Placeholder analytic approximation for Corvi23-S.
    
    Since Corvi23-S is already differentiable, this approximation is not needed
    for W1 attacks (which can be done directly). This class is kept for compatibility
    with the framework.
    """
    
    def __init__(self, original_extractor: Corvi23S):
        """
        Initialize Corvi23-S approximation.
        
        Args:
            original_extractor: Original Corvi23-S extractor
        """
        super().__init__(original_extractor)
    
    def extract_fingerprint_approx(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract approximate Corvi23-S fingerprints.
        
        Since Corvi23-S is already differentiable, this just calls the original method.
        
        Args:
            images: Input images tensor
            
        Returns:
            Fingerprint features tensor
        """
        return self.original_extractor.extract_fingerprint(images)
