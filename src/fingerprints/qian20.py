"""
Qian20 (F3-Net) fingerprint extraction method implementation.

Based on "Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues"
by Qian et al. (2020).

This method uses frequency-domain analysis to detect forged faces through:
1. Frequency-aware Decomposition (FAD): DCT-based image decomposition into frequency bands
2. Local Frequency Statistics (LFS): Local frequency statistics from sliding window DCT

The fingerprint captures frequency-domain artifacts that are often invisible in RGB space
but evident in frequency spectra, making it effective for detecting face forgeries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List

from .base import FingerprintExtractor, AnalyticApproximation


def DCT_mat(size: int) -> np.ndarray:
    """Generate DCT matrix for given size."""
    m = [[(np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * 
          np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] 
         for i in range(size)]
    return np.array(m)


def generate_filter(start: int, end: int, size: int) -> np.ndarray:
    """Generate frequency filter mask."""
    return np.array([[0. if i + j > end or i + j <= start else 1. 
                     for j in range(size)] for i in range(size)])


def norm_sigma(x: torch.Tensor) -> torch.Tensor:
    """Normalize learnable parameters to [-1, 1] range."""
    return 2. * torch.sigmoid(x) - 1.


class Filter(nn.Module):
    """Learnable frequency filter for FAD."""
    
    def __init__(self, size: int, band_start: int, band_end: int, 
                 use_learnable: bool = True, norm: bool = False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable
        
        # Base filter (fixed)
        self.base = nn.Parameter(
            torch.tensor(generate_filter(band_start, band_end, size), dtype=torch.float32), 
            requires_grad=False
        )
        
        if self.use_learnable:
            # Learnable filter component
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
        
        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(
                torch.sum(torch.tensor(generate_filter(band_start, band_end, size), dtype=torch.float32)), 
                requires_grad=False
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base
        
        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


class FAD_Head(nn.Module):
    """Frequency-aware Decomposition head."""
    
    def __init__(self, size: int):
        super(FAD_Head, self).__init__()
        self.size = size
        
        # Initialize DCT matrices
        self._DCT_all = nn.Parameter(
            torch.tensor(DCT_mat(size), dtype=torch.float32), requires_grad=False
        )
        self._DCT_all_T = nn.Parameter(
            torch.transpose(torch.tensor(DCT_mat(size), dtype=torch.float32), 0, 1), 
            requires_grad=False
        )
        
        # Define frequency band filters
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1 || 0 - 1 (all frequencies)
        low_filter = Filter(size, 0, size // 16)
        middle_filter = Filter(size, size // 16, size // 8)
        high_filter = Filter(size, size // 8, size)
        all_filter = Filter(size, 0, size * 2)
        
        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply DCT to input
        x_freq = self._DCT_all @ x @ self._DCT_all_T  # [N, 3, size, size]
        
        # Apply 4 frequency band filters
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)  # [N, 3, size, size]
            y = self._DCT_all_T @ x_pass @ self._DCT_all  # [N, 3, size, size]
            y_list.append(y)
        
        # Concatenate all frequency components
        out = torch.cat(y_list, dim=1)  # [N, 12, size, size]
        return out


class LFS_Head(nn.Module):
    """Local Frequency Statistics head."""
    
    def __init__(self, size: int, window_size: int = 10, M: int = 6):
        super(LFS_Head, self).__init__()
        self.window_size = window_size
        self._M = M
        
        # Initialize DCT matrices for patches
        self._DCT_patch = nn.Parameter(
            torch.tensor(DCT_mat(window_size), dtype=torch.float32), requires_grad=False
        )
        self._DCT_patch_T = nn.Parameter(
            torch.transpose(torch.tensor(DCT_mat(window_size), dtype=torch.float32), 0, 1), 
            requires_grad=False
        )
        
        # Unfold operation for sliding window
        self.unfold = nn.Unfold(
            kernel_size=(window_size, window_size), 
            stride=2, 
            padding=4
        )
        
        # Initialize frequency band filters
        self.filters = nn.ModuleList([
            Filter(window_size, int(window_size * 2. / M * i), 
                   int(window_size * 2. / M * (i+1)), norm=True) 
            for i in range(M)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert RGB to grayscale
        x_gray = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        x = x_gray.unsqueeze(1)
        
        # Rescale to 0-255 range
        # If input is in [-1, 1] range, convert to [0, 255]
        # If input is in [0, 1] range, convert to [0, 255]
        if x.min() >= -1 and x.max() <= 1:
            x = (x + 1.) * 122.5
        elif x.min() >= 0 and x.max() <= 1:
            x = x * 255.
        else:
            # If input is already in [0, 255] range, keep as is
            pass
        
        # Calculate output size
        N, C, W, H = x.size()
        S = self.window_size
        size_after = int((W - S + 8) / 2) + 1
        
        # Sliding window unfold and DCT
        x_unfold = self.unfold(x)  # [N, C * S * S, L] where L is number of blocks
        L = x_unfold.size()[2]
        x_unfold = x_unfold.transpose(1, 2).reshape(N, L, C, S, S)  # [N, L, C, S, S]
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T
        
        # Apply M frequency band filters
        y_list = []
        for i in range(self._M):
            y = torch.abs(x_dct)
            y = torch.log10(y + 1e-15)
            y = self.filters[i](y)
            y = torch.sum(y, dim=[2, 3, 4])  # [N, L]
            y = y.reshape(N, size_after, size_after).unsqueeze(dim=1)  # [N, 1, size_after, size_after]
            y_list.append(y)
        
        out = torch.cat(y_list, dim=1)  # [N, M, size_after, size_after]
        return out


@FingerprintExtractor.register("qian20")
class Qian20(FingerprintExtractor, nn.Module):
    """
    Qian20 (F3-Net) complete attribution model using frequency-aware decomposition.
    
    This is an implicit fingerprint method where the F3-Net itself is the attribution model.
    The network combines:
    1. Frequency-aware Decomposition (FAD): DCT-based image decomposition
    2. Local Frequency Statistics (LFS): Local frequency statistics from sliding windows
    3. Classification head: Maps frequency features to model attribution
    
    This method is particularly effective for detecting face forgeries as it captures
    frequency-domain artifacts that are often invisible in RGB space.
    """
    
    def __init__(self, img_size: int = 256, LFS_window_size: int = 10, LFS_M: int = 6, 
                 num_classes: int = 2, hidden_dims: list = [512, 256], device: str = None):
        """
        Initialize Qian20 F3-Net attribution model.
        
        Args:
            img_size: Input image size (assumed square)
            LFS_window_size: Window size for local frequency statistics
            LFS_M: Number of frequency bands for LFS
            num_classes: Number of model classes for attribution
            hidden_dims: Hidden layer dimensions for classification head
            device: Device to run on ("cuda", "cpu", "mps", or None for auto)
        """
        nn.Module.__init__(self)
        
        self.img_size = img_size
        self.LFS_window_size = LFS_window_size
        self.LFS_M = LFS_M
        self.num_classes = num_classes
        
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
        # Calculate feature dimensions
        # FAD: 4 frequency bands * 3 channels = 12 channels
        fad_dim = 12 * img_size * img_size
        
        # LFS: M frequency bands * output_size * output_size
        lfs_output_size = int((img_size - LFS_window_size + 8) / 2) + 1
        lfs_dim = LFS_M * lfs_output_size * lfs_output_size
        
        feature_dim = fad_dim + lfs_dim
        
        FingerprintExtractor.__init__(
            self,
            method_name="qian20",
            is_differentiable=True,  # F3-Net is fully differentiable
            has_analytic_approx=False,  # No approximation needed as it's already differentiable
            feature_dim=feature_dim,
            is_implicit_fingerprint=True  # This is an implicit fingerprint method
        )
        
        # Initialize FAD and LFS heads
        self.fad_head = FAD_Head(img_size)
        self.lfs_head = LFS_Head(img_size, LFS_window_size, LFS_M)
        
        # Build classification head
        self.classification_head = self._build_classification_head(feature_dim, num_classes, hidden_dims)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Move the model to the specified device
        self.to(self.device)
    
    def _build_classification_head(self, input_dim: int, num_classes: int, hidden_dims: list) -> nn.Module:
        """
        Build classification head for attribution.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of model classes
            hidden_dims: Hidden layer dimensions
            
        Returns:
            Classification head module
        """
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                # Use GroupNorm instead of BatchNorm to avoid issues with small batch sizes
                nn.GroupNorm(min(32, hidden_dim // 4), hidden_dim) if hidden_dim >= 4 else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        return nn.Sequential(*layers)
    
    def extract_fingerprint(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract frequency-aware fingerprints from images.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Frequency-aware fingerprints tensor of shape (N, feature_dim)
        """
        batch_size = images.shape[0]
        
        # Ensure images are in correct range [-1, 1] and size
        original_min, original_max = images.min().item(), images.max().item()
        # If images are in [0, 1] range, convert to [-1, 1] range
        if images.min() >= 0 and images.max() <= 1:
            images = images * 2 - 1
        elif images.min() < -1 or images.max() > 1:
            # If images are outside expected range, normalize to [-1, 1]
            images = (images - images.min()) / (images.max() - images.min()) * 2 - 1
        # Debug: print image range if it's very different from expected
        if abs(original_min) > 0.1 or abs(original_max - 1) > 0.1:
            print(f"Qian20 extract_fingerprint: Image range adjusted from [{original_min:.3f}, {original_max:.3f}] to [{images.min().item():.3f}, {images.max().item():.3f}]")
        
        # Move images to the correct device
        images = images.to(self.device)
        
        # Resize images if needed
        if images.shape[2] != self.img_size or images.shape[3] != self.img_size:
            images = F.interpolate(images, size=(self.img_size, self.img_size), 
                                 mode='bilinear', align_corners=False)
        
        # Extract FAD features
        fad_features = self.fad_head(images)  # [N, 12, img_size, img_size]
        
        # Extract LFS features
        lfs_features = self.lfs_head(images)  # [N, LFS_M, lfs_output_size, lfs_output_size]
        
        # Flatten and concatenate features
        fad_flat = fad_features.view(batch_size, -1)  # [N, 12 * img_size * img_size]
        lfs_flat = lfs_features.view(batch_size, -1)  # [N, LFS_M * lfs_output_size * lfs_output_size]
        
        # Concatenate FAD and LFS features
        fingerprints = torch.cat([fad_flat, lfs_flat], dim=1)  # [N, feature_dim]
        
        return fingerprints
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for attribution prediction.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Attribution logits tensor of shape (N, num_classes)
        """
        # Extract frequency features
        fingerprints = self.extract_fingerprint(images)
        
        # Apply dropout
        fingerprints = self.dropout(fingerprints)
        
        # Pass through classification head
        logits = self.classification_head(fingerprints)
        
        return logits
    
    def predict_attribution(self, images: torch.Tensor) -> torch.Tensor:
        """
        Predict model attribution directly.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Attribution logits tensor of shape (N, num_classes)
        """
        # Use eval mode to disable dropout
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            logits = self.forward(images)
        
        # Restore training state
        if was_training:
            self.train()
        
        return logits
    
    def extract_fad_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract only FAD features for analysis.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            FAD features tensor of shape (N, 12, img_size, img_size)
        """
        # Ensure images are in correct range [-1, 1] and size
        # If images are in [0, 1] range, convert to [-1, 1] range
        if images.min() >= 0 and images.max() <= 1:
            images = images * 2 - 1
        elif images.min() < -1 or images.max() > 1:
            # If images are outside expected range, normalize to [-1, 1]
            images = (images - images.min()) / (images.max() - images.min()) * 2 - 1
        
        if images.shape[2] != self.img_size or images.shape[3] != self.img_size:
            images = F.interpolate(images, size=(self.img_size, self.img_size), 
                                 mode='bilinear', align_corners=False)
        
        return self.fad_head(images)
    
    def extract_lfs_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract only LFS features for analysis.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            LFS features tensor of shape (N, LFS_M, lfs_output_size, lfs_output_size)
        """
        # Ensure images are in correct range [-1, 1] and size
        # If images are in [0, 1] range, convert to [-1, 1] range
        if images.min() >= 0 and images.max() <= 1:
            images = images * 2 - 1
        elif images.min() < -1 or images.max() > 1:
            # If images are outside expected range, normalize to [-1, 1]
            images = (images - images.min()) / (images.max() - images.min()) * 2 - 1
        
        if images.shape[2] != self.img_size or images.shape[3] != self.img_size:
            images = F.interpolate(images, size=(self.img_size, self.img_size), 
                                 mode='bilinear', align_corners=False)
        
        return self.lfs_head(images)


class Qian20_Approx(AnalyticApproximation):
    """
    Analytic approximation for Qian20 fingerprint extraction.
    
    Since Qian20 is already differentiable and is an implicit fingerprint method,
    this approximation is not needed for W1 attacks (which can be done directly).
    This class is kept for compatibility with the framework.
    """
    
    def __init__(self, original_extractor: Qian20):
        """
        Initialize Qian20 approximation.
        
        Args:
            original_extractor: Original Qian20 extractor
        """
        super().__init__(original_extractor)
        # For implicit fingerprint methods, we can access the model directly
        self.attribution_model = original_extractor.get_attribution_model()
    
    def extract_fingerprint_approx(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract approximate frequency-aware fingerprints.
        
        For implicit fingerprint methods, this returns the attribution logits.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Attribution logits tensor of shape (N, num_classes)
        """
        # For implicit fingerprint methods, return attribution predictions
        return self.original_extractor.predict_attribution(images)
